# fine_tune_comm.py
# -----------------------------------------------------------
#   CoMM – MM-IMDb  |  Lightning fine-tuning (multi-label)
#   Loads a self-supervised checkpoint and trains a genre head
# -----------------------------------------------------------
from dataloaders.data_loaders import IMDbDataModule
from enoders.blip import Blip2LanguageTransformer, Blip2VisionTransformer
import math, torch, pytorch_lightning as pl
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torchmetrics.classification import MultilabelF1Score
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from typing import Optional, List, Tuple
import random
from PIL import ImageFilter
import torch.distributed as dist
import torch.autograd as autograd
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LRScheduler
from pytorch_lightning import Callback
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import warnings
import math
from collections import OrderedDict
from lavis.models import load_model
import numpy as np
from sklearn.metrics import f1_score

from src.utils.utils import all_gather_batch_with_grad, get_unique_genres



class FusionTransformer(nn.Module):
    def __init__(self, hidden=768, heads=8):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=4 * hidden,
            activation="gelu",
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=1)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden))

    def forward(self,
                img_tok, txt_tok,
                img_pad=None, txt_pad=None):   # padding masks (B , L)
        if img_tok is None and txt_tok is None:
            raise ValueError("Both modalities None")

        # --- build token sequence ---
        if img_tok is None:
            seq  = txt_tok
            mask = txt_pad
        elif txt_tok is None:
            seq  = img_tok
            mask = img_pad
        else:
            seq  = torch.cat([img_tok, txt_tok], dim=1)
            mask = torch.cat([img_pad, txt_pad], dim=1)

        # prepend CLS
        cls = self.cls.expand(seq.size(0), 1, -1)
        seq = torch.cat([cls, seq], 1)
        if mask is not None:
            zeros = torch.zeros(seq.size(0), 1, dtype=torch.bool, device=seq.device)
            mask = torch.cat([zeros, mask], 1)          # CLS never masked

        # forward
        z = self.enc(seq, src_key_padding_mask=mask)    # bool mask OK
        return z[:, 0] 

class CriticMLP(nn.Module):
    def __init__(self, in_dim: int, mlp_dim: int=512, out_dim: int=256):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim,   mlp_dim)),
            ("bn1",    nn.SyncBatchNorm(mlp_dim)),
            ("relu1",  nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim,  mlp_dim)),
            ("bn2",    nn.SyncBatchNorm(mlp_dim)),
            ("relu2",  nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim,  out_dim)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)





class CoMMCore(nn.Module):
    """Pure trainable core (fusion + critic) – no encoders inside."""
    def __init__(self, hidden=768, heads=8):
        super().__init__()
        self.fusion = FusionTransformer(hidden, heads)
        self.critic = CriticMLP(hidden, 512, 256)

    def fuse_project(self, img_pack, txt_pack):
        # img_pack / txt_pack  are tuples or None
        if img_pack is not None:
            img_tok, img_pad = img_pack
        else:
            img_tok, img_pad = None, None
        if txt_pack is not None:
            txt_tok, txt_pad = txt_pack
        else:
            txt_tok, txt_pad = None, None
        z = self.fusion(img_tok, txt_tok, img_pad, txt_pad)
        return self.critic(z)

    @staticmethod
    def comm_loss(zp, zpp, z1, z2, T=0.1):
        # 1) gather ALL embeddings from every GPU
        zp, zpp, z1, z2 = all_gather_batch_with_grad([zp, zpp, z1, z2])
        def infonce(a, b):
            N = a.size(0)
            a = F.normalize(a, dim=-1)
            b = F.normalize(b, dim=-1)
            # similarities
            sim_aa = (a @ a.T) / T
            sim_bb = (b @ b.T) / T
            sim_ab = (a @ b.T) / T
            # kill diag
            INF = 1e8
            sim_aa = sim_aa - torch.eye(N, device=a.device) * INF
            sim_bb = sim_bb - torch.eye(N, device=b.device) * INF
            # build 2N×2N log-softmax matrix
            top    = torch.cat([sim_ab, sim_aa], dim=1)  # [N, 2N]
            bottom = torch.cat([sim_bb, sim_ab.T], dim=1)
            simZ   = torch.cat([top, bottom], dim=0)     # [2N, 2N]
            logZ   = F.log_softmax(simZ, dim=1)
            # loss = −mean of the 2N "positive" positions
            return -torch.diag(logZ).mean()
        L_main = infonce(zp,  zpp)
        L_1    = 0.5 * (infonce(z1, zp)   + infonce(z1, zpp))
        L_2    = 0.5 * (infonce(z2, zp)   + infonce(z2, zpp))
        return (L_main + L_1 + L_2)/3
    


# ------------------------------------------------------------------
# 3. LightningModule for linear-probe or fine-tune
# ------------------------------------------------------------------
class CoMMGenre(pl.LightningModule):
    def __init__(self, ssl_ckpt: str,
                 lr=1e-3, final_lr=1e-5,
                 warmup_epochs=10, max_epochs=100,
                 num_labels=23):
        super().__init__()
        self.save_hyperparameters()

        # ---- load pretrained fusion block -----------------------------
        core = CoMMCore(hidden=768, heads=8)
        ssl_state = torch.load(ssl_ckpt, map_location="cpu")["state_dict"]
        core.load_state_dict({k.replace("core.",""):v
                              for k,v in ssl_state.items()
                              if k.startswith("core.")}, strict=True)
        self.fusion = core.fusion     # 1-layer, 8-head; critic unused

        # ---- linear classifier ---------------------------------------
        self.fusion.eval()
        self.head = nn.Linear(768, num_labels)

        # freeze backbone if linear probe
    
        for p in self.fusion.parameters(): 
            p.requires_grad = False

        # ---- frozen BLIP-2 encoders ----------------------------------
        self.vis_enc = Blip2VisionTransformer("token_embeddings")
        self.txt_enc = Blip2LanguageTransformer("token_embeddings", mask_prob=0.0)
        self.vis_enc.eval().requires_grad_(False)
        self.txt_enc.eval().requires_grad_(False)

        # ---- metrics --------------------------------------------------
        self.f1_macro   = MultilabelF1Score(num_labels, average="macro")
        self.f1_weight  = MultilabelF1Score(num_labels, average="weighted")
        
        # Store predictions and labels for test set
        self.test_preds = []
        self.test_labels = []

    # ----- encoding helper --------------------------------------------
    @torch.no_grad()
    def encode(self, img, plots):
        tok_i = self.vis_enc(img)                               # (B,32,768)
        pad_i = torch.zeros(tok_i.size(0), tok_i.size(1),
                            dtype=torch.bool, device=tok_i.device)

        txt = self.txt_enc(plots)
        tok_t, pad_t = txt["token_embeddings"], txt["padding_mask"]

        cls = self.fusion.cls.expand(img.size(0),1,-1)
        joint = torch.cat([cls, tok_i, tok_t], 1)
        mask  = torch.cat([torch.zeros_like(pad_i[:,:1]), pad_i, pad_t], 1)
        z = self.fusion.enc(joint, src_key_padding_mask=mask)[:,0]   # (B,768)
        return z

    # ----- common step -------------------------------------------------
    def _step(self, batch):
        img, plots, y = batch
        z = self.encode(img, plots)
        logits = self.head(z)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        return loss, logits.sigmoid(), y

    def training_step(self, batch, _):
        loss, preds, y = self._step(batch)
        self.log("train_loss", loss,
                 on_step=True,  on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        loss, preds, y = self._step(batch)
        self.log("val_loss", loss,
                 on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True)
        self.log("val_macroF1",  self.f1_macro(preds,y),
                 on_step=False, on_epoch=True,
                 prog_bar=True,  sync_dist=True)
        self.log("val_weightF1", self.f1_weight(preds,y),
                 on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True)

    def test_step(self, batch, _):
        loss, preds, y = self._step(batch)
        self.log("test_loss", loss)
        self.log("test_macroF1", self.f1_macro(preds, y), prog_bar=True)
        self.log("test_weightF1", self.f1_weight(preds, y), prog_bar=True)
        
        # Store predictions and labels for detailed analysis
        self.test_preds.extend(preds.cpu().numpy())
        self.test_labels.extend(y.cpu().numpy())

    def on_test_end(self):
        # Convert lists to numpy arrays
        test_preds = np.array(self.test_preds)
        test_labels = np.array(self.test_labels)
        
        # Binarise predictions with a fixed threshold
        preds_bin = (test_preds > 0.5).astype(int)

        # ---- F1 metrics -------------------------------------------------
        #   • per-class F1 (array of length num_labels)
        #   • macro F1  : unweighted mean across classes
        #   • weighted F1: class-wise F1 weighted by support
        per_class_f1 = f1_score(test_labels, preds_bin, average=None, zero_division=0)
        macro_f1     = f1_score(test_labels, preds_bin, average='macro',   zero_division=0)
        weighted_f1  = f1_score(test_labels, preds_bin, average='weighted', zero_division=0)
        
        # Print per-class F1 scores
        print("\nPer-class F1 scores:")
        for i, f1_val in enumerate(per_class_f1):
            print(f"Class {i}: {f1_val:.4f}")
        
        # Print aggregated metrics
        print("\nTest Set Results:")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")

        # Optionally, log to wandb if logger is available and is a WandbLogger
        if isinstance(self.logger, WandbLogger):
            exp = self.logger.experiment
            exp.log({"test_macro_f1_final": macro_f1, "test_weighted_f1_final": weighted_f1})
            for i, f1_val in enumerate(per_class_f1):
                exp.log({f"test_f1_class_{i}": f1_val})

    # ----- optimiser & schedule ---------------------------------------
    def configure_optimizers(self):
        train_params = filter(lambda p: p.requires_grad, self.parameters())
        opt = AdamW(train_params, lr=self.hparams.lr, weight_decay=1e-2)

        steps_per_epoch = self.trainer.estimated_stepping_batches // self.hparams.max_epochs
        warm = steps_per_epoch * self.hparams.warmup_epochs
        total= steps_per_epoch * self.hparams.max_epochs
        ratio= self.hparams.final_lr / self.hparams.lr

        def sched(step):
            if step < warm: return (step+1)/warm
            prog = (step-warm)/(total-warm)
            return ratio + (1-ratio)*0.5*(1+math.cos(math.pi*prog))

        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": LambdaLR(opt, sched),
                                 "interval":"step"}}

# ------------------------------------------------------------------
# 4. Run
# ------------------------------------------------------------------
def run_finetune(ssl_ckpt="/home/vazgen/CoMM_REIMPLEMENTATION/checkpoints/comm-epoch=95-val_loss=0.1121.ckpt"):
    # Reduce batch size
    dm = IMDbDataModule(batch_size=64)  # Reduced from 128 to 64
    model = CoMMGenre(ssl_ckpt)

    logger = WandbLogger(project="comm-mmimdb",
                         name="batch64")  # Updated name to reflect batch size change
    lrmon  = LearningRateMonitor("step")

    ckpt   = ModelCheckpoint(
        dirpath="checkpoints",
        filename="genre-{epoch:02d}-{val_macroF1:.4f}",
        monitor="val_macroF1",
        mode="max",
        save_top_k=1
    )

    # Clear GPU cache
    torch.cuda.empty_cache()

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        epochs=70,
        devices=1,
        precision=16,  # Enable mixed precision training
        callbacks=[lrmon, ckpt],
        logger=logger,
        log_every_n_steps=3,
    )
    
    # Train the model
    trainer.fit(model, dm)
    # 
    # Load best model and evaluate on test se


    ssl_ckpt   = "/home/vazgen/CoMM_REIMPLEMENTATION/checkpoints/comm-epoch=95-val_loss=0.1121.ckpt"
    genre_ckpt = "/home/vazgen/CoMM_REIMPLEMENTATION/checkpoints/last-v1.ckpt"

    # --- datamodule & model ---
    dm    = IMDbDataModule(batch_size=32)
    model = CoMMGenre.load_from_checkpoint(
           genre_ckpt,
           ssl_ckpt=ssl_ckpt,
           map_location="cpu")              #

    # --- tester ---
    trainer = pl.Trainer(accelerator="cpu")     #
    trainer.test(model, dm)
     

def test_finetuned_checkpoint(
    genre_ckpt: str,
    ssl_ckpt: str,
    batch_size: int = 32,
    train_json: str = "train.json",
    dev_json: str = "dev.json",
    test_json: str = "test.json",
    accelerator: str = "cpu"  # Use "cuda" if GPU is free, otherwise "cpu"
):
    """
    Loads a finetuned genre classifier checkpoint and evaluates it on the test set.
    """
    # 1. DataModule
    dm = IMDbDataModule(
        batch_size=batch_size,
        train_json=train_json,
        dev_json=dev_json,
        test_json=test_json
    )
    # 2. Model
    model = CoMMGenre.load_from_checkpoint(
        genre_ckpt,
        ssl_ckpt=ssl_ckpt,
        map_location=accelerator
    )
    # 3. Trainer
    trainer = pl.Trainer(accelerator=accelerator, devices=1)
    # 4. Test
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    test_finetuned_checkpoint(
        genre_ckpt="/home/vazgen/CoMM_REIMPLEMENTATION/checkpoints/genre-epoch=44-val_macroF1=0.5311.ckpt",
        ssl_ckpt="/home/vazgen/CoMM_REIMPLEMENTATION/checkpoints/comm-epoch=95-val_loss=0.1121.ckpt",
        batch_size=32,
        accelerator="cpu"  # or "cuda" if GPU is free
    )
    
    

   
    
    