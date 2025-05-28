# fine_tune_comm.py
# -----------------------------------------------------------
#   CoMM – MM-IMDb  |  Lightning fine-tuning (multi-label)
#   Loads a self-supervised checkpoint and trains a genre head
# -----------------------------------------------------------
import math, json, torch, pytorch_lightning as pl
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





# # ------------------------------------------------------------------
# # 1. Dataset (single view, no heavy work)
# # ------------------------------------------------------------------
class TextMasking(torch.nn.Module):
    """
        Randomly mask input tokens using a special `mask` token.
    """
    def __init__(self, mask_prob: float, mask_token_id: int, mask_ignored_ids: Optional[List[int]] = None) -> None:
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.mask_ignored_ids = mask_ignored_ids or [] # ignore these tokens for masking

    def _init_full_mask(self, seq: torch.Tensor) -> torch.Tensor:
        # Returns `True` for tokens to not ignore in `seq`
        full_mask = torch.full_like(seq, True, dtype=torch.bool)
        for ignored_id in self.mask_ignored_ids:
            full_mask &= (seq != ignored_id)
        return full_mask

    def _get_mask_subset_with_prob(self, mask: torch.Tensor) -> torch.Tensor:
        # Returns a subset of input `mask`
        random_mask = torch.rand(mask.shape, device=mask.device) < self.mask_prob
        mask &= random_mask
        return mask

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mask_prob == 0:
            return seq
        else:
            mask = self._init_full_mask(seq)
            mask = self._get_mask_subset_with_prob(mask)
            masked_seq = seq.clone().detach()
            masked_seq.masked_fill_(mask, self.mask_token_id)
            return masked_seq
        


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    




def get_unique_genres(json_file: str):
    with open(json_file, "r") as f:
        movies = json.load(f)
    unique = set()
    for m in movies:
        for g in m.get("genres", []):
            unique.add(g)
    return unique
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
from typing import List
from lavis.models import load_model



class Blip2VisionTransformer(nn.Module):
    """ BLIP2 pretrained vision encoder"""

    def __init__(self, output_value: str = 'token_embeddings'):
        super().__init__()
        assert output_value in {'embedding', 'token_embeddings'}
        self.output_value = output_value
        self.model = load_model(name="blip2_feature_extractor", model_type="pretrain")
        # Freeze all weights (no fine-tuning)
        for params in self.model.parameters():
            params.requires_grad = False

    def forward(self, x: torch.Tensor):
        features = self.model.extract_features(dict(image=x), mode="image")
        if self.output_value == "embedding":
            return features["image_embeds_proj"][:, 0, :] # shape (B, 256)
        else:
            return features["image_embeds"] # shape (B, L, 768)


class Blip2LanguageTransformer(nn.Module):
    """ BLIP-2 pretrained text encoder that implements random masking of input text.
    """

    def __init__(self,
                 output_value: str = 'token_embeddings',
                 mask_prob: float = 0.0
                 ):
        """
        :param output_value:  Default "token_embeddings", to get wordpiece token embeddings with shape (N, L, 768)
            where N == batch size, L == # tokens
            Can be set to "embedding" to get sentence embeddings with shape (N, 768).
        :param mask_prob: probability of randomly masking input tokens with mask tokens.
        """

        super().__init__()
        assert output_value in {"token_embeddings", "embedding"}

        self.model = load_model(name="blip2_feature_extractor", model_type="pretrain")
        # Feeze all weights (no fine-tuning)
        for params in self.model.parameters():
            params.requires_grad = False

        mask_ignore_token_ids = [self.model.tokenizer.pad_token_id,
                                 self.model.tokenizer.cls_token_id,
                                 self.model.tokenizer.sep_token_id]
        mask_token_id = self.model.tokenizer.mask_token_id
        self.mask = TextMasking(mask_prob, mask_token_id, mask_ignore_token_ids)
        self.output_value = output_value

    def forward(self, x: List[str]):
        text = self.model.tokenizer(x, return_tensors="pt",
                                    padding=True, truncation=True).to(self.model.device)
        text["input_ids"] = self.mask(text["input_ids"]).to(self.model.device)

        # return text features
        with torch.no_grad():
            text_output = self.model.Qformer.bert(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                )
        text_embeds = text_output.last_hidden_state
        attn_mask = text.attention_mask == 0   
          
        if self.output_value == "embedding":
            return text_embeds[:, 0, :]
        return {"token_embeddings": text_embeds,
                "padding_mask":    attn_mask}
    



def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor




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
    


from torchmetrics.classification import MultilabelF1Score
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

# ------------------------------------------------------------------
# 2. Dataset for supervised phase  (one clean view, + labels)
# ------------------------------------------------------------------
class IMDbSupDataset(Dataset):
    def __init__(self, json_path: str, genre2idx: dict[str, int], train=True):
        super().__init__()
        self.items = json.load(open(json_path))
        self.g2i   = genre2idx
        self.train = train

        norm = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        self.tf_train = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(), norm
        ])
        self.tf_test = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(), norm
        ])

    def _multi_hot(self, genres):
        y = torch.zeros(len(self.g2i))
        for g in genres:
            if g in self.g2i: y[self.g2i[g]] = 1.
        return y

    def __getitem__(self, idx):
        item  = self.items[idx]
        img   = Image.open(item["image_path"]).convert("RGB")
        img   = self.tf_train(img) if self.train else self.tf_test(img)
        plot  = item["plot"]
        label = self._multi_hot(item["genres"])
        return img, plot, label

    def __len__(self): return len(self.items)


class IMDbDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=8,
                 train_json="train.json", dev_json="dev.json",
                 test_json="test.json"):
        super().__init__()
        self.bs, self.nw = batch_size, num_workers
        genres = get_unique_genres(train_json)
        self.g2i = {g:i for i,g in enumerate(sorted(genres))}
        self.train_json, self.dev_json = train_json, dev_json
        self.test_json = test_json

    def setup(self, stage=None):
        self.train_set = IMDbSupDataset(self.train_json, self.g2i, train=True)
        self.val_set   = IMDbSupDataset(self.dev_json,  self.g2i, train=False)
        self.test_set  = IMDbSupDataset(self.test_json, self.g2i, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.bs, True,  num_workers=self.nw)

    def val_dataloader(self):
        return DataLoader(self.val_set,   self.bs, False, num_workers=self.nw)
    def test_dataloader(self):
        return DataLoader(self.test_set,  self.bs, False, num_workers=self.nw)

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
    
    

   
    
    