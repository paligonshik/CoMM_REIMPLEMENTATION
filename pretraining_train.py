
# -----------------------------------------------------------

import math
import torch
from pathlib import Path
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from src.dataloaders.data_loaders import MMIMDBDataModule
from src.enoders.blip import Blip2LanguageTransformer, Blip2VisionTransformer
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import List
from lavis.models import load_model
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor  # type: ignore
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from src.utils.utils import all_gather_batch_with_grad  # type: ignore # noqa: E402
        
Image.MAX_IMAGE_PIXELS = None




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
            # loss = −mean of the 2N “positive” positions
            return -torch.diag(logZ).mean()
        L_main = infonce(zp,  zpp)
        L_1    = 0.5 * (infonce(z1, zp)   + infonce(z1, zpp))
        L_2    = 0.5 * (infonce(z2, zp)   + infonce(z2, zpp))
        return (L_main + L_1 + L_2)/3
    
    

# ------------------------------------------------------------------
# ---- 4. LightningModule ------------------------------------------
# ------------------------------------------------------------------

class CoMMLightning(pl.LightningModule):
    def __init__(
        self,
        hidden_dim=768,
        num_heads=8,
        lr=1e-4,
        final_lr=1e-6,
        warmup_epochs=10,
        total_epochs=100,
        train_len: int = 1,  # filled later by datamodule
        encoder_name: str = "Salesforce/blip2-itm-vit-g",
    ):
        super().__init__()
        self.core = CoMMCore(hidden_dim, num_heads)
        self.save_hyperparameters(ignore=["train_len"])
        self.train_len = train_len

        
        
        self.text_model=Blip2LanguageTransformer(output_value="token_embeddings",mask_prob=0.15)
        self.vision_model = Blip2VisionTransformer(output_value="token_embeddings")
        self.text_model.eval()              # keep enc frozen deterministic
        self.text_model.mask.train()        # still enables 15 % random masks
        self.text_model.requires_grad_(False)
        self.vision_model.requires_grad_(False)
    
    @torch.no_grad()
    def _embed_text(self, plots):
        out = self.text_model(plots)            # dict
        return out["token_embeddings"], out["padding_mask"]     
    @torch.no_grad()
    def _embed_img(self, img):
        z = self.vision_model(img)              # (B , 32 , 768)
        pad = torch.zeros(z.size(0), z.size(1), dtype=torch.bool, device=z.device)
        return z, pad                           # no padding inside 32 queries

    def forward(self, batch):
        # ----- encode -----------------------------------------------------
        img_a = self._embed_img(batch[0][0])
        img_b = self._embed_img(batch[1][0])
        img_only = img_b
        

        txt_a = self._embed_text(batch[0][1])
        txt_b = self._embed_text(batch[1][1])
        txt_only = txt_b

        zp  = self.core.fuse_project(img_a, txt_a)
        zpp = self.core.fuse_project(img_b, txt_b)
        z1  = self.core.fuse_project(img_only, None)
        z2  = self.core.fuse_project(None, txt_only)
        
        return zp, zpp, z1, z2

    # ===== steps =======================================================
    def training_step(self, batch, batch_idx):
        zp, zpp, z1, z2 = self(batch)
        loss = CoMMCore.comm_loss(zp, zpp, z1, z2)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("curr_loss", loss.detach(), on_step=True, logger=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        zp, zpp, z1, z2 = self(batch)
        loss = CoMMCore.comm_loss(zp, zpp, z1, z2)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ===== optimizer / scheduler ======================================
    def configure_optimizers(self):
        # trainable params = core only (encoders frozen)
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = AdamW(params, lr=self.hparams.lr, weight_decay=1e-2)
        total_steps = self.train_len * self.hparams.total_epochs
        warmup_steps = self.train_len * self.hparams.warmup_epochs
        ratio = self.hparams.final_lr / self.hparams.lr

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        sched = LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }



def main():
    pl.seed_everything(42, workers=True)

    dm = MMIMDBDataModule(batch_size=64, num_workers=8)
    dm.setup()
    train_len = len(dm.train_dataloader())

    model = CoMMLightning(
        hidden_dim=768,
        lr=1e-4,
        final_lr=1e-6,
        warmup_epochs=10,
        total_epochs=100,
        train_len=train_len,
    )

    wandb_logger = WandbLogger(project="last_time", name="final_shot", log_model=True)

    ckpt = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename="comm-{epoch:02d}-{val_loss:.4f}",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor("step")

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=[ckpt, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=3,
    )

    trainer.fit(model, dm)
    print(f"\nBest checkpoint stored at: {ckpt.best_model_path}")
