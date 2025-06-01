
from dataloaders.data_loaders import IMDbDataModule
from enoders.blip import Blip2LanguageTransformer, Blip2VisionTransformer
from models.comm import CoMMCore
import math, torch, pytorch_lightning as pl
from torch import nn
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torchmetrics.classification import MultilabelF1Score
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
from sklearn.metrics import f1_score



    

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
        
        # Binarise predictions ։with a fixed threshold
        preds_bin = (test_preds > 0.5).astype(int)
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
            if step < warm:
                return (step+1)/warm
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
    accelerator: str = "gpu"  # Use "cuda" if GPU is free, otherwise "cpu"
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
    run_finetune()