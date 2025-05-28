
# -----------------------------------------------------------
from torch.nn import BatchNorm1d
import os, math, json, random, torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from dataloaders.data_loaders import MMIMDBDataModule
# from enoders.blip import Blip2LanguageTransformer, Blip2VisionTransformer
from torchvision import transforms # type: ignore
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
import torch
import torch.nn as nn
from typing import List
from lavis.models import load_model
# from utils.utils import GaussianBlur

import json
import random
import torch
from typing import Optional, List

from PIL import ImageFilter 
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

import pytorch_lightning as pl  # type: ignore # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor  # type: ignore # noqa: E402
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger  # type: ignore # noqa: E402

Image.MAX_IMAGE_PIXELS = None

from collections import OrderedDict  # noqa: E402
from PIL import ImageFilter  # noqa: E402
import torch.autograd as autograd  # noqa: E402


from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
# from utils.utils import GaussianBlur, get_unique_genres
import pytorch_lightning as pl
from PIL import Image
import torch



def collate(batch):
    imgs_a, imgs_b, plots = [], [], []
    for (i_a, p_a), (i_b, p_b) in batch:
        imgs_a.append(i_a)
        imgs_b.append(i_b)
        plots.append((p_a, p_b))
    imgs_a = torch.stack(imgs_a)        # (B,C,H,W)
    imgs_b = torch.stack(imgs_b)
    plots_a, plots_b = zip(*plots)      # tuples of str
    return (imgs_a, list(plots_a)), (imgs_b, list(plots_b))


class MMIMDBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_json="train.json",
        dev_json="dev.json",
        batch_size=64,
        num_workers=8,
    ):
        super().__init__()
        self.train_json, self.dev_json = Path(train_json), Path(dev_json)
        self.batch_size = batch_size
        self.num_workers = num_workers
        genres = get_unique_genres(self.train_json)
        self.genre2idx = {g: i for i, g in enumerate(sorted(genres))}

    def setup(self, stage=None):
        self.train_ds = MMIMDBDataset(self.train_json, self.genre2idx,train=True)
        self.val_ds = MMIMDBDataset(self.dev_json, self.genre2idx, train=False)

    def train_dataloader(self):

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate
        )
    def val_dataloader(self):

        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate
        )
    



class MMIMDBDataset(Dataset):
    """I/O + lightweight transforms only – **no heavy encoders here**."""

    def __init__(
        self,
        json_path: str,
        genre2idx: dict[str, int],
        tokenizer_name: str = "Salesforce/blip2-itm-vit-g",
        max_length: int = 128,
        train=True
    ):
        super().__init__()
        self.data = json.load(open(json_path))
        self.genre2idx = genre2idx
        self.max_length = max_length
        self.train=train

        # ---- transforms -----------------------------------------------------
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        min_scale = 0.5  # same as “crop-0.08”; tweak if you’re varying s via img_augment
        self.simclr = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(min_scale, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
        
        # 2) Single-view “projection” (test) transform
        self.image_proc = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        self.train=train
            
    # ----- utils ------------------------------------------------------------
    def _multi_hot(self, genres):
        y = torch.zeros(len(self.genre2idx), dtype=torch.float32)
        for g in genres:
            if g in self.genre2idx:
                y[self.genre2idx[g]] = 1.0
        return y

    # -----------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        
        item = self.data[idx]

        # -------- images ---------------------------------------------------
        img = Image.open(item["image_path"]).convert("RGB")
        if self.train:
            img_a, img_b = self.simclr(img), self.simclr(img)
        else:
            img_a, img_b = self.image_proc(img), self.image_proc(img)


        # -------- text -----------------------------------------------------
        plot = item["plot"] 
        return  [img_a, plot], [img_b, plot]
    


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

    wandb_logger = WandbLogger(project="last_time", name="trial", log_model=True)

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
        devices=1,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=[ckpt, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=3,
    )

    trainer.fit(model, dm)
    print(f"\nBest checkpoint stored at: {ckpt.best_model_path}")


if __name__ == "__main__":
    main()