import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from src.utils.utils import all_gather_batch_with_grad


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
    
    