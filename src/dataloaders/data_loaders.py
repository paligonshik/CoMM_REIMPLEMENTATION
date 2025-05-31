from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.utils import GaussianBlur, get_unique_genres
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
        train_json="../../data/train.json",
        dev_json="../../data/dev.json",
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
    