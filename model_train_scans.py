import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import dask.array as da
from tqdm import tqdm
from pathlib import Path
import os
import zarr
import s3fs


class Brain3DDataset(Dataset):
    def __init__(self, csv_path, label_map, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['Target_Cell_Population'].notnull() & (df['Target_Cell_Population'] != '')]
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s3_path = row['STPT Data File Path']
        label_str = row['Target_Cell_Population']

        if label_str not in self.label_map:
            raise KeyError(f"Label '{label_str}' not found in label_map")
        label = self.label_map[label_str]

        # AWS access (can be removed if using env vars/IAM roles)
        os.environ["AWS_ACCESS_KEY_ID"] = ""
        os.environ["AWS_SECRET_ACCESS_KEY"] = ""

        fs = s3fs.S3FileSystem(anon=True)
        store = fs.get_mapper(s3_path)
        root = zarr.open(store, mode='r')

        # Load image volume: shape (C, Z, H, W)
        img_np = root['7'][:]

        def normalize(img, clip=99.5):
            max_val = np.percentile(img, clip)
            if max_val < 1e-5:
                max_val = 1.0
            return np.clip(img / max_val, 0, 1)

        C, Z, H, W = img_np.shape
        rgb_volume = []

        for z in range(Z):
            r = normalize(img_np[0, z])
            g = normalize(img_np[1, z])
            b = normalize(img_np[2, z])

            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            green_threshold = np.percentile(g, 99)
            green_mask = g > green_threshold

            r_final = np.where(green_mask, 0.0, gray)
            g_final = np.where(green_mask, g, gray)
            b_final = np.where(green_mask, 0.0, gray)

            highlighted_rgb = np.stack([r_final, g_final, b_final], axis=0)  # (3, H, W)
            rgb_volume.append(highlighted_rgb)

        rgb_volume = np.stack(rgb_volume, axis=1)  # (3, Z, H, W)
        rgb_volume = torch.from_numpy(rgb_volume).float()

        if self.transform:
            rgb_volume = self.transform(rgb_volume)

        return rgb_volume, label



# Label mapping loader
def get_subclass_to_index_lookup(cache_path="subclass_to_index_isocortex.csv"):
    cache_path = Path(cache_path)
    print(f"Loading subclass_to_index from {cache_path}")
    df = pd.read_csv(cache_path)
    return dict(zip(df['subclass'], df['index']))


# === Label encoding ===
def get_label_map(csv_path):
    df = pd.read_csv(csv_path)
    labels = sorted(df['label'].unique())
    return {label: idx for idx, label in enumerate(labels)}

# === Model ===
def get_model(num_classes, pretrained=False, device='cuda'):
    model = r3d_18(pretrained=pretrained)
    # Adjust final layer for number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model


# === Training and validation ===
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_top1 = 0
    total = 0
    for imgs, labels in tqdm(loader, desc='Training', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)

        loss = F.cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)

        preds = outputs.argmax(dim=1)
        correct_top1 += (preds == labels).sum().item()

    return total_loss / len(loader), correct_top1 / total

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    correct_top1 = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Validating', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, labels)

            total_loss += loss.item()
            total += labels.size(0)

            preds = outputs.argmax(dim=1)
            correct_top1 += (preds == labels).sum().item()

    return total_loss / len(loader), correct_top1 / total

from torch.utils.data import DataLoader, random_split

def main():
    csv_path = "MapMySections_Training.csv"  # CSV with 'STPT Data File Path' and 'Target_Cell_Population'
    batch_size = 4
    epochs = 10
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load label mapping from subclass CSV (subclass -> index)
    label_map = get_subclass_to_index_lookup()
    num_classes = len(label_map)
    print(f"Number of classes: {num_classes}")

    # Create dataset (filters out rows with empty labels)
    dataset = Brain3DDataset(csv_path, label_map)

    # Train/validation split (80%-20%)
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = n - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # DataLoaders with multi-worker loading & pin_memory for faster GPU transfer
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Instantiate your 3D model (replace with your model)
    model = get_model(num_classes, pretrained=False, device=device)
    model.load_state_dict(torch.load("./resnet18/best_brainscan3d_model__isocortex.pth", map_location=device))
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_path = "./resnet18/best_brainscan3d_model__isocortex.pth"

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            print("Saving best model...")
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print("Training complete!")


if __name__ == "__main__":
    main()
