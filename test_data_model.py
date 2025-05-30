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
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s3_path = row['STPT Data File Path']

        # AWS access (can be removed if using env vars/IAM roles)

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

        return rgb_volume

# === Model ===
def get_model(num_classes, pretrained=False, device='cuda'):
    model = r3d_18(pretrained=pretrained)
    # Adjust final layer for number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model

# Label mapping loader
def get_subclass_to_index_lookup(cache_path="subclass_to_index_isocortex.csv"):
    cache_path = Path(cache_path)
    print(f"Loading subclass_to_index from {cache_path}")
    df = pd.read_csv(cache_path)
    return dict(zip(df['subclass'], df['index']))


def test_model(test_csv, model_path, subclass_csv, output_csv, batch_size=1, device='cuda'):
    # Load label map
    label_map = get_subclass_to_index_lookup(subclass_csv)
    index_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    # Load model
    model = get_model(num_classes=num_classes, pretrained=False, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test data
    dataset = Brain3DDataset(test_csv)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = []

    with torch.no_grad():
        for i, inputs in enumerate(tqdm(loader, desc="Testing")):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_classes = probs.topk(3, dim=1)

            for j in range(inputs.size(0)):
                results.append({
                    "MapMySectionsID": dataset.df.iloc[i * batch_size + j]['MapMySectionsID'],
                    "FilePath": dataset.df.iloc[i * batch_size + j]['STPT Data File Path'],
                    "top1_class": index_to_label.get(top_classes[j][0].item(), f"Unknown({top_classes[j][0].item()})"), 
                    "top1_prob": top_probs[j][0].item(),
                    "top2_class": index_to_label.get(top_classes[j][1].item(), f"Unknown({top_classes[j][1].item()})"),
                    "top2_prob": top_probs[j][1].item(),
                    "top3_class": index_to_label.get(top_classes[j][2].item(), f"Unknown({top_classes[j][2].item()})"),
                    "top3_prob": top_probs[j][2].item(),
                })


    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

# Example usage
if __name__ == "__main__":
    test_model(
        test_csv="MapMySections_TestData.csv",
        model_path="./resnet18/best_brainscan3d_model__isocortex.pth",
        subclass_csv="subclass_to_index_isocortex.csv",
        output_csv="test_predictions.csv",
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
