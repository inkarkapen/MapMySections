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
import streamlit as st


# === Model ===
def get_model(num_classes, weights=None, device='cuda'):
    model = r3d_18(weights=weights)
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

def normalize(img, clip=99.5):
    max_val = np.percentile(img, clip)
    if max_val < 1e-5:
        max_val = 1.0
    return np.clip(img / max_val, 0, 1)

def load_preprocess_volume(s3_path):
    fs = s3fs.S3FileSystem(anon=True)
    store = fs.get_mapper(s3_path)
    root = zarr.open(store, mode='r')
    img_np = root['7'][:]  # (C, Z, Y, X)

    C, Z, Y, X = img_np.shape
    rgb_volume = []

    for z in range(Z):
        r = normalize(img_np[0, z])
        g = normalize(img_np[1, z])
        b = normalize(img_np[2, z])

        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        green_mask = g > np.percentile(g, 99)

        r_final = np.where(green_mask, 0.0, gray)
        g_final = np.where(green_mask, g, gray)
        b_final = np.where(green_mask, 0.0, gray)

        rgb = np.stack([r_final, g_final, b_final], axis=0)
        rgb_volume.append(rgb)

    rgb_volume = np.stack(rgb_volume, axis=1)  # (3, Z, H, W)
    tensor = torch.from_numpy(rgb_volume).float()
    return tensor

def predict_from_s3_path(s3_path, model, index_to_label, top_k=3, device='cuda'):
    model.eval()
    input_tensor = load_preprocess_volume(s3_path).to(device)

    with torch.no_grad():
        input_tensor = input_tensor.unsqueeze(0)
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_labels = probs.topk(top_k, dim=1)

    results = []
    for i in range(top_k):
        class_idx = top_labels[0][i].item()
        class_prob = top_probs[0][i].item()
        class_name = index_to_label[class_idx]
        results.append((class_name, class_prob))
    
    return results


# Load csv files
model_path="./resnet18/best_brainscan3d_model__isocortex.pth"
subclass_csv="subclass_to_index_isocortex.csv"
test_csv = "MapMySections_TestData.csv"
df = pd.read_csv(test_csv)

# Load labels look up table
label_map = get_subclass_to_index_lookup(subclass_csv)
index_to_label = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes, weights=None, device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

uploaded_model = st.file_uploader(model_path, type=["pth"])
if uploaded_model is not None:
    model = torch.load(uploaded_model, map_location="cpu")
    st.success("uploaded_model")

# SectionsId to s3 link look up
section_to_s3 = dict(zip(df['MapMySectionsID'], df['STPT Data File Path']))

# Streamlit app layout
st.title("Brain Scan 3D Section Inference")

selected_section = st.selectbox("Select MapMySectionsID", sorted(section_to_s3.keys()))

if selected_section:
    s3_path = section_to_s3[selected_section]
    st.write(f"Running inference on S3 path:\n{s3_path}")
    
    with st.spinner("Predicting..."):
        predictions = predict_from_s3_path(s3_path, model, index_to_label=index_to_label, top_k=3, device=device)
    
    st.subheader("Top Predictions:")
    for label, prob in predictions:
        st.write(f"**{label}**: {prob:.4f}")
