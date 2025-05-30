import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
import torch
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache


# get cell type and coordinated xyz data
def get_cell_joined_data(abc_cache):
    # ==== load cell data ====
    cell = abc_cache.get_metadata_dataframe(directory='MERFISH-C57BL6J-638850', file_name='cell_metadata_with_cluster_annotation')
    cell.rename(columns={'x': 'x_section',
                        'y': 'y_section',
                        'z': 'z_section'},
                inplace=True)
    cell.set_index('cell_label', inplace=True)

    # ==== load reconst data ====
    reconstructed_coords = abc_cache.get_metadata_dataframe(
        directory='MERFISH-C57BL6J-638850-CCF',
        file_name='reconstructed_coordinates',
        dtype={"cell_label": str}
    )
    reconstructed_coords.rename(columns={'x': 'x_reconstructed',
                                        'y': 'y_reconstructed',
                                        'z': 'z_reconstructed'},
                                inplace=True)
    reconstructed_coords.set_index('cell_label', inplace=True)

    cell_joined = cell.join(reconstructed_coords, how='inner')

    # ==== load ccf data ====
    ccf_coords = abc_cache.get_metadata_dataframe(
        directory='MERFISH-C57BL6J-638850-CCF',
        file_name='ccf_coordinates',
        dtype={"cell_label": str}
    )
    ccf_coords.rename(columns={'x': 'x_ccf',
                            'y': 'y_ccf',
                            'z': 'z_ccf'},
                    inplace=True)
    ccf_coords.drop(['parcellation_index'], axis=1, inplace=True)
    ccf_coords.set_index('cell_label', inplace=True)

    cell_joined = cell_joined.join(ccf_coords, how='inner')

    # ==== load parcellation data ====
    parcellation_annotation = abc_cache.get_metadata_dataframe(directory='Allen-CCF-2020',
                                                            file_name='parcellation_to_parcellation_term_membership_acronym')
    parcellation_annotation.set_index('parcellation_index', inplace=True)
    parcellation_annotation.columns = ['parcellation_%s'% x for x in  parcellation_annotation.columns]

    parcellation_color = abc_cache.get_metadata_dataframe(directory='Allen-CCF-2020',
                                                        file_name='parcellation_to_parcellation_term_membership_color')
    parcellation_color.set_index('parcellation_index', inplace=True)
    parcellation_color.columns = ['parcellation_%s'% x for x in  parcellation_color.columns]

    cell_joined = cell_joined.join(parcellation_annotation, on='parcellation_index')
    cell_joined = cell_joined.join(parcellation_color, on='parcellation_index')

    isocortex_cell_joined = cell_joined[cell_joined['parcellation_division'] == 'Isocortex'].copy()
    return isocortex_cell_joined


def get_ccf_data(abc_cache, cell_joined):
    abc_cache.list_data_files('MERFISH-C57BL6J-638850-CCF')
    print("reading resampled_average_template")
    file = abc_cache.get_data_path(directory='MERFISH-C57BL6J-638850-CCF',
                                file_name='resampled_average_template')
    average_template_image = sitk.ReadImage(file)

    print("reading resampled_annotation_boundary")
    file = abc_cache.get_data_path(directory='MERFISH-C57BL6J-638850-CCF',
                                file_name='resampled_annotation_boundary')
    annotation_boundary_image = sitk.ReadImage(file)
    annotation_boundary_array = sitk.GetArrayViewFromImage(annotation_boundary_image)

    return average_template_image, annotation_boundary_array

# ==== plot func ====
def plot_section(xx=None, yy=None, cc=None, val=None, pcmap=None, 
                 overlay=None, extent=None, bcmap=plt.cm.Greys_r, alpha=1.0,
                 fig_width = 6, fig_height = 6):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_width, fig_height)

    if xx is not None and yy is not None and pcmap is not None:
        plt.scatter(xx, yy, s=0.5, c=val, marker='.', cmap=pcmap)
    elif xx is not None and yy is not None and cc is not None:
        plt.scatter(xx, yy, s=0.5, color=cc, marker='.', zorder=1)   
        
    if overlay is not None and extent is not None and bcmap is not None:
        plt.imshow(overlay, cmap=bcmap, extent=extent, alpha=alpha, zorder=2)
        
    ax.set_ylim(11, 0)
    ax.set_xlim(0, 11)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig, ax

def generate_tensor_images(section, target_subclass, boundary_slice, average_template_image):
    # calculate ccf overlay
    size = average_template_image.GetSize()
    spacing = average_template_image.GetSpacing()
    extent = (-0.5 * spacing[0], (size[0]-0.5) * spacing[0], (size[1]-0.5) * spacing[1], -0.5 * spacing[1])
    
    fluorescent_green = "#00FF00"
    section['highlight_color'] = np.where(
        section['subclass'] == target_subclass,
        fluorescent_green,
        "#B0B0B0"
    )
    fig, ax = plot_section(section['x_reconstructed'], section['y_reconstructed'], 
                          cc=section['highlight_color'], overlay=boundary_slice,
                          extent=extent, bcmap=plt.cm.Greys, alpha=1.0*(boundary_slice>0),
                          fig_width=6, fig_height=6)

    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1).convert('RGB')

    # Resize to smaller dimension for 3D stacking (change as needed)
    img = img.resize((224, 224))

    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)

    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)
    plt.close(fig)

    return img_tensor


def save_tensor_image(img_tensor, brain_section, subclass):
    # If batched, remove batch dimension: (1, 3, H, W) → (3, H, W)
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[0]

    # Convert to NumPy (H, W, 3)
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    # Optional: clip just in case
    img_np = np.clip(img_np, 0, 1)

    # Set target DPI
    dpi = 300
    height, width = img_np.shape[:2]
    figsize = (width / dpi, height / dpi)

    # Plot with high DPI
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img_np)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(f"./model_training_image_samples/{brain_section}__{subclass}.png", dpi=dpi, bbox_inches='tight', pad_inches=0)

    plt.close(fig)


# replaced by OnTheFlyTensorDataset __getitem__
def process_image_data():
    # ==== load atlas ====
    download_base = Path('../../data/abc_atlas')
    abc_cache = AbcProjectCache.from_cache_dir(download_base)

    cell_joined = get_cell_joined_data(abc_cache)
    average_template_image, annotation_boundary_array = get_ccf_data(abc_cache, cell_joined)

    image_tensors = []
    labels = []
    brain_section_labels = cell_joined['brain_section_label'].unique().tolist()

    i = 0
    for brain_section in brain_section_labels:
        pred = (cell_joined['brain_section_label'] == brain_section)
        section = cell_joined.loc[pred].copy()
        zindex = int(section.iloc[0]['z_reconstructed'] / 0.2)
        boundary_slice = annotation_boundary_array[zindex, :, :]

        # Define the subclass to highlight
        target_subclasses = section['subclass'].unique()

        for target_subclass in target_subclasses:
            img_tensor = generate_tensor_images(section, target_subclass, boundary_slice, average_template_image)
            image_tensors.append(img_tensor[0])  # remove batch dim
            labels.append(target_subclass)
            # save some samples
            if i % 10000 == 0:
                save_tensor_image(img_tensor, brain_section, target_subclass)
            i += 1
    return image_tensors, labels


def get_subclass_to_index_lookup(cell_joined, cache_path="subclass_to_index_isocortex.csv"):
    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"Loading subclass_to_index from {cache_path}")
        df = pd.read_csv(cache_path)
        return dict(zip(df['subclass'], df['index']))

    print("Generating subclass_to_index from cell_joined")
    label_encoder = LabelEncoder()
    label_encoder.fit(cell_joined['subclass'])

    subclass_to_index = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    df = pd.DataFrame(list(subclass_to_index.items()), columns=["subclass", "index"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)

    return subclass_to_index


# --- 3D Dataset Class ---

class Subclass3DVolumeDataset(Dataset):
    def __init__(self, cell_joined, average_template_image, boundary_array, label_encoder, transform=None):
        self.cell_joined = cell_joined
        self.atlas_img = average_template_image
        self.boundary_array = boundary_array
        self.label_encoder = label_encoder
        self.transform = transform

        # List unique subclasses
        self.subclasses = sorted(cell_joined['subclass'].unique())

        # Pre-group by subclass and brain_section_label for faster loading
        self.subclass_to_sections = defaultdict(list)
        for subclass in self.subclasses:
            df = cell_joined[cell_joined['subclass'] == subclass]
            groups = df.groupby('brain_section_label')
            for brain_section_label, group in groups:
                zindex = int(group.iloc[0]['z_reconstructed'] / 0.2)
                self.subclass_to_sections[subclass].append((zindex, group))

    def __len__(self):
        return len(self.subclasses)

    def __getitem__(self, idx):
        subclass = self.subclasses[idx]
        sections = self.subclass_to_sections[subclass]

        tensor_slices = []
        z_indices = []

        for zindex, section in sections:
            #if 0 <= zindex < self.boundary_array.shape[0]:
            boundary_slice = self.boundary_array[zindex, :, :]
            img_tensor = generate_tensor_images(section, subclass, boundary_slice, self.atlas_img)
            tensor_slices.append(img_tensor.squeeze(0))  # (3, H, W)
            z_indices.append(zindex)

        if not tensor_slices:
            raise ValueError(f"No valid slices for subclass: {subclass}")

        # Ensure tensor_slices is a list of individual slice tensors
        if isinstance(tensor_slices, torch.Tensor):
            tensor_slices = [tensor_slices[i] for i in range(tensor_slices.shape[0])]

        # z_indices already confirmed to be a list of ints
        sorted_pairs = sorted(zip(z_indices, tensor_slices), key=lambda x: x[0])
        sorted_slices = [t for _, t in sorted_pairs]
        
        volume_tensor = torch.stack(sorted_slices, dim=0).permute(1, 0, 2, 3)  # (3, D, H, W)

        if self.transform:
            volume_tensor = self.transform(volume_tensor)

        label_index = self.label_encoder[subclass]
        return volume_tensor, label_index


# --- Main execution setup ---

print('==== load atlas ====')
download_base = Path('../../data/abc_atlas')
abc_cache = AbcProjectCache.from_cache_dir(download_base)

cell_joined = get_cell_joined_data(abc_cache)
average_template_image, annotation_boundary_array = get_ccf_data(abc_cache, cell_joined)

subclass_to_index = get_subclass_to_index_lookup(cell_joined)

print('2. DataLoader Setup')
from sklearn.model_selection import StratifiedGroupKFold

metadata_df = cell_joined[['brain_section_label', 'subclass']].drop_duplicates()
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in sgkf.split(metadata_df, y=metadata_df['subclass'], groups=metadata_df['brain_section_label']):
    train_df = metadata_df.iloc[train_idx].reset_index(drop=True)
    val_df = metadata_df.iloc[val_idx].reset_index(drop=True)
    break

train_dataset = Subclass3DVolumeDataset(cell_joined, average_template_image, annotation_boundary_array, subclass_to_index)
val_dataset = Subclass3DVolumeDataset(cell_joined, average_template_image, annotation_boundary_array, subclass_to_index)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

print('3. Model Setup (ResNet18 adapted to 3D input)')

import torch.nn as nn
import torchvision.models as models

num_classes = len(subclass_to_index)

# Use 3D ResNet variant or adapt 2D model for 3D input

# Here's a simple trick: Replace first conv with Conv3d and adapt classifier
from torchvision.models.video import r3d_18

model = r3d_18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {device}")
model = model.to(device)

print('4. Training loop')

import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)

        loss = F.cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)

        probs = torch.softmax(outputs, dim=1)
        top_probs, top_classes = probs.topk(3, dim=1)

        correct_top1 += (top_classes[:, 0] == labels).sum().item()
        correct_top3 += (top_classes == labels.unsqueeze(1)).any(dim=1).sum().item()

    return total_loss / len(loader), correct_top1 / total, correct_top3 / total


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, labels)

            total_loss += loss.item()
            total += labels.size(0)

            probs = torch.softmax(outputs, dim=1)
            top_probs, top_classes = probs.topk(3, dim=1)

            correct_top1 += (top_classes[:, 0] == labels).sum().item()
            correct_top3 += (top_classes == labels.unsqueeze(1)).any(dim=1).sum().item()

    return total_loss / len(loader), correct_top1 / total, correct_top3 / total

# Example Training Loop
from torch.utils.data import DataLoader
from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=1e-4)
epochs = 10

best_val_acc = 0.0
best_model_path = "./resnet18/best_model_3D_isocortex.pth"

for epoch in range(epochs):
    print('train one epoch')
    train_loss, train_acc1, train_acc3 = train_one_epoch(model, train_loader, optimizer, device)
    print('validate')
    val_loss, val_acc1, val_acc3 = validate(model, val_loader, device)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Top-1 Acc: {train_acc1:.4f}, Top-3 Acc: {train_acc3:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, Top-1 Acc: {val_acc1:.4f}, Top-3 Acc: {val_acc3:.4f}")
    
    if val_acc1 > best_val_acc:
        print('4. Save the best model')
        best_val_acc = val_acc1
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model at epoch {epoch+1} with acc {best_val_acc:.4f}")
