import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
import numpy as np
import pandas as pd
from pathlib import Path
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
    
# load
@st.cache_resource
def load_model(model_path, num_classes, device='cuda'):
    model = get_model(num_classes, weights=None, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
    
# Label mapping loader
@st.cache_data
def get_subclass_to_index_lookup(cache_path="subclass_to_index_isocortex.csv"):
    cache_path = Path(cache_path)
    print(f"Loading subclass_to_index from {cache_path}")
    df = pd.read_csv(cache_path)
    return dict(zip(df['subclass'], df['index']))

@st.cache_data
def get_s3_test_data_lookup(test_csv = "MapMySections_TestData.csv"):
    df = pd.read_csv(test_csv, usecols=["MapMySectionsID", "STPT Data File Path", "STPT Thumbnail Image"])
    return df
    
def normalize(img, clip=99.5):
    max_val = np.percentile(img, clip)
    if max_val < 1e-5:
        max_val = 1.0
    return np.clip(img / max_val, 0, 1)

def load_preprocess_volume(s3_path):
    fs = s3fs.S3FileSystem(anon=True)
    store = fs.get_mapper(s3_path)
    root = zarr.open(store, mode='r')
    img_np = root['8']  # (C, Z, Y, X)

    C, Z, Y, X = img_np.shape
    # Preallocate output array in float32
    rgb_volume = np.empty((C, Z, Y, X), dtype=np.float16)
    
    for z in range(Z):
        r = normalize(img_np[0, z])
        g = normalize(img_np[1, z])
        b = normalize(img_np[2, z])

        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        green_mask = g > np.percentile(g, 99)

        # Apply conditional logic directly into preallocated array
        rgb_volume[0, z] = np.where(green_mask, 0.0, gray)
        rgb_volume[1, z] = np.where(green_mask, g, gray)
        rgb_volume[2, z] = np.where(green_mask, 0.0, gray)

    tensor = torch.from_numpy(rgb_volume).float()
    return tensor

def predict_from_s3_path(s3_path, model, index_to_label, top_k=3, device='cuda'):
    input_tensor = load_preprocess_volume(s3_path)

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
#model_path="./resnet18/best_brainscan3d_model__isocortex.pth"
subclass_csv="subclass_to_index_isocortex.csv"
test_csv = "MapMySections_TestData.csv"
pre_calc_test_predictions = pd.read_csv("test_predictions.csv")
df = get_s3_test_data_lookup(test_csv)

# Load labels look up table
label_map = get_subclass_to_index_lookup(subclass_csv)
index_to_label = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

from huggingface_hub import hf_hub_download

# Replace with your actual user/repo and filename
model_path = hf_hub_download(
    repo_id="InkarK/ResNet_18_3D_Brain",
    filename="best_brainscan3d_model__isocortex.pth"
)

print("Model downloaded to:", model_path)

model = load_model(model_path, num_classes, device='cpu')
    
s3_path = None
thumbnail_url = None
selected_section = None

# Streamlit app layout
st.title("3D Brain Scan Cell Type Inference")
# Choose input mode
mode = st.radio("***Choose input method:***", ["Select from list of Test Data", "Enter custom S3 path"])

# If user wants to pick from the Test Data list
if mode == "Select from list of Test Data":
    selected_section = st.selectbox("***Select MapMySectionsID***", sorted(df["MapMySectionsID"].unique()))

    if selected_section:
        # Filter the row corresponding to the selected ID
        selected_row = df[df["MapMySectionsID"] == selected_section].iloc[0]
        s3_path = selected_row['STPT Data File Path']
        thumbnail_url = selected_row['STPT Thumbnail Image']

# If user wants to input their own s3 and hope for the best
elif mode == "Enter custom S3 path":
    s3_path = st.text_input("Enter your S3 path here (e.g., s3://your-bucket/path/to/zarr/)")

# If a valid S3 path is set, proceed with prediction
if s3_path:
    try:
        st.write(f"Running inference on S3 path:")
        st.write(f"{s3_path}")
                
        with st.spinner("Predicting..."):
            predictions = predict_from_s3_path(s3_path, model, index_to_label=index_to_label, top_k=3, device='cpu')

        # Check if the selected section exists in the DataFrame
        if selected_section and selected_section in pre_calc_test_predictions['MapMySectionsID'].values:
            # Filter the row corresponding to selected_section
            row = pre_calc_test_predictions[pre_calc_test_predictions['MapMySectionsID'] == selected_section].iloc[0]
        
            st.markdown("### Top Predictions (were pre-calculated using high resolution image on a gpu node):")
        
            # Extract top predictions
            precalc_predictions = [
                (row['top1_class'], row['top1_prob']),
                (row['top2_class'], row['top2_prob']),
                (row['top3_class'], row['top3_prob']),
            ]
        
            precalc_predictions_df = pd.DataFrame(precalc_predictions, columns=["Label", "Probability"])
            st.dataframe(precalc_predictions_df)
        
        # Show real time predicted labels 
        st.markdown("### Top Predictions (computed real-time using low resolution image):")
        predictions_df = pd.DataFrame(predictions, columns=["Label", "Probability"])
        st.dataframe(predictions_df)
        
        # Display the image from the 'STPT Thumbnail Image' column
        if pd.notna(thumbnail_url):
            st.subheader("Thumbnail Image:")
            st.image(thumbnail_url, caption=f"Thumbnail for {selected_section}",  width=500)
            
    except Exception as e:
        st.error(f"Failed to load S3 path: {s3_path}")
        st.error(f"{e}")
        st.error("Try a different S3 URL")
