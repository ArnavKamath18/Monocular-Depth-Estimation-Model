import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datasets import load_dataset
import random
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import torch
from scipy.ndimage import label
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

PATCH_H, PATCH_W = 8, 8
NUM_IMAGES = 100

def extract_features_and_target(sample):
    image = np.array(sample["image"]) 
    depth = np.array(sample["depth_map"])

    img_height, img_width = image.shape[:2]
    features, targets = [], []

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    for row in range(0, img_height - PATCH_H + 1, PATCH_H):
        for col in range(0, img_width - PATCH_W + 1, PATCH_W):
            patch = image[row:row+PATCH_H, col:col+PATCH_W]

            r_mean = np.mean(patch[:, :, 0])
            g_mean = np.mean(patch[:, :, 1])
            b_mean = np.mean(patch[:, :, 2])
            row_norm = row / (img_height - PATCH_H)
            col_norm = col / (img_width - PATCH_W)
            center_y = (row + PATCH_H / 2) - img_height / 2
            center_x = (col + PATCH_W / 2) - img_width / 2
            center_dist = np.sqrt(center_x**2 + center_y**2) / np.sqrt((img_width/2)**2 + (img_height/2)**2)
            focal_y_weight = 1.0 - row / img_height
            focal_x_weight = np.abs((col + PATCH_W/2) - img_width/2) / (img_width/2)
            focal_prior = focal_y_weight + focal_x_weight
            patch_grad = np.mean(gradient_magnitude[row:row+PATCH_H, col:col+PATCH_W])
            patch_brightness = np.mean(gray[row:row+PATCH_H, col:col+PATCH_W])

            feature_vector = [r_mean, g_mean, b_mean, row_norm, col_norm, center_dist, focal_prior, patch_grad, patch_brightness]

            features.append(feature_vector)

            patch_depth = depth[row:row+PATCH_H, col:col+PATCH_W]
            targets.append(np.mean(patch_depth))

    return np.array(features), np.array(targets)

dataset = load_dataset("sayakpaul/nyu_depth_v2", split="train", trust_remote_code=True)
indices = random.sample(range(len(dataset)), NUM_IMAGES)

X_all, y_all = [], []
for idx in indices:
    sample = dataset[idx]
    X, y = extract_features_and_target(sample)
    X_all.append(X)
    y_all.append(np.log1p(np.clip(y, 0, 10)))

X = np.vstack(X_all)
y = np.concatenate(y_all)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

model = LinearRegression()
model.fit(X_train_poly, y_train)
preds_log = model.predict(X_test_poly)
preds = np.expm1(preds_log)
y_true = np.expm1(y_test)

print("Test MSE:", mean_squared_error(y_true, preds))

model2 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model2.eval()

def visualize_prediction(idx):
    sample = dataset[idx]
    img = np.array(sample["image"]) 
    true_depth = np.array(sample["depth_map"])

    features, _ = extract_features_and_target(sample)
    features_scaled = scaler.transform(features)
    features_poly = poly.transform(features_scaled)
    pred_patches_log = model.predict(features_poly)
    pred_patches = np.expm1(pred_patches_log)

    h_patches = true_depth.shape[0] // PATCH_H
    w_patches = true_depth.shape[1] // PATCH_W
    pred_map = pred_patches.reshape(h_patches, w_patches)

    pred_resized = cv2.resize(pred_map, (true_depth.shape[1], true_depth.shape[0]), interpolation=cv2.INTER_CUBIC)
    pred_filtered = cv2.bilateralFilter(pred_resized.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

    min_depth = min(true_depth.min(), pred_filtered.min())
    max_depth = max(true_depth.max(), pred_filtered.max())
    denom = (max_depth - min_depth) + 1e-8
    true_normalized = (true_depth - min_depth) / denom
    pred_normalized = (pred_filtered - min_depth) / denom

    img_pil = Image.fromarray(img)
    img_tensor = F.to_tensor(img_pil)

    with torch.no_grad():
        outputs = model2([img_tensor])[0]

    mask_map = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    all_masks = outputs['masks'][:, 0].detach().cpu().numpy() > 0.5

    vis_img = img.copy()
    H, W, C = vis_img.shape
    for mask in all_masks:
        mask = mask.astype(bool)
        if np.any(mask):
            mask_map |= mask
            masked_pixels = vis_img[mask]
            darkest_color = masked_pixels.min(axis=0)
            for c in range(3):
                vis_img[:, :, c][mask] = darkest_color[c]

    labeled_mask, num_features = label(mask_map)
    pred2 = pred_filtered.copy()
    for i in range(1, num_features + 1):
        region = labeled_mask == i
        if np.any(region):
            region_depth = pred_filtered[region]
            closest_depth = region_depth.min()
            pred2[region] = closest_depth

    predfinal = cv2.addWeighted(pred_normalized, 0.9, pred2, 0.1, 0)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[1].imshow(true_normalized, cmap='plasma')
    axes[1].set_title("True Depth Map")
    axes[2].imshow(predfinal, cmap='plasma')
    axes[2].set_title("Predicted Depth Map")

    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

for i in random.sample(indices, 3):
    visualize_prediction(i)
