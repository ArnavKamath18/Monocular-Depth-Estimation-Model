import cv2
import numpy as np
import matplotlib.pyplot as plt

patch_h, patch_w = 8, 8

image_path = "sample.jpg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
h, w, _ = img.shape
h_patches = h // patch_h
w_patches = w // patch_w

r_mean_map = np.zeros((h_patches, w_patches))
g_mean_map = np.zeros((h_patches, w_patches))
b_mean_map = np.zeros((h_patches, w_patches))
row_norm_map = np.zeros((h_patches, w_patches))
col_norm_map = np.zeros((h_patches, w_patches))
center_dist_map = np.zeros((h_patches, w_patches))
focal_prior_map = np.zeros((h_patches, w_patches))
patch_grad_map = np.zeros((h_patches, w_patches))
patch_brightness_map = np.zeros((h_patches, w_patches))

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

center_y, center_x = h / 2, w / 2

for i in range(h_patches):
    for j in range(w_patches):
        y, x = i * patch_h, j * patch_w
        patch = img[y:y+patch_h, x:x+patch_w]
        patch_gray = gray[y:y+patch_h, x:x+patch_w]
        patch_grad = gradient_magnitude[y:y+patch_h, x:x+patch_w]

        r_mean_map[i, j] = np.mean(patch[:, :, 0])
        g_mean_map[i, j] = np.mean(patch[:, :, 1])
        b_mean_map[i, j] = np.mean(patch[:, :, 2])

        row_norm_map[i, j] = i / h_patches
        col_norm_map[i, j] = j / w_patches

        patch_center_y = y + patch_h / 2
        patch_center_x = x + patch_w / 2
        dist = np.sqrt((patch_center_y - center_y)**2 + (patch_center_x - center_x)**2)
        center_dist_map[i, j] = dist / np.sqrt(center_x**2 + center_y**2)

        focal_prior_map[i, j] = np.exp(-((patch_center_x - center_x)**2 + (patch_center_y - center_y)**2) / (2 * (0.25 * h)**2))

        patch_grad_map[i, j] = np.mean(patch_grad)

        patch_brightness_map[i, j] = np.mean(patch_gray)

center_dist_map /= (center_dist_map.max() + 1e-7)

fig, axes = plt.subplots(3, 3, figsize=(9, 3))

axes[0][0].imshow(img)
axes[0][0].set_title('Original Image', fontsize=8)
axes[0][0].axis('off')

axes[0][1].imshow(r_mean_map, cmap='Reds')
axes[0][1].set_title('R Mean Map', fontsize=8)
axes[0][1].axis('off')

axes[0][2].imshow(g_mean_map, cmap='Greens')
axes[0][2].set_title('G Mean Map', fontsize=8)
axes[0][2].axis('off')

axes[1][0].imshow(b_mean_map, cmap='Blues')
axes[1][0].set_title('B Mean Map', fontsize=8)
axes[1][0].axis('off')

axes[1][1].imshow(row_norm_map, cmap='viridis')
axes[1][1].set_title('Row Norm Map', fontsize=8)
axes[1][1].axis('off')

axes[1][2].imshow(col_norm_map, cmap='viridis')
axes[1][2].set_title('Col Norm Map', fontsize=8)
axes[1][2].axis('off')

axes[2][0].imshow(center_dist_map, cmap='magma')
axes[2][0].set_title('Center Dist Map', fontsize=8)
axes[2][0].axis('off')

axes[2][1].imshow(focal_prior_map, cmap='inferno')
axes[2][1].set_title('Focal Prior Map', fontsize=8)
axes[2][1].axis('off')

axes[2][2].imshow(patch_grad_map, cmap='gray')
axes[2][2].set_title('Gradient Magnitude Map', fontsize=8)
axes[2][2].axis('off')

plt.tight_layout()
plt.show()