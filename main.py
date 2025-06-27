import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

image_path = "sample.jpg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#brightness
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

patch_h, patch_w = 8, 8
img_h, img_w = gray.shape

brightness_map = []
for y in range(0, img_h, patch_h):
    row = []
    for x in range(0, img_w, patch_w):
        patch = gray[y:y+patch_h, x:x+patch_w]
        brightness = np.mean(patch)
        row.append(brightness)
    brightness_map.append(row)

brightness_map = np.array(brightness_map)


#HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s_channel = hsv[:, :, 1]

img_h, img_w = s_channel.shape

saturation_map = []
for y in range(0, img_h, patch_h):
    row = []
    for x in range(0, img_w, patch_w):
        patch = s_channel[y:y+patch_h, x:x+patch_w]
        mean_saturation = np.mean(patch)
        row.append(mean_saturation)
    saturation_map.append(row)

saturation_map = np.array(saturation_map)


#shadow
darkness_map = []
for y in range(0, img_h, patch_h):
    row = []
    for x in range(0, img_w, patch_w):
        patch = gray[y:y+patch_h, x:x+patch_w]
        mean_brightness = np.mean(patch)
        darkness = 255 - mean_brightness
        row.append(darkness)
    darkness_map.append(row)

darkness_map = np.array(darkness_map)

edges = cv2.Canny(gray, threshold1=100, threshold2=200)
img_h, img_w = edges.shape

edge_density_map = []
for y in range(0, img_h, patch_h):
    row = []
    for x in range(0, img_w, patch_w):
        patch = edges[y:y+patch_h, x:x+patch_w]
        edge_pixels = np.sum(patch > 0)
        density = edge_pixels / (patch_h * patch_w)
        row.append(density)
    edge_density_map.append(row)

edge_density_map = np.array(edge_density_map)



sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

img_h, img_w = gradient_magnitude.shape

gradient_map = []
for y in range(0, img_h, patch_h):
    row = []
    for x in range(0, img_w, patch_w):
        patch = gradient_magnitude[y:y+patch_h, x:x+patch_w]
        mean_grad = np.mean(patch)
        row.append(mean_grad)
    gradient_map.append(row)

gradient_map = np.array(gradient_map)

lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

img_h,img_w = lbp.shape

texture_map = []
for y in range(0,img_h, patch_h):
    row = []
    for x in range(0, img_w, patch_w):
        patch = lbp[y:y+patch_h, x:x+patch_w]
        variance = np.var(patch)
        row.append(variance)
    texture_map.append(row)

texture_map = np.array(texture_map)


radius = 1
n_points = 8 * radius

def compute_lbp_map(gray, patch_size=8):
    height, width = gray.shape
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    h_patches = height // patch_size
    w_patches = width // patch_size
    lbp_map = np.zeros((h_patches, w_patches))
    for i in range(h_patches):
        for j in range(w_patches):
            patch = lbp[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            hist, _ = np.histogram(patch.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype(float) / (patch_size * patch_size)
            entropy = -np.sum(hist * np.log(hist + 1e-7))
            lbp_map[i, j] = entropy
    lbp_map /= lbp_map.max()
    lbp_map_resized = cv2.resize(lbp_map, (width, height), interpolation=cv2.INTER_LINEAR)
    return lbp_map_resized

texture_gradient_map = compute_lbp_map(gray, patch_size=8)

fig, axes = plt.subplots(2, 4, figsize=(15, 8))

axes[0][0].imshow(img)
axes[0][0].set_title('Original Image')
axes[0][0].axis('off')

axes[0][1].imshow(brightness_map, cmap='gray')
axes[0][1].set_title('Brightness Map')
axes[0][1].axis('off')

axes[0][2].imshow(saturation_map, cmap='inferno')
axes[0][2].set_title('Saturation Map')
axes[0][2].axis('off')

axes[0][3].imshow(darkness_map, cmap='gray')
axes[0][3].set_title('Darkness Map')
axes[0][3].axis('off')

axes[1][0].imshow(edge_density_map, cmap='gray')
axes[1][0].set_title('Edge Density Map')
axes[1][0].axis('off')

axes[1][1].imshow(gradient_map, cmap='gray')
axes[1][1].set_title('Gradient Magnitude Map')
axes[1][1].axis('off')

axes[1][2].imshow(texture_map, cmap='gray')
axes[1][2].set_title('Texture Map (Gabor)')
axes[1][2].axis('off')

axes[1][3].imshow(texture_gradient_map, cmap='inferno')
axes[1][3].set_title('Texture Map (LBP Entropy)')
axes[1][3].axis('off')

plt.tight_layout()
plt.show()