# file 337a_WSI_Nuclei_Torchvision_Instance.py

import openslide
from skimage.measure import label
import numpy as np
import cv2
import tifffile
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import matplotlib.pyplot as plt

# --- Load the Model ---
# We will use a pre-trained Mask R-CNN model from torchvision.
# The model will be automatically downloaded the first time you run this.
# You can use a custom trained model by loading its state_dict.

# Get the weights and model
weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = maskrcnn_resnet50_fpn_v2(weights=weights)

# Set the model to evaluation mode for inference
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Get the transforms from the pre-trained weights
preprocess = weights.transforms()

# --- Load the .svs file ---
slide = openslide.OpenSlide("/content/drive/MyDrive/ColabNotebooks/data/svs_files/sample.svs")
slide_dims = slide.dimensions
print(f"Slide dimensions: {slide_dims}")

# --- Segmenting the large image by applying prediction tile by tile ---
# Parameters
tile_size = (512, 512)
overlap = 50

# Initialize an empty array to store the full mask
full_mask_shape = (slide.level_dimensions[-1][1], slide.level_dimensions[-1][0])
full_mask = np.zeros(full_mask_shape, dtype=np.uint16)
max_label = 0

# Loop through the slide and extract tiles
for x in range(0, slide.level_dimensions[0][0], tile_size[0] - overlap):
    print(f"Processing column starting at x = {x}")
    for y in range(0, slide.level_dimensions[0][1], tile_size[1] - overlap):
        # Calculate the size of the tile to read
        w, h = tile_size
        if x + w > slide.level_dimensions[0][0]:
            w = slide.level_dimensions[0][0] - x
        if y + h > slide.level_dimensions[0][1]:
            h = slide.level_dimensions[0][1] - y

        # Extract the tile and convert to RGB
        tile = slide.read_region((x, y), 0, (w, h))
        tile_rgb = tile.convert("RGB")
        tile_array = np.array(tile_rgb)

        # Prepare the tile for the model and run prediction
        # The model expects input in a specific format, so we use the transforms
        # and convert the image to a tensor.
        input_tensor = preprocess(tile_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)

        # Process the outputs to get the masks
        masks = outputs[0]['masks'] > 0.5  # Apply a threshold to get binary masks
        scores = outputs[0]['scores']
        labels = outputs[0]['labels']

        # Filter masks based on a confidence score (e.g., 0.7) and the class label
        # The pre-trained model is for COCO dataset, we need to adapt it.
        # Here we will just use a score threshold as a starting point.
        # For custom models, you'd check for your specific nuclei class label.
        pred_masks = masks[scores > 0.7].squeeze(1).cpu().numpy()

        if pred_masks.size > 0:
            combined_mask = np.sum(pred_masks.astype(np.uint8), axis=0)
            labeled_mask = label(combined_mask)
            
            # Update labels to continue from where the last tile left off
            labeled_mask[labeled_mask > 0] += max_label
            max_label = labeled_mask.max()

            # Update the full mask (handle overlaps using max pooling)
            for i in range(h):
                for j in range(w):
                    if y + i < full_mask_shape[0] and x + j < full_mask_shape[1]:
                        full_mask[y + i, x + j] = max(full_mask[y + i, x + j], labeled_mask[i, j])

        print("Segmentation complete. Full mask updated.")

# Save the full mask as a 16-bit TIFF file
tifffile.imsave("/content/drive/MyDrive/ColabNotebooks/data/svs_files/torchvision_output_mask.tiff", full_mask.astype(np.uint16))

# --- Plotting input image and the segmented image ---
lowest_level = len(slide.level_dimensions) - 1
whole_slide_image = slide.read_region((0, 0), lowest_level, slide.level_dimensions[lowest_level])
whole_slide_image = whole_slide_image.convert("RGB")
whole_slide_image = np.array(whole_slide_image)

plt.figure(figsize=(10, 120))
plt.subplot(1, 2, 1)
plt.imshow(whole_slide_image)
plt.title('Whole Slide Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(full_mask, cmap='nipy_spectral')
plt.title('Full Mask')
plt.axis('off')
plt.show()