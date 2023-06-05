#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import skimage
import os
import cv2
from PIL import Image

from sklearn.mixture import GaussianMixture
from skimage import data, io, img_as_ubyte
from skimage.color import rgb2hed, hed2rgb, label2rgb
from skimage.exposure import rescale_intensity
import pyclesperanto_prototype as cle


# In[2]:


Image.MAX_IMAGE_PIXELS = None


# # Data preparation:

# The 'src.jpg' file is way too big to handle, it might even be hard using more advanced GPUs. I only have access to my CPU, so the first thing I did was to split 'src.jpg' into 1024 equivalent sized, 512x512 images. The created patches are much easier to deal with. The 'save_patches()' function is to sava the created mini-images into a new directory, so they can be accessed easily later on when needed.

# In[3]:


def split_image(image_path, num_cols=32, num_rows=32):
    with Image.open(image_path) as img:
        # size of the image
        width, height = img.size
        # size of each patch
        patch_width = width // num_cols
        patch_height = height // num_rows
        patches = []
        # Loop over the rows and columns to extract each patch
        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate the position of the patch
                left = col * patch_width
                top = row * patch_height
                right = left + patch_width
                bottom = top + patch_height
                # Crop the patch from the image
                patch = img.crop((left, top, right, bottom))
                patches.append(patch)
    return patches


# In[4]:


patches = split_image('src.jpg')
patches


# In[5]:


def save_patches(patches, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, patch in enumerate(patches):
        patch_path = os.path.join(output_dir, f'patch_{i}.jpg')
        patch.save(patch_path)
        print(f'Saved patch {i} to {patch_path}')

save_patches(patches, 'output_patches')


# In[6]:


img1 = Image.open('./output_patches/patch_0.jpg')
img1


# # MRF approach:

# Since the segmentation is color-based, one of my idea was to use Markov Random Fiel, since there is a high chance that a pixel's neighbours share similar properties like intensity and color. MRF can capture these properties.

# In[7]:


def segment_and_classify_nuclei_MRF(image_path):

    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Mixture Model (GMM) to estimate the number of classes
    gmm = GaussianMixture(n_components=3, random_state=42)
    flattened_image = grayscale_image.reshape(-1, 1)
    gmm.fit(flattened_image)
    classification = gmm.predict(flattened_image)

    # Reshape the classification back to the original image shape
    classification = classification.reshape(grayscale_image.shape)

    # Create a mask for the classified nuclei
    nuclei_mask = np.uint8(classification)

    # Apply morphological operations to enhance the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_CLOSE, kernel)
    nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of nuclei in the mask
    contours, _ = cv2.findContours(nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the nuclei
    segmented_image = image.copy()
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the segmented image
    cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Specify the path to your image
image_path = "output_patches/patch_0.jpg"

# Call the segmentation and classification function
segment_and_classify_nuclei_MRF(image_path)


# # Color separation approach:

# The function color_separate() is from the official scikit-image documentation (https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_ihc_color_separation.html#sphx-glr-auto-examples-color-exposure-plot-ihc-color-separation-py). Since the 'src.jpg' file does seem to be the result of IHC staining, in order to avoid the problem of non-existent masks to train any model with, I chose this approach to have a chance at segmenting specific cell nuclei. Sadly, only the D image (possibly DAB staining) could be used for segmentation, because the other H image (possibly hematoxilyn) might have also stained the DAB positive cell nuclei. The outlines could therefore only be generated for ony class of nuclei.

# In[8]:


def color_separate(ihc_rgb):

    #Convert the RGB image to HED using the prebuilt skimage method
    ihc_hed = rgb2hed(ihc_rgb)
    
    # Create an RGB image for each of the separated stains
    #Convert them to ubyte for easy saving to drive as an image
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = img_as_ubyte(hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)))
    ihc_e = img_as_ubyte(hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1)))
    ihc_d = img_as_ubyte(hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))

    #Optional fun exercise of combining H and DAB stains into a single image with fluorescence look
    
    h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1),
                          in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)))
    d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1),
                          in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)))

# Cast the two channels into an RGB image, as the blue and green channels
#Convert to ubyte for easy saving as image to local drive
    zdh = img_as_ubyte(np.dstack((null, d, h))) #DAB in green and H in Blue

    return (ihc_h, ihc_e, ihc_d, zdh)


# In[9]:


ihc_rgb =io.imread("output_patches/patch_0.jpg")
H,E,D,HD = color_separate(ihc_rgb)


# In[10]:


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(H)
plt.axis("off")
plt.title("H image")

plt.subplot(1,2,2)
plt.imshow(D)
plt.axis("off")
plt.title("D image")


# In[11]:


H_img = np.invert(H[:,:,2]) # -> H image is not suitable for meaningful segmentation
D_img = np.invert(D[:,:,2]) # -> D image only contains cell nuclei which got colored brown and nothing else

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(H_img)
plt.title("H image")

plt.subplot(1,2,2)
plt.imshow(D_img)
plt.title("D image")


# I don't have access to GPU, but in case you do, the code can be modified by uncommenting the first lines.

# # Segmentation with Voronoi-Otsu labeling:

# In[12]:


def segment_and_outline_D_images(input_img, output_path):
    #device = cle.select_device()
    #print(f"Used GPU: {device}")
    #input_gpu = cle.push(input_image)
    
    
    cle.imshow(input_img)
    sigma_spot_detection = 11 #larger value -> larger spot detected
    sigma_outline = 5

    segmented = cle.voronoi_otsu_labeling(input_img, spot_sigma=sigma_spot_detection, 
                                              outline_sigma=sigma_outline)
    cle.imshow(segmented, labels = True)
    outline = cle.detect_label_edges(segmented)
    plt.imshow(outline, cmap = 'gray')
    

    os.makedirs(output_path, exist_ok=True)
    # Save the outline image to the output directory
    output_filename = "outline_image_D.jpg"
    output_file_path = os.path.join(output_path, output_filename)
    plt.savefig(output_file_path, format = 'jpg', dpi=300, bbox_inches='tight')  # Save the image with higher resolution and without extra whitespace
input_img = D_img
output_path = 'segmented_D_images'
segment_and_outline_D_images(input_img, output_path)

