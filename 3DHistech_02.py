#!/usr/bin/env python
# coding: utf-8

# In[19]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import os
import requests
import zipfile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2


# In[2]:


print(f" Version of PyTorch: {torch.__version__}")


# In[3]:


Image.MAX_IMAGE_PIXELS = None


# In[4]:


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


# In[5]:


patches = split_image('src.jpg')
patches


# In[6]:


def save_patches(patches, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, patch in enumerate(patches):
        patch_path = os.path.join(output_dir, f'patch_{i}.jpg')
        patch.save(patch_path)
        print(f'Saved patch {i} to {patch_path}')

save_patches(patches, 'output_patches')


# In[23]:


img1 = Image.open('./output_patches/patch_0.jpg')
img1


# In[24]:


class NucleiSegmentationModel(nn.Module):
    def __init__(self, num_classes = 4):
        super(NucleiSegmentationModel, self).__init__()
        self.unet = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        self.unet.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(3, 3), stride=(1, 1))
    
    def forward(self, x)->torch.Tensor:
        return self.unet(x)['out']


# In[25]:


# Custom dataset 
class NucleiDataset(Dataset):
    def __init__(self, source_path = 'output_patches'):
        self.source_path = source_path
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.source_path)

    def __getitem__(self, index):
        # Load and preprocess the image and its corresponding mask (if available)
        image = Image.open(self.source_path[index])
        image = self.transform(image)

        return image


# In[29]:


# Define the function for nuclei segmentation and classification
def segment_and_classify_nuclei(source_path , output_path):
    # Apply the model
    model = NucleiSegmentationModel()

    # Evaluation
    model.eval()

    # Image preprocessing:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #list of all image files in the source folder
    image_files = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith('.jpg')]
    first_ten_images = image_files[:10]

    for i, image_path in enumerate(image_files):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        #Inference
        with torch.no_grad():
            output = model(image_tensor)


        #red outlines for one class, orange and yellow and blue for another
        red_outlines = get_red_outlines(output)
        orange_outlines = get_orange_outlines(output)
        yellow_outlines = get_yellow_outlines(output)
        blue_outlines = get_blue_outlines(output)

        # Draw the outlines on the actual image
        draw = ImageDraw.Draw(image)

        for outline in red_outlines:
            draw.polygon(outline, outline="red")

        for outline in yellow_outlines:
            draw.polygon(outline, outline="yellow")

        for outline in blue_outlines:
            draw.polygon(outline, outline="blue")

        # Save the image with outlines
        output_filename = os.path.basename(image_path)
        output_image_path = os.path.join(output_path, output_filename)
        image.save(output_image_path)


# In[30]:


segment_and_classify_nuclei('output_patches', 'segmented_classified_images')


# In[ ]:




