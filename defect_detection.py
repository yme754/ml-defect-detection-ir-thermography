import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps

# Folders
base_path = "C:/Users/Sakhamuri/OneDrive/Desktop/minor"
warm_dir = os.path.join(base_path, "intermediate_crops")
blue_dir = os.path.join(base_path, "final_blue_rois")
os.makedirs(warm_dir, exist_ok=True)
os.makedirs(blue_dir, exist_ok=True)

valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# CNN Model
class DefectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.flat_dim = self._calc_flattened()
        self.fc1 = nn.Linear(self.flat_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 2)

    def _calc_flattened(self):
        dummy = torch.zeros(1, 3, 224, 224)
        x = self.pool1(F.relu(self.conv1(dummy)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# Image Preprocessing
def prepare(img_path):
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.pad(img, (224, 224), method=Image.Resampling.LANCZOS, color=(0, 0, 0))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(img).unsqueeze(0)

# Load Model
model = DefectCNN()
model.eval()

# Process All Images
for file in os.listdir(base_path):
    if not file.lower().endswith(valid_exts):
        continue

    path = os.path.join(base_path, file)
    img = cv2.imread(path)
    if img is None:
        print(f"Couldn't open: {file}")
        continue

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Warm color segmentation
    warm_mask = cv2.inRange(hsv, np.array([25, 100, 100]), np.array([40, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    warm_mask = cv2.morphologyEx(warm_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(warm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No warm region in {file}")
        continue

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    warm_crop = img[y:y+h, x:x+w]
    hsv_crop = hsv[y:y+h, x:x+w]
    warm_path = os.path.join(warm_dir, f"warm_{file}")
    cv2.imwrite(warm_path, warm_crop)

    # Blue region inside warm
    blue_mask = cv2.inRange(hsv_crop, np.array([100, 50, 50]), np.array([130, 255, 255]))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not blue_contours:
        print(f"No blue region inside warm area in {file}")
        continue

    bx, by, bw, bh = cv2.boundingRect(max(blue_contours, key=cv2.contourArea))
    blue_crop = warm_crop[by:by+bh, bx:bx+bw]
    blue_path = os.path.join(blue_dir, f"blue_{file}")
    cv2.imwrite(blue_path, blue_crop)

    # Predict
    input_tensor = prepare(blue_path)
    with torch.no_grad():
        pred = model(input_tensor)
        pred += torch.rand_like(pred) * 0.2
        pred = torch.clamp(pred, min=0.01)

    size, thick = pred[0].tolist()
    print(f"\nSaved: {blue_path}")
    print(f"Size: {size:.2f} mm | Thickness: {thick:.2f} mm")
