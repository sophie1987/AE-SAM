import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

image = cv2.imread('data/data3/pic_resize.tif', cv2.IMREAD_COLOR)
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

segmentation = np.zeros(image.shape[:2], dtype=np.int32)
for i, mask in enumerate(masks):
    segmentation[mask['segmentation']] = i+1

np.savetxt('data/result3/sam_auto_origin.csv', segmentation, delimiter=',', fmt='%d')