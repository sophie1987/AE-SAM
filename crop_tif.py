import cv2
import os
def crop_image(img_path, mask_path, output_dir, size=512):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)
    h, w = img.shape[:2]
    for i in range(0, h, size):
        for j in range(0, w, size):
            crop_img = img[i:i+size, j:j+size]
            crop_mask = mask[i:i+size, j:j+size]
            if crop_img.shape[:2] == (size, size):  # 忽略边缘小块
                cv2.imwrite(os.path.join(output_dir, 'images', f'crop_{i}_{j}.tif'), crop_img)
                cv2.imwrite(os.path.join(output_dir, 'masks', f'crop_{i}_{j}.tif'), crop_mask)
crop_image('D:\\temp\\paper\\data\\LandCover.ai\\5\\test1\\swiss_IMG_8766.JPG', 'D:\\temp\\paper\\data\\LandCover.ai\\5\\test1\\swiss_IMG_8766.png', 'D:\\temp\\paper\\data\\LandCover.ai\\5',size=600)
