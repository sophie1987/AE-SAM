import numpy as np
from osgeo import gdal
import torch
from segment_anything import SamPredictor, sam_model_registry
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from scipy import interpolate
from sklearn.metrics import accuracy_score, cohen_kappa_score
import cv2
import random
from common import *


class TerrainSegmenter:
    def __init__(self, model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth"):
        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.predictor = SamPredictor(self.model)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model.to(self.device)

    def sam_predict(self, image, input_point, input_label):
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        return list(zip(masks,scores))

    def find_pixel_from_prompt(self, prompt, terrain_type_number, select_number=30, min_select_number=1):
        result_index = np.where(prompt == terrain_type_number)
        x=result_index[0]
        y=result_index[1]
        result = list(zip(y,x))
        if len(result) < min_select_number: # 特征点少于指定个数，不计算
            return []
        elif len(result) > select_number: # 特征点大于指定个数，取指定个数
            return random.sample(result,select_number)
        else:
            return result

    def change_prompt_pixel_to_pic_pixel(self, prompt_point_list, low_shape, high_shape):
        """
        将低精度栅格图中的点，映射为高精度栅格图中对应格子的中心点坐标
        :param prompt_point_list: 低精度点列表，如 [[i, j], ...]
        :param low_shape: 低精度栅格图 shape，如 (height, width)
        :param high_shape: 高精度栅格图 shape，如 (height, width)
        :return: 高精度中心点的像素坐标（int）
        """
        scale_row = high_shape[0] / low_shape[0]
        scale_col = high_shape[1] / low_shape[1]
        tmp = np.array(prompt_point_list)
        # 计算高精度格子左上角
        high_res_lefttop = np.zeros_like(tmp, dtype=float)
        high_res_lefttop[:, 0] = tmp[:, 0] * scale_row
        high_res_lefttop[:, 1] = tmp[:, 1] * scale_col
        # 计算高精度格子中心点
        high_res_center = high_res_lefttop + np.array([scale_row, scale_col]) / 2 - 0.5
        input_point = np.rint(high_res_center).astype(int)
        return input_point


    def fill_passible_easy(self, image_array, classification_list):
        classification_list.sort(key=lambda x:x[1])
        for mask, score, value in classification_list:
            image_array[mask] = value
        return image_array    

    def fill_passible(self, image_array, classification_list):
        """
        使用层次化融合和自适应融合策略填充结果
        
        参数:
            image_array: 初始化的结果数组
            classification_list: 包含(掩码, 置信度, 类别)的列表
            
        返回:
            填充后的结果数组
        """
        if not classification_list:
            return image_array
        
        height, width = image_array.shape
        class_ids = list(set(mask[2] for mask in classification_list))
        
        # 第一阶段：层次化融合 - 对每个类别内部进行融合
        class_merged_masks = {}
        
        for class_id in class_ids:
            # 获取该类别的所有掩码和置信度
            class_masks = [(mask, score) for mask, score, cid in classification_list if cid == class_id]
            
            if not class_masks:
                continue
                
            # 对每个类别内部的掩码进行平均融合
            merged_mask = np.zeros((height, width))
            total_weight = 0
            
            for mask, score in class_masks:
                weight = score  # 使用置信度作为权重
                merged_mask += mask.astype(float) * weight
                total_weight += weight
            
            if total_weight > 0:
                merged_mask /= total_weight
                class_merged_masks[class_id] = merged_mask
        
        if not class_merged_masks:
            return image_array
        
        # 第二阶段：自适应融合 - 在类别间进行竞争
        # 1. 计算基于置信度的分数
        confidence_scores = np.zeros((len(class_ids), height, width))
        for i, class_id in enumerate(class_ids):
            confidence_scores[i] = class_merged_masks[class_id]
        
        # 2. 计算空间一致性分数
        spatial_scores = np.zeros_like(confidence_scores)
        kernel = np.ones((5, 5), np.uint8)
        
        for i, class_id in enumerate(class_ids):
            class_mask = (confidence_scores[i] > 0.5).astype(np.uint8)  # 二值化
            if np.any(class_mask):
                # 使用开运算和闭运算平滑边界
                opened = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
                spatial_scores[i] = closed.astype(float)
        
        # 3. 结合置信度和空间一致性（权重可调）
        spatial_weight = 0.3  # 空间一致性权重
        combined_scores = (1 - spatial_weight) * confidence_scores + spatial_weight * spatial_scores
        
        # 4. 获取每个像素点分数最高的类别
        winner_indices = np.argmax(combined_scores, axis=0)
        
        # 5. 将结果填充到输出数组中
        for i, class_id in enumerate(class_ids):
            image_array[winner_indices == i] = class_id
        
        return image_array

    def get_result(self, prompt_low_csv_path="data/data1/prompt.csv", predict_img_path="data/data1/pic_resize.tif", result_csv_path="data/result1/sam_update_segmentation.csv"):
        img = cv2.imread(predict_img_path, cv2.IMREAD_COLOR)
        prompt = np.loadtxt(prompt_low_csv_path, delimiter=',', dtype=int)
        terrian_list = get_distinct_classification(prompt_low_csv_path)
        image_array = np.zeros((img.shape[0], img.shape[1]))
        classification_list = []
        for key in terrian_list:
            result_index = self.find_pixel_from_prompt(prompt, key, 10)
            input_point= self.change_prompt_pixel_to_pic_pixel(result_index, prompt.shape, img.shape)
            input_label = np.ones(input_point.shape[0],dtype=int)
            if len(input_point)>0:
                terrian_result = self.sam_predict(img, input_point, input_label) # list[(masks, scores)], masks: (number_of_masks) x H x W, (3, 334, 128)
                terrian_result.sort(key=lambda x:x[1],reverse=True)
                for mask,score in terrian_result:
                    classification_list.append([mask, score, key])
        self.fill_passible_easy(image_array, classification_list)

        np.savetxt(result_csv_path, image_array, delimiter=',', fmt='%d')

if __name__ == '__main__':
    terrain_segmenter = TerrainSegmenter()
    terrain_segmenter.get_result(prompt_low_csv_path="data/data4/prompt.csv", predict_img_path="data/data4/pic_resize.tif", result_csv_path="data/result4/sam_origin_easy_fill.csv")