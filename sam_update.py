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
import os

# 设置环境变量，限制 OpenMP 线程数以避免多线程冲突
os.environ["OMP_NUM_THREADS"] = "1"

class TerrainSegmenter:
    def __init__(self, model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth"):
        """
        初始化地物分割器，加载 Segment Anything Model (SAM)。
        
        参数:
            model_type: 模型类型，默认为 'vit_h'（Vision Transformer Huge）。
            checkpoint: 预训练模型权重文件的路径。
        """
        # 加载 SAM 模型和权重
        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        # 初始化 SAM 预测器
        self.predictor = SamPredictor(self.model)

    def sam_predict(self, image, input_point, input_label):
        """
        使用 SAM 模型对输入影像进行分割预测。
        
        参数:
            image: 输入影像，格式为 NumPy 数组（RGB 或多光谱）。
            input_point: 提示点坐标，格式为 NumPy 数组 [[x1, y1], [x2, y2], ...]。
            input_label: 提示点标签，格式为 NumPy 数组 [1, 1, ...]（1 表示正样本）。
        
        返回:
            包含掩码和置信度的列表，格式为 [(mask1, score1), (mask2, score2), ...]。
        """
        # 设置 SAM 预测器的输入影像
        self.predictor.set_image(image)
        # 使用 SAM 进行分割预测，启用多掩码输出
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        return list(zip(masks, scores))

    def extract_features(self, image):
        """
        从输入影像中提取多种特征，包括光谱、植被指数、纹理和边缘特征。
        
        参数:
            image: 输入影像，格式为 NumPy 数组（RGB 或多光谱）。
        
        返回:
            features: 字典，键为特征名称（如 'spectral_band_0'、'lbp'），值为对应特征的二维数组。
        """
        features = {}
        
        # 1. 提取光谱特征（原始波段）
        if len(image.shape) == 3:  # 确保是多光谱或 RGB 影像
            for b in range(min(3, image.shape[2])):  # 提取前 3 个波段
                features[f'spectral_band_{b}'] = image[:, :, b]
        
        # 2. 计算植被指数（NDVI），需有近红外和红波段
        if image.shape[2] >= 4:  # 假设第 3 通道为近红外，第 2 通道为红波段
            nir = image[:, :, 3].astype(float)
            red = image[:, :, 2].astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                # 计算 NDVI: (NIR - Red) / (NIR + Red)
                ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))
                ndvi = np.nan_to_num(ndvi)  # 处理 NaN 和无穷值
            features['ndvi'] = ndvi
        
        # 3. 转换为灰度图以提取纹理和边缘特征
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)  # RGB 转灰度
        else:
            gray = image
        
        # 4. 计算局部二值模式（LBP）纹理特征
        radius = 3  # LBP 半径
        n_points = 8 * radius  # 采样点数
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        features['lbp'] = lbp
        
        # 5. 提取边缘特征（Canny 边缘检测）
        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)  # 阈值 100 和 200
        features['edges'] = edges
    
        return features

    def find_pixel_from_prompt(self, prompt, terrain_type_number, select_number=10, min_select_number=1):
        """
        从低分辨率分类图中选择指定地物类别的提示点。
        
        参数:
            prompt: 低分辨率分类图，格式为 NumPy 数组。
            terrain_type_number: 目标地物类别的 ID。
            select_number: 每类地物选取的提示点数量，默认为 10。
            min_select_number: 最小提示点数量，低于此值返回空列表。
        
        返回:
            提示点坐标列表，格式为 [(x1, y1), (x2, y2), ...]。
        """
        # 找到分类图中属于指定地物类别的像素点坐标
        result_index = np.where(prompt == terrain_type_number)
        x = result_index[0]  # 行坐标
        y = result_index[1]  # 列坐标
        result = list(zip(y, x))  # 转换为 (x, y) 格式

        if len(result) < min_select_number:  # 点数少于最小要求，返回空列表
            return []
        elif len(result) > select_number:  # 点数多于指定数量，随机采样
            return random.sample(result, select_number)
        else:
            return result
    
    def find_pixel(self, points, image):
        """
        基于影像特征和空间约束选择与提示点特征相似的点，用于增强 SAM 分割。
        
        参数:
            points: 初始提示点坐标，格式为 [(x1, y1), (x2, y2), ...]。
            image: 输入影像，格式为 NumPy 数组。
        
        返回:
            优化后的提示点坐标列表。
        """
        # 使用默认权重调用特征引导采样
        points = self._feature_guided_sampling(points, image)
        return points

    def _feature_guided_sampling(self, points, image, select_number=20, feature_weights=None, max_radius=5):
        """
        基于影像特征和空间约束选择与提示点特征相似的点。
        
        参数:
            points: 初始提示点坐标，格式为 [(x1, y1), (x2, y2), ...]。
            image: 输入影像，格式为 NumPy 数组。
            select_number: 目标提示点数量，默认为 20。
            feature_weights: 特征权重字典，控制不同特征的重要性。
            max_radius: 空间约束的最大半径（像素）。
        
        返回:
            优化后的提示点坐标列表，格式为 [(x1, y1), (x2, y2), ...]。
        """
        # 提取影像特征
        features = self.extract_features(image)
        
        # 设置默认特征权重
        if feature_weights is None:
            feature_weights = {
                'spectral': 1.5,  # 光谱特征权重
                'ndvi': 1.2,      # 植被指数权重
                'texture': 1.0,   # 纹理特征权重
                'edges': 1.1,     # 边缘特征权重
            }        
        # 为每个特征分配权重
        weights = []
        for feat_name in features.keys():
            if 'spectral_band' in feat_name:
                weights.append(feature_weights.get('spectral', 1.0))
            elif 'ndvi' in feat_name:
                weights.append(feature_weights.get('ndvi', 1.0))
            elif 'lbp' in feat_name or 'texture' in feat_name:
                weights.append(feature_weights.get('texture', 1.0))
            elif 'edges' in feat_name:
                weights.append(feature_weights.get('edges', 1.0))
            else:
                weights.append(1.0)
        
        # 将输入点转换为 NumPy 数组并限制坐标范围
        points = np.array(points)
        x_coords = np.clip(points[:, 0].astype(int), 0, image.shape[1]-1)
        y_coords = np.clip(points[:, 1].astype(int), 0, image.shape[0]-1)
        
        # 提取提示点的特征向量
        feature_vectors = []
        for x, y in zip(x_coords, y_coords):
            fv = []
            for feat_name, feat_map in features.items():
                if len(feat_map.shape) == 2:
                    fv.append(feat_map[y, x])  # 单通道特征
                else:
                    fv.extend(feat_map[y, x, :])  # 多通道特征
            feature_vectors.append(fv)
        
        feature_vectors = np.array(feature_vectors)
        
        # 应用特征权重
        if len(weights) == feature_vectors.shape[1]:
            feature_vectors = feature_vectors * np.array(weights)
        
        # 当输入点数量较多时，进行异常点过滤
        if len(points) > select_number * 2:  # 当输入点数量是目标点数的两倍以上时
            from sklearn.covariance import EllipticEnvelope
            try:
                # 使用鲁棒协方差估计来检测异常点
                cov = EllipticEnvelope(contamination=0.1, random_state=42)
                inliers_mask = cov.fit_predict(feature_vectors) == 1
                if np.any(inliers_mask):
                    points = points[inliers_mask]
                    feature_vectors = feature_vectors[inliers_mask]
                    print(f"Filtered out {np.sum(~inliers_mask)} outlier points")
            except Exception as e:
                print(f"Outlier detection failed: {e}")
        
        # 标准化特征向量
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_vectors = scaler.fit_transform(feature_vectors)
        
        # 使用 K-means 聚类提示点
        n_clusters = min(len(points), max(2, select_number // 2))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(feature_vectors)
        cluster_centers = kmeans.cluster_centers_
        
        # 生成候选点（空间约束）
        height, width = image.shape[:2]
        all_points = []
        for x, y in points:
            x_min, x_max = max(0, x - max_radius), min(width, x + max_radius + 1)
            y_min, y_max = max(0, y - max_radius), min(height, y + max_radius + 1)
            all_points.extend([(i, j) for i in range(x_min, x_max) for j in range(y_min, y_max)])
        
        all_points = list(set(all_points))  # 去重
        all_x_coords = np.array([p[0] for p in all_points])
        all_y_coords = np.array([p[1] for p in all_points])
        
        # 提取候选点的特征向量
        all_feature_vectors = []
        for x, y in zip(all_x_coords, all_y_coords):
            fv = []
            for feat_name, feat_map in features.items():
                if len(feat_map.shape) == 2:
                    fv.append(feat_map[y, x])
                else:
                    fv.extend(feat_map[y, x, :])
            all_feature_vectors.append(fv)
        
        all_feature_vectors = np.array(all_feature_vectors)
        all_feature_vectors = scaler.transform(all_feature_vectors) * np.array(weights)
        
        # 为每个簇选择最接近中心的点
        selected_points = []
        points_per_cluster = max(1, select_number // n_clusters)
        for i in range(n_clusters):
            distances = np.linalg.norm(all_feature_vectors - cluster_centers[i], axis=1)
            cluster_indices = np.argsort(distances)[:points_per_cluster]
            selected_points.extend([(int(all_points[idx][0]), int(all_points[idx][1])) 
                                for idx in cluster_indices])
        
        # 验证所选点的特征相似度
        selected_features = []
        for x, y in selected_points:
            fv = []
            for feat_name, feat_map in features.items():
                if len(feat_map.shape) == 2:
                    fv.append(feat_map[y, x])
                else:
                    fv.extend(feat_map[y, x, :])
            selected_features.append(fv)
        
        selected_features = np.array(selected_features)
        selected_features = scaler.transform(selected_features) * np.array(weights)
        
        # 过滤异常点
        min_distances = np.min([np.linalg.norm(selected_features - center, axis=1) 
                            for center in cluster_centers], axis=0)
        threshold = np.percentile(min_distances, 75)
        valid_indices = min_distances <= threshold
        selected_points = [selected_points[i] for i in range(len(selected_points)) if valid_indices[i]]
        
        # 补充随机点
        if len(selected_points) < select_number and all_points:
            remaining = select_number - len(selected_points)
            available_points = list(set(all_points) - set(selected_points))
            selected_points.extend(random.sample(available_points, 
                                                min(remaining, len(available_points))))
        
        return selected_points[:select_number]
    
    def change_prompt_pixel_to_pic_pixel(self, prompt_point_list, low_shape, high_shape):
        """
        将低分辨率分类图中的提示点映射到高分辨率影像的格子中心点坐标。
        
        参数:
            prompt_point_list: 低分辨率提示点坐标，格式为 [[x1, y1], ...]。
            low_shape: 低分辨率分类图的形状 (height, width)。
            high_shape: 高分辨率影像的形状 (height, width)。
        
        返回:
            高分辨率提示点坐标，格式为 NumPy 数组 [[x1, y1], ...]。
        """
        scale_row = high_shape[0] / low_shape[0]  # 行缩放比例
        scale_col = high_shape[1] / low_shape[1]  # 列缩放比例
        tmp = np.array(prompt_point_list)
        high_res_lefttop = np.zeros_like(tmp, dtype=float)
        high_res_lefttop[:, 0] = tmp[:, 0] * scale_row  # 映射到高分辨率左上角
        high_res_lefttop[:, 1] = tmp[:, 1] * scale_col
        high_res_center = high_res_lefttop + np.array([scale_row, scale_col]) / 2 - 0.5  # 计算格子中心
        input_point = np.rint(high_res_center).astype(int)  # 四舍五入取整
        return input_point

    def fill_passible(self, image_array, classification_list):
        """
        使用层次化融合和自适应融合策略填充地物分类结果。
        
        参数:
            image_array: 初始化的结果数组，格式为 NumPy 数组。
            classification_list: 包含 (掩码, 置信度, 类别) 的列表。
        
        返回:
            填充后的地物分类结果数组。
        """
        if not classification_list:
            return image_array
        
        height, width = image_array.shape
        class_ids = list(set(mask[2] for mask in classification_list))
        
        # 第一阶段：层次化融合
        class_merged_masks = {}
        for class_id in class_ids:
            class_masks = [(mask, score) for mask, score, cid in classification_list if cid == class_id]
            if not class_masks:
                continue
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
        
        # 第二阶段：自适应融合
        confidence_scores = np.zeros((len(class_ids), height, width))
        for i, class_id in enumerate(class_ids):
            confidence_scores[i] = class_merged_masks[class_id]
        
        # 计算空间一致性分数
        spatial_scores = np.zeros_like(confidence_scores)
        kernel = np.ones((5, 5), np.uint8)
        for i, class_id in enumerate(class_ids):
            class_mask = (confidence_scores[i] > 0.5).astype(np.uint8)
            if np.any(class_mask):
                opened = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)  # 开运算
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)  # 闭运算
                spatial_scores[i] = closed.astype(float)
        
        # 结合置信度和空间一致性
        spatial_weight = 0.3  # 空间一致性权重
        combined_scores = (1 - spatial_weight) * confidence_scores + spatial_weight * spatial_scores
        
        # 获取每个像素点分数最高的类别
        winner_indices = np.argmax(combined_scores, axis=0)
        
        # 填充结果
        for i, class_id in enumerate(class_ids):
            image_array[winner_indices == i] = class_id
        
        # 修正边缘错误
        image_array[0, :] = image_array[1, :]
        image_array[-1, :] = image_array[-2, :]
        image_array[:, 0] = image_array[:, 1]
        image_array[:, -1] = image_array[:, -2]

        return image_array

    def get_result(self, prompt_low_csv_path, predict_img_path, result_csv_path):
        """
        主函数，执行地物分割全流程。
        
        参数:
            prompt_low_csv_path: 低分辨率分类图的 CSV 文件路径。
            predict_img_path: 高分辨率影像的路径。
            result_csv_path: 输出结果的 CSV 文件路径。
        
        返回:
            无（结果保存到文件）。
        """
        # 加载影像和低分辨率分类数据
        img = cv2.imread(predict_img_path, cv2.IMREAD_COLOR)
        prompt = np.loadtxt(prompt_low_csv_path, delimiter=',', dtype=int)
        
        # 获取所有地物类别
        terrian_list = get_distinct_classification(prompt_low_csv_path)
        image_array = np.zeros((img.shape[0], img.shape[1]))  # 初始化结果数组
        classification_list = []
        
        for key in terrian_list:
            # 从低分辨率分类图中选择提示点
            points_from_prompt = self.find_pixel_from_prompt(prompt, key)
            # 映射到高分辨率影像
            input_point = self.change_prompt_pixel_to_pic_pixel(points_from_prompt, prompt.shape, img.shape)
            if len(input_point) == 0:
                continue
            # 自适应采样优化提示点
            result_index = self.find_pixel(input_point, image=img)
            
            if len(result_index) > 0:
                points = np.array(result_index)
                input_label = np.ones(points.shape[0], dtype=int)  # 正样本标签
                terrian_result = self.sam_predict(img, points, input_label)
                terrian_result.sort(key=lambda x: x[1], reverse=True)  # 按置信度排序
                for mask, score in terrian_result:
                    classification_list.append([mask, score, key])
        
        # 填充分类结果
        self.fill_passible(image_array, classification_list)
        # 保存结果
        np.savetxt(result_csv_path, image_array, delimiter=',', fmt='%d')

if __name__ == '__main__':
    # 初始化地物分割器
    terrain_segmenter = TerrainSegmenter()
    
    # 运行分割任务
    terrain_segmenter.get_result(
        prompt_low_csv_path="data/test/prompt.csv", 
        predict_img_path="data/test/pic.tif",
        result_csv_path="data/test/result.csv"
    )