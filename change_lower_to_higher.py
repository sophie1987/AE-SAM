import numpy as np
from osgeo import gdal
import logging
import scipy.interpolate as interpolate

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(file_path, bands=None):
    """加载影像数据"""
    try:
        ds = gdal.Open(file_path)
        if bands is None:
            bands = range(1, ds.RasterCount + 1)
        image = np.array([ds.GetRasterBand(i).ReadAsArray() for i in bands])
        image = image.transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
        return image
    except Exception as e:
        logging.error(f"加载影像文件 {file_path} 失败: {e}")
        return None

def load_landcover(file_path, target_shape):
    try:
        ds = gdal.Open(file_path)
        landcover = ds.GetRasterBand(1).ReadAsArray()
        print(landcover.shape)
        # 上采样至目标分辨率
        x = np.linspace(0, landcover.shape[1]-1, landcover.shape[1])
        y = np.linspace(0, landcover.shape[0]-1, landcover.shape[0])
        # 使用RegularGridInterpolator替代interp2d，使用最近邻插值算法将地物图匹配到高清影像图的分辨率
        interp = interpolate.RegularGridInterpolator((y, x), landcover, method='nearest', bounds_error=False, fill_value=None)
        x_new = np.linspace(0, landcover.shape[1]-1, target_shape[1])
        y_new = np.linspace(0, landcover.shape[0]-1, target_shape[0])
        xx, yy = np.meshgrid(x_new, y_new)
        points = np.column_stack((yy.ravel(), xx.ravel()))
        landcover_resampled = interp(points).reshape(target_shape[0], target_shape[1])
        return landcover_resampled
    except Exception as e:
        logging.error(f"加载地物图 {file_path} 失败: {e}")
        return None

image = load_image('data/data4/pic_resize.tif')
landcover_resampled = load_landcover('data/data4/prompt.tif', image.shape[:2])
np.savetxt('data/data4/prompt_resize.csv', landcover_resampled, delimiter=',', fmt='%d')