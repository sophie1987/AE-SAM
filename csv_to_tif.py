import numpy as np
import rasterio
from rasterio.transform import from_origin

def csv_to_tif(csv_path, output_tif_path, dtype=np.uint8):
    # 读取CSV文件
    data = np.genfromtxt(csv_path, delimiter=',', dtype=dtype)
    
    # 获取数据的行数和列数
    height, width = data.shape
    
    # 定义输出TIFF的元数据
    transform = from_origin(0, 0, 1, 1)  # 设置地理参考信息，这里使用单位像素
    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'count': 1,  # 单波段
        'width': width,
        'height': height,
        'transform': transform,
        'crs': None,  # 没有坐标参考系统
        'compress': 'lzw'  # 使用LZW压缩
    }
    
    # 保存为TIFF文件
    with rasterio.open(output_tif_path, 'w', **profile) as dst:
        dst.write(data, 1)  # 写入第一个波段

# 使用示例
if __name__ == "__main__":
    csv_path = 'data/data3/prompt.csv'
    output_tif_path = 'data/data3/prompt.tif'
    
    csv_to_tif(csv_path, output_tif_path)
    print(f"TIFF文件已保存至: {output_tif_path}")