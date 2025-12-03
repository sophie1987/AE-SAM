import numpy as np
import rasterio
from rasterio.transform import from_origin

def tif_to_csv(tif_path, output_csv_path, dtype=np.uint8):
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        np.savetxt(output_csv_path, data, delimiter=',', fmt='%d')

# 使用示例
if __name__ == "__main__":
    tif_path = 'data/data1/height.tif'
    output_csv_path = 'data/data1/height.csv'
    
    tif_to_csv(tif_path, output_csv_path)
    print(f"CSV文件已保存至: {output_csv_path}")