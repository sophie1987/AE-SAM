from PIL import Image

# 打开 TIF 影像
input_path = "data/test/pic.tif"  # 输入 TIF 文件路径
output_path = "data/test/pic_resize.tif"  # 输出 TIF 文件路径

# 读取图像
image = Image.open(input_path)

# 获取图像尺寸
width, height = image.size
print(width,height)

# 指定截取区域的像素大小
#crop_width = 600  # 横轴 100 像素
#crop_height = 600  # 纵轴 100 像素

# 计算左下角区域的坐标
'''left = 0
right = crop_width + left
top = height - crop_height
bottom = height'''
left = 0
right = 50
top = 70
bottom = 120

# 检查是否超出图像边界
if right > width or bottom > height:
    raise ValueError("指定的截取区域超出图像边界！")

# 截取左下角指定区域
cropped_image = image.crop((left, top, right, bottom))

# 保存新图像
cropped_image.save(output_path, format="TIFF")

# 关闭图像
image.close()
cropped_image.close()