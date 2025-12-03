from PIL import Image

# 打开 TIF 图片
image_path = "data/data4/pic_resize.tif"  # 替换为你的 TIF 文件路径
image = Image.open(image_path)

# 调整图片大小为 7x8 像素
resized_image = image.resize((6, 8), Image.Resampling.NEAREST)  # 使用 NEAREST 插值方法以保留像素化效果

# 保存调整后的图片
output_path = "data/data4/prompt.tif"
resized_image.save(output_path, format="TIFF")

print(f"图片已调整为 6x8 像素并保存为 {output_path}")