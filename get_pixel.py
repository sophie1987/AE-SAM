import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import MultipleLocator
from PIL import Image

def show_image_with_coordinates(image_path):
    # 读取TIFF图片
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)
    
    # 显示图片
    img_display = ax.imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
    
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Image with Pixel Coordinates')
    
    # 设置X轴和Y轴的主刻度为1
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    
    # 添加缩放滑块
    ax_zoom = plt.axes([0.2, 0.1, 0.6, 0.03])
    zoom_slider = Slider(ax_zoom, 'Zoom', 1, 10, valinit=1, valstep=0.1)
    
    # 显示坐标的函数
    def format_coord(x, y):
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        if 0 <= x_int < img_array.shape[1] and 0 <= y_int < img_array.shape[0]:
            return f'x={x_int}, y={y_int}'
        return ''
    
    ax.format_coord = format_coord
    
    # 更新缩放的函数
    def update_zoom(val):
        zoom = zoom_slider.val
        ax.set_xlim([img_array.shape[1]/2 - img_array.shape[1]/(2*zoom), 
                    img_array.shape[1]/2 + img_array.shape[1]/(2*zoom)])
        ax.set_ylim([img_array.shape[0]/2 - img_array.shape[0]/(2*zoom), 
                    img_array.shape[0]/2 + img_array.shape[0]/(2*zoom)][::-1])
        fig.canvas.draw_idle()
    
    zoom_slider.on_changed(update_zoom)
    
    # 初始缩放
    update_zoom(1)
    
    # 添加网格
    ax.grid(True, color='red', alpha=0.3)
    
    # 调整刻度标签的显示，避免重叠
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    plt.show()

# 使用示例
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    
    # 使用文件对话框选择TIFF文件
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="选择TIFF图片",
        filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
    )
    
    if file_path:
        show_image_with_coordinates(file_path)
    else:
        print("未选择文件")