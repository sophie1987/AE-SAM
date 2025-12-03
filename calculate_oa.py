from calendar import c
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cv2
from common import *
import matplotlib.patches as mpatches
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def calculate_accuracy_metrics(predicted_path, ground_truth_path):
    predicted_array = np.loadtxt(predicted_path, delimiter=',', dtype=int)
    ground_truth_array = np.loadtxt(ground_truth_path, delimiter=',', dtype=int)
    if predicted_array.shape != ground_truth_array.shape:
        print("错误：预测影像和真实影像的尺寸不匹配。")
        return None, None

    predicted_flat = predicted_array.flatten()
    ground_truth_flat = ground_truth_array.flatten()


    cm = confusion_matrix(ground_truth_flat, predicted_flat)
    
    # 获取唯一的类别标签并排序
    classes = sorted(np.unique(np.concatenate((ground_truth_flat, predicted_flat))))
    
    # 打印带有标签的混淆矩阵
    print("\n混淆矩阵 (行: 真实标签, 列: 预测标签):")
    # 打印列标签
    header = " " * 2 + "Pred: "
    for cls in classes:
        header += f"{cls:8d} "
    print(header)
    print(" " * 8 + "-" * (len(header) - 8))
    
    # 打印每一行
    for i, row in enumerate(cm):
        row_str = f"True {classes[i]:2d} | "
        for val in row:
            row_str += f"{val:8d} "
        print(row_str)
    
    # 分类报告：
    class_report = classification_report(ground_truth_flat, predicted_flat)
    print("\n分类报告 (Classification Report):")
    print(class_report)

    # 计算总体精度（OA）
    # OA = (正确分类的样本数) / (总样本数)
    correctly_classified = np.trace(cm)  # 混淆矩阵对角线元素之和即为正确分类的样本数
    total_samples = np.sum(cm)          # 混淆矩阵所有元素之和即为总样本数

    if total_samples == 0:
        oa = 0.0
    else:
        oa = correctly_classified / total_samples

    # 计算 Kappa 系数
    # cohen_kappa_score 函数直接计算 Kappa 系数
    kappa = cohen_kappa_score(ground_truth_flat, predicted_flat)

    if oa is not None and kappa is not None:
        print(f"\n影像分类结果的总体精度（OA）: {oa:.4f}")
        print(f"影像分类结果的Kappa系数: {kappa:.4f}")

def draw_all_pic(data_folder, result_folder, ground_truth_image_path=None):
    # 文件路径
    origin_img_path = os.path.join(data_folder, 'pic_resize.tif')
    prompt_csv_path = os.path.join(data_folder, 'prompt.csv')
    sam_update_path = os.path.join(result_folder, 'sam_update_segmentation.csv')
    #sam_easy_path = os.path.join(result_folder, 'sam_origin_easy_fill.csv')
    sam_auto_path = os.path.join(result_folder, 'sam_auto.csv')
    unet_path = os.path.join(result_folder, 'unet.csv')
    #gb_path = os.path.join(result_folder, 'gradientboost.csv')
    #svm_path = os.path.join(result_folder, 'svm.csv')


    # 读取原图
    origin_img = cv2.imread(origin_img_path, cv2.IMREAD_COLOR)
    # 读取提示图和分割图
    prompt_map = np.loadtxt(prompt_csv_path, delimiter=',', dtype=int)
    sam_update_map = np.loadtxt(sam_update_path, delimiter=',', dtype=int)
    #sam_easy_map = np.loadtxt(sam_easy_path, delimiter=',', dtype=int)
    sam_auto_map = np.loadtxt(sam_auto_path, delimiter=',', dtype=int)
    # 读取三种机器学习结果
    unet_img = np.loadtxt(unet_path, delimiter=',', dtype=int)
    #gb_img = np.loadtxt(gb_path, delimiter=',', dtype=int)
    #svm_img = np.loadtxt(svm_path, delimiter=',', dtype=int)

    if ground_truth_image_path:
        print("\n sam_update metrics:")
        calculate_accuracy_metrics(sam_update_path, ground_truth_image_path)
        print("\n origin sam metrics:")
        calculate_accuracy_metrics(sam_auto_path, ground_truth_image_path)
        print("\n unet metrics:")
        calculate_accuracy_metrics(unet_path, ground_truth_image_path)
        #print("\n gradient boosting metrics:")
        #calculate_accuracy_metrics(gb_path, ground_truth_image_path)
        #print("\n svm metrics:")
        #calculate_accuracy_metrics(svm_path, ground_truth_image_path)


    # colormap
    custom_class_colors = get_custom_colormap(prompt_csv_path)
    sorted_labels = sorted(custom_class_colors.keys())
    #print(sorted_labels)
    colors_list = [custom_class_colors[label] for label in sorted_labels]
    #print(colors_list)
    cmap_display = ListedColormap(colors_list)

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    plt.subplots_adjust(bottom=0.12, wspace=0.02, hspace=0.05)

    # 第一行
    axes[0, 0].imshow(origin_img)
    axes[0, 0].set_title('(a)Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.ones_like(origin_img) * 255)  # 先铺一张白底
    axes[0, 1].set_title('(b)Origin Prompt')
    axes[0, 1].axis('off')
    # 插入小的inset
    axins = inset_axes(axes[0, 1], width="40%", height="40%", loc='center')
    im1 = axins.imshow(prompt_map, cmap=cmap_display)
    axins.set_xticks([])
    axins.set_yticks([])

    im2 = axes[0, 2].imshow(sam_update_map, cmap=cmap_display)
    axes[0, 2].set_title('(c)AE-SAM')
    axes[0, 2].axis('off')


    im3 = axes[1, 0].imshow(sam_auto_map, cmap=cmap_display)
    axes[1, 0].set_title('(d)AMG-SAM')
    axes[1, 0].axis('off')

    im4 = axes[1, 1].imshow(unet_img, cmap=cmap_display)
    axes[1, 1].set_title('(e)U-Net')
    axes[1, 1].axis('off')

    '''im5 = axes[1, 2].imshow(svm_img, cmap=cmap_display)
    axes[1, 2].set_title('(f)SVM')
    axes[1, 2].axis('off')'''

    # Create legend

    cutom_label = []
    for label,color in zip(sorted_labels,colors_list):
        if label == 0:
            label = 'Forest(20)'
            cutom_label.append(plt.Rectangle((0,0), 1, 1, fc=color, edgecolor='black', label=label))
        elif label == 30:
            label = "Grass(30)"
            cutom_label.append(plt.Rectangle((0,0), 1, 1, fc=color, edgecolor='black', label=label))
        elif label == 60:
            label = "Water(60)"
            cutom_label.append(plt.Rectangle((0,0), 1, 1, fc=color, edgecolor='black', label=label))
        elif label == 70:
            label = "Tundra(70)"
            cutom_label.append(plt.Rectangle((0,0), 1, 1, fc=color, edgecolor='black', label=label))
        elif label == 90:
            label = "Bareland(90)"
            cutom_label.append(plt.Rectangle((0,0), 1, 1, fc=color, edgecolor='black', label=label))
        else:
            continue

    legend_elements = cutom_label
    '''legend_elements = [
        plt.Rectangle((0,0), 1, 1, fc=color, edgecolor='black', label=label) 
        for label, color in zip(sorted_labels, colors_list)
    ]'''
    
    # Adjust subplot layout first
    plt.tight_layout(rect=[0, 0.15, 1, 1])  # Leave space at bottom
    
    # Add legend below the subplots
    legend = fig.legend(
        handles=legend_elements, 
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),  # Position below subplots
        ncol=min(5, len(legend_elements)),
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_transform=fig.transFigure
    )
    
    # Adjust the figure to make room for the legend
    plt.subplots_adjust(bottom=0.1)
    
    plt.show()


def draw_accurate(csv_file='data/result4/accurate.csv'):
    accurate_map = np.loadtxt(csv_file, delimiter=',', dtype=int)
    # colormap
    custom_class_colors = get_custom_colormap(csv_file)
    sorted_labels = sorted(custom_class_colors.keys())
    colors_list = [custom_class_colors[label] for label in sorted_labels]
    cmap_display = ListedColormap(colors_list)
    plt.imshow(accurate_map, cmap=cmap_display)
    plt.axis('off')
    # 保存图片
    plt.savefig('data/accurate_png/accurate-4.png')
    plt.show()

def draw_csv_heatmap(csv_file, output_path=None, colormap='viridis', show_colorbar=True):
    # 读取CSV数据
    data = np.loadtxt(csv_file, delimiter=',')
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    im = ax.imshow(data, cmap=colormap, aspect='auto')
    
    # 关闭坐标轴
    ax.axis('off')
    
    # 添加颜色条
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('数值', rotation=270, labelpad=15)
    
    # 为每个不同数值添加文本标签
    unique_values = np.unique(data)
    for value in unique_values:
        # 找到该数值的所有位置
        positions = np.where(data == value)
        if len(positions[0]) > 0:
            # 取第一个位置作为标签位置
            y, x = positions[0][0], positions[1][0]
            
            # 根据数值大小选择文本颜色
            text_color = 'red' if value > np.mean(unique_values) else 'white'
            
            # 添加文本标签
            ax.text(x, y, str(int(value)), 
                   ha='center', va='top', 
                   color=text_color, 
                   fontsize=10, 
                   fontweight='bold')

    plt.show()


# --- 使用示例 ---
if __name__ == "__main__":
    draw_all_pic('data/data4','data/result4',ground_truth_image_path='data/result4/accurate.csv')
    #draw_accurate('data/result1/sam_auto.csv')
    #draw_csv_heatmap('data/result1/sam_auto_origin.csv')