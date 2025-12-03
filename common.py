import numpy as np

def get_distinct_classification(csv_path):
    data = np.loadtxt(csv_path, delimiter=',', dtype=int)
    unique = np.unique(data)
    unique = unique[unique != 0]
    return unique

def get_custom_colormap(prompt_csv_path):
    all_custom_class_colors = {
        0: (1.0, 0.0, 1.0), # 未分类，红色
        1: (0, 0.5, 0),      # 森林: 绿色，用于sam分割算法的结果
        2: (1.0, 1.0, 1.0),      # 裸地：白色，用于sam分割算法的结果
        3: (1.0, 1.0, 0.0), # 粮食作物: 黄色
        4: (0, 0.5, 0),      # 森林: 绿色
        5: (0.6, 1.0, 0.6),   # 草地：浅绿色
        6: (0.0, 0.0, 1.0),      # 灌丛: 深绿色
        7: (0.0, 1.0, 0.0), # 草地：浅绿色
        8: (0.0, 0.0, 0.0),      # 水体：黑色
        9: (1.0, 1.0, 1.0),      # 裸地：白色
        10: (1.0, 1.0, 0.0), # 粮食作物: 黄色
        20: (0, 0.5, 0),      # 森林: 绿色
        30: (0.6, 1.0, 0.6),   # 草地：浅绿色
        40: (0.0, 0.0, 1.0),      # 灌丛: 深绿色
        50: (0.0, 1.0, 0.0), # 草地：浅绿色
        60: (0.0, 0.0, 0.0),      # 水体：黑色
        70: (1.0, 1.0, 0.0), # 不透水层，浅黄色
        90: (1.0, 1.0, 1.0),      # 裸地：白色

    }

    custom_class_colors = {
        label: all_custom_class_colors[label] for label in get_distinct_classification(prompt_csv_path)
    }

    custom_class_colors[0] = all_custom_class_colors[0]

    return custom_class_colors
