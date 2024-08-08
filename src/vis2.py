import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_points(data, key_ids):
    hulls = []
    for nodule in data["ct_nodule"]:
        for contour in nodule["contour3D"]:
            if contour["sliceId"] in key_ids:
                points = np.array(contour["data"][0], dtype=np.float32)
                hull = cv2.convexHull(points)
                hulls.append((hull, contour["sliceId"]))
    return hulls

def get_bounds(hull):
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for point in hull:
        x, y = point[0]
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

    return min_x, min_y, max_x, max_y

def read_axes(data):
    axes = []
    key_ids = set()
    for nodule in data["ct_nodule"]:
        keyslice_id = nodule.get("keySliceId")
        long_axis = nodule.get("longAxis", [])
        short_axis = nodule.get("shortAxis", [])
        long_axis_points = None
        short_axis_points = None

        if long_axis:
            long_axis_points = ((long_axis[0], long_axis[1]), (long_axis[2], long_axis[3]))
        if short_axis:
            short_axis_points = ((short_axis[0], short_axis[1]), (short_axis[2], short_axis[3]))
        
        axes.append((keyslice_id, long_axis_points, short_axis_points))
        key_ids.add(keyslice_id)
    return axes, key_ids

def read_diameters(file_path, key_ids):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    diameters = []
    for line in lines:
        values = list(map(float, line.strip().split()))
        slice_id = int(values[0])
        p1 = (values[1], values[2])
        p2 = (values[3], values[4])
        p3 = (values[5], values[6])
        p4 = (values[7], values[8])
        if slice_id in key_ids:
            diameters.append((slice_id, p1, p2, p3, p4))
    return diameters

def visualize_hull(hull, bounds, diameter, axes):
    min_x, min_y, max_x, max_y = bounds
    slice_id, p1, p2, p3, p4 = diameter
    _, long_axis, short_axis = axes

    # 计算图像尺寸和缩放比例
    img_size = 800  # 固定图像大小
    scale_x = img_size / (max_x - min_x + 20)
    scale_y = img_size / (max_y - min_y + 20)

    # 创建空白图像
    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # 平移点到新的图像范围内并缩放
    translated_points = []
    for point in hull:
        x, y = point[0]
        translated_x = int((x - min_x + 10) * scale_x)
        translated_y = int((y - min_y + 10) * scale_y)
        translated_points.append([translated_x, translated_y])

    translated_points = np.array(translated_points, dtype=np.int32)

    # 平移并缩放长径和短径的点
    p1 = (int((p1[0] - min_x + 10) * scale_x), int((p1[1] - min_y + 10) * scale_y))
    p2 = (int((p2[0] - min_x + 10) * scale_x), int((p2[1] - min_y + 10) * scale_y))
    p3 = (int((p3[0] - min_x + 10) * scale_x), int((p3[1] - min_y + 10) * scale_y))
    p4 = (int((p4[0] - min_x + 10) * scale_x), int((p4[1] - min_y + 10) * scale_y))

    # 绘制凸包
    cv2.polylines(image, [translated_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # 绘制长径和短径
    cv2.line(image, p1, p2, (255, 0, 0), 2)  # 长径
    cv2.line(image, p3, p4, (0, 0, 255), 2)  # 短径

    # 绘制长轴和短轴
    if long_axis:
        long_p1 = (int((long_axis[0][0] - min_x + 10) * scale_x), int((long_axis[0][1] - min_y + 10) * scale_y))
        long_p2 = (int((long_axis[1][0] - min_x + 10) * scale_x), int((long_axis[1][1] - min_y + 10) * scale_y))
        cv2.line(image, long_p1, long_p2, (0, 255, 255), 2)  # 黄色表示original长轴

    if short_axis:
        short_p1 = (int((short_axis[0][0] - min_x + 10) * scale_x), int((short_axis[0][1] - min_y + 10) * scale_y))
        short_p2 = (int((short_axis[1][0] - min_x + 10) * scale_x), int((short_axis[1][1] - min_y + 10) * scale_y))
        cv2.line(image, short_p1, short_p2, (255, 0, 255), 2)  # 紫色表示original短轴

    # 在图像上显示 sliceId
    cv2.putText(image, f"Slice ID: {slice_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 使用 Matplotlib 显示图像，并添加坐标轴
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title("Convex Hull")

    # 设置坐标轴的刻度
    ticks_x = np.linspace(0, img_size, num=10)
    ticks_y = np.linspace(0, img_size, num=10)
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)
    ax.set_xticklabels([f'{min_x + (max_x - min_x) * i / 10:.1f}' for i in range(10)])
    ax.set_yticklabels([f'{min_y + (max_y - min_y) * i / 10:.1f}' for i in range(10)])

    # 显示网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()

def main():
    # JSON 文件路径
    json_file_path = 'C:/code/extract_visual/json/41.json'
    # 长径和短径文件路径
    diameters_file_path = 'C:/code/extract_visual/output/41.txt'

    # 从 JSON 文件中提取数据
    data = load_json(json_file_path)

    # 读取长轴和短轴数据
    axes, key_ids = read_axes(data)

    # 提取点并计算凸包
    hulls = extract_points(data, key_ids)

    # 读取长径和短径数据
    diameters = read_diameters(diameters_file_path, key_ids)

    # 依次可视化每个凸包
    for hull, diameter in zip(hulls, diameters):
        for axis in axes:
            if diameter[0] == axis[0]:
                bounds = get_bounds(hull[0])
                visualize_hull(hull[0], bounds, diameter, axis)

if __name__ == "__main__":
    main()
