import cv2
import os
import csv
import math
import numpy as np
image_path =r"E:\example1.jpg"
save_dir=r"E:\output"
#无法读取时进行的操作
os.makedirs(save_dir, exist_ok=True)

image = cv2.imread(image_path)
height, width = image.shape[:2]
#后面的shape操作的格式
print(f"原图尺寸: 宽={width}, 高={height}")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 这组参数针对你的example1图片：选出较亮的蓝色天空
lower_sky = np.array([90, 50, 150], dtype=np.uint8)
upper_sky = np.array([115, 255, 255], dtype=np.uint8)
sky_mask = cv2.inRange(hsv, lower_sky, upper_sky)

# 反色：天空变黑，飞机等非天空区域变白
candidate_mask = cv2.bitwise_not(sky_mask)

#  2. 提取飞机外轮廓
contours, _ = cv2.findContours(
    candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
if not contours:
    raise ValueError("没有检测到轮廓，请检查图片或调整天空颜色范围")

# 本图中选择面积最大的外轮廓作为飞机主体
plane_contour = max(contours, key=cv2.contourArea)

# 建立与原图同样高、宽的全黑单通道图像
plane_mask = np.zeros((height, width), dtype=np.uint8)

# 填满飞机区域：255表示白色，最后的-1表示填充而不是只描边
cv2.drawContours(plane_mask, [plane_contour], -1, 255, -1)

#  3. 检测飞机内部线条
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(blurred, 40, 100, L2gradient=True)

# 将飞机掩膜向内缩，减少外边界被重复画成内部线条
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
inner_mask = cv2.erode(plane_mask, kernel, iterations=1)

# 只保留既是边缘、又在飞机内部的像素
internal_edges = cv2.bitwise_and(edges, inner_mask)

# ===== 4. 计算主体面积与周长 =====
area = cv2.contourArea(plane_contour)
perimeter = cv2.arcLength(plane_contour, True)  # True表示闭合轮廓
print(f"飞机主体面积：{area:.2f} 像素平方")
print(f"飞机主体周长：{perimeter:.2f} 像素")

# ===== 5. 叠加描画并保存 =====
result = image.copy()  # 在副本上画，保留原图数据
result[internal_edges > 0] = (0, 255, 255)  # 黄色：内部线条，BGR顺序
cv2.drawContours(result, [plane_contour], -1, (0, 255, 0), 2)

for name, img in [
    ("plane_mask.png", plane_mask),
    ("plane_internal_edges.png", internal_edges),
    ("plane_result.png", result),
]:
    output_path = os.path.join(save_dir, name)
    if not cv2.imwrite(output_path, img):
        raise OSError(f"保存失败：{output_path}")
    print(f"已保存：{output_path}")
