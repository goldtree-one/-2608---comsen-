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

b, g, r = cv2.split(image)
cv2.imwrite(os.path.join(save_dir, "blue_channel.jpg"),  b)
cv2.imwrite(os.path.join(save_dir, "green_channel.jpg"), g)
cv2.imwrite(os.path.join(save_dir, "red_channel.jpg"), r)
#以上三通道分离操作完成
new_width = int(input("请输入目标宽度: "))
new_height = int(input("请输入目标高度: "))
angle = float(input("请输入旋转角度: "))
tx=int(input("请输入需要平移的水平距离"))
ty=int(input("请输入需要平移的竖直距离"))
aditon=int(input("请输入你所期望的翻转状态："))
resized_image = cv2.resize(image, (new_width, new_height))
#保存操作cv2.imwrite(os.path.join(save_dir, "resized.jpg"), resized_image)
cv2.imwrite(os.path.join(save_dir, "resized.jpg"), resized_image)
#缩放操作完成
center=(width/2, height/2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
#计算一个旋转矩阵，并将其运用到图片格式上
cv2.imwrite(os.path.join(save_dir, "rotated.jpg"), rotated_image)

translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
cv2.imwrite(os.path.join(save_dir, "translated.jpg"), translated_image)

flipped_image = cv2.flip(image, aditon)
cv2.imwrite(os.path.join(save_dir, "flipped.jpg"), flipped_image)

#对图片进行缩放，旋转，平移，翻转