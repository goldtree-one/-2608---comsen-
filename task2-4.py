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

VIDEO_PATH = r"E:\comsenmate\example2.mp4"


# 3) 输出文件名
OUT_FILE = os.path.join(save_dir, "compare_framediff_vs_mog2.mp4")
# ========================================================

# 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"无法打开视频: {VIDEO_PATH}")
    print("请检查：1) 路径是否正确  2) 路径是否含中文  3) 文件是否存在")
    exit()

fps     = cap.get(cv2.CAP_PROP_FPS)
width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"视频信息: {width}x{height}, {fps:.2f} fps, 共 {total} 帧")

# 拼接后宽 = 原宽 × 2（左右并排）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # Windows 用 mp4v
writer = cv2.VideoWriter(OUT_FILE, fourcc, fps, (width * 2, height))
if not writer.isOpened():
    print("VideoWriter 打开失败！换 fourcc: 'XVID' + .avi")
    exit()

# 初始化两种方法
# 帧差法：先读首帧
ret, prev_frame = cap.read()
if not ret:
    print("读取首帧失败")
    cap.release()
    exit()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# MOG2
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=True
)

kernel = np.ones((3, 3), np.uint8)

# 统计
frame_idx = 0
sum_fg_diff = 0
sum_fg_mog  = 0

#  主循环
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # 方法 1：帧差法
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, mask_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    prev_gray = gray

    #  方法 2：MOG2
    fg = mog2.apply(frame)
    _, mask_mog = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    # =
    mask_diff = cv2.morphologyEx(mask_diff, cv2.MORPH_OPEN,  kernel, 1)
    mask_diff = cv2.morphologyEx(mask_diff, cv2.MORPH_CLOSE, kernel, 2)

    mask_mog  = cv2.morphologyEx(mask_mog,  cv2.MORPH_OPEN,  kernel, 1)
    mask_mog  = cv2.morphologyEx(mask_mog,  cv2.MORPH_CLOSE, kernel, 2)

    # 统计前景像素
    sum_fg_diff += int(np.count_nonzero(mask_diff))
    sum_fg_mog  += int(np.count_nonzero(mask_mog))

    #
    vis_diff = frame.copy()
    contours, _ = cv2.findContours(mask_diff, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 200:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(vis_diff, (x, y), (x+w, y+h), (0, 0, 255), 2)

    vis_mog = frame.copy()
    contours, _ = cv2.findContours(mask_mog, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 200:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(vis_mog, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # 文字标签
    cv2.putText(vis_diff, "Frame Diff", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_mog,  "MOG2",       (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #左右并排
    combined = np.hstack([vis_diff, vis_mog])
    # 确保尺寸匹配
    combined = cv2.resize(combined, (width * 2, height))
    # 写入
    writer.write(combined)
    # 实时显示
    # 显示时缩小（比如缩到 1600 宽）
    disp_w = 1600
    disp_h = int(combined.shape[0] * disp_w / combined.shape[1])
    display = cv2.resize(combined, (disp_w, disp_h))

    cv2.imshow("Compare: FrameDiff | MOG2", display)
    # 进度打印
    if frame_idx % 30 == 0:
        print(f"已处理 {frame_idx}/{total} 帧")

    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("用户中断")
        break

# ---------- 收尾 ----------
cap.release()
writer.release()
cv2.destroyAllWindows()

# ---------- 统计结果 ----------
if frame_idx > 0:
    print("\n===== 统计结果 =====")
    print(f"总帧数: {frame_idx}")
    print(f"帧差法 平均前景像素: {sum_fg_diff / frame_idx:.0f}")
    print(f"MOG2  平均前景像素: {sum_fg_mog / frame_idx:.0f}")
    print(f"\n对比视频已保存到: {OUT_FILE}")
