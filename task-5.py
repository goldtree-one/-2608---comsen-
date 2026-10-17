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

VIDEO_DIR = r"E:\open"                     # 两个 mp4 所在目录
VIDEOS = ["example3.mp4", "example4.mp4"]  # 要处理的两个视频

# 可选的初始框预设表：{视频名: (x, y, w, h, 从第几帧开始)}
# 留空 = 每个视频都弹窗让你手动框选（正常用法）。
# 如果你的环境没有图形界面，就把框写在这里，程序会跳过弹窗直接跑。
PRESET_ROI = {
    # "example3.mp4": (300, 11, 112, 105, 0),
    # "example4.mp4": (1095, 685, 280, 275, 11),
}


def create_tracker():
    """
    创建一个追踪器。

    """
    for name in ("TrackerCSRT_create", "TrackerKCF_create"):
        factory = getattr(cv2, name, None)
        if factory is not None:
            print(f"[tracker] 使用 cv2.{name}")
            return factory()
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        for name in ("TrackerCSRT_create", "TrackerKCF_create"):
            factory = getattr(legacy, name, None)
            if factory is not None:
                print(f"[tracker] 使用 cv2.legacy.{name}")
                return factory()
    print("[tracker] 使用 cv2.TrackerMIL_create")
    return cv2.TrackerMIL_create()


def track_one_video(video_name, init_frame=0, roi=None):
    """读一个视频 -> 框选小球 -> 逐帧追踪 -> 保存结果视频"""
    video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 打不开视频：{video_path}")
        return

    # 帧率、分辨率、总帧数。写视频时必须用同样的帧率和分辨率。
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n=== {video_name}  {width}x{height}  fps={fps:.2f}  共{total}帧 ===")

    # 跳到想开始的那一帧（example4 第 0 帧小球只露出一半，所以允许换个起点）
    if init_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)

    ok, frame = cap.read()
    if not ok:
        print("[错误] 读第一帧失败")
        cap.release()
        return

    if roi is None:
        # cv2.selectROI：弹出窗口让鼠标拖一个矩形框，返回 (x, y, w, h)。
        # 回车/空格确认，按 c 取消。框选时尽量贴紧小球边缘，框太松会让追踪器学到背景。
        window = "第一帧：框选小球后按回车"
        roi = cv2.selectROI(window, frame, showCrosshair=False, fromCenter=False)
        cv2.destroyWindow(window)
    if roi[2] == 0 or roi[3] == 0:
        print("[跳过] 没有框选区域")
        cap.release()
        return
    print(f"[ROI] 初始框 = {tuple(int(v) for v in roi)}")

    # 用初始框初始化追踪器：把这一小块图像的“长相”记下来，后面靠它去找球。
    tracker = create_tracker()
    tracker.init(frame, roi)

    # 输出视频：文件名、编码器、帧率、分辨率
    #   VideoWriter_fourcc(*"mp4v") = 告诉 OpenCV 用 MP4 的编码方式，写成 .mp4 才能被播放器打开
    #   帧率/分辨率一定要和输入一致，否则视频会快放/慢放或被压变形
    out_name = os.path.splitext(video_name)[0] + "_track_basic.mp4"
    out_path = os.path.join(save_dir, out_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"[错误] 无法创建输出视频：{out_path}")
        cap.release()
        return

    frame_idx = init_frame
    # 顺便把每一帧的框存成 CSV，方便事后算误差、画轨迹图
    csv_path = os.path.join(save_dir, os.path.splitext(video_name)[0] + "_track_basic.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
    csv_w = csv.writer(csv_file)
    csv_w.writerow(["frame", "status", "cx", "cy", "w", "h"])

    while True:
        # tracker.update(frame)：在当前帧里找目标，返回 (是否成功, (x, y, w, h))
        ok, box = tracker.update(frame)

        if ok:
            x, y, w, h = [int(v) for v in box]
            # 画绿框 + 中心点
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (x + w // 2, y + h // 2), 3, (0, 0, 255), -1, cv2.LINE_AA)
            status = "Tracking"
        else:
            # 追踪失败（被遮挡、出画面、动得太快）时不要崩，画个提示继续跑
            cv2.putText(frame, "Lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2, cv2.LINE_AA)
            status = "Lost"
            x = y = w = h = 0

        # 左上角叠加帧号信息，方便回看时定位
        cv2.putText(frame, f"{video_name}  frame={frame_idx}  {status}",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2, cv2.LINE_AA)

        # 把这一帧写进结果视频，然后读下一帧
        writer.write(frame)
        csv_w.writerow([frame_idx, status, x + w // 2, y + h // 2, w, h])
        frame_idx += 1

        ok, frame = cap.read()
        if not ok:
            break  # 读到结尾

    cap.release()
    writer.release()
    csv_file.close()
    print(f"[完成] 结果已保存：{out_path}  共写入 {frame_idx - init_frame} 帧")
    print(f"[完成] 轨迹已保存：{csv_path}")


if __name__ == "__main__":
    for name in VIDEOS:
        preset = PRESET_ROI.get(name)
        if preset is not None:
            track_one_video(name, init_frame=preset[4], roi=preset[:4])
        else:
            track_one_video(name, init_frame=0)
    cv2.destroyAllWindows()
    print("\n全部完成。")
