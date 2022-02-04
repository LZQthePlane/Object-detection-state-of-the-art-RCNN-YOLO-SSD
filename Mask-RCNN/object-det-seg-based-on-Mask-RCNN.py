import cv2 as cv
import argparse
import numpy as np
import os.path
import sys
from pathlib import Path
from imutils.video import FPS

file_path = Path.cwd()
out_file_path = Path(file_path / "test_out/")

# 初始化相关参数
confThreshold = 0.7  # 若检测的detection置信度低于此值，忽略显示其bounding box
maskThreshold = 0.2  # 检测到的mask区域内的像素点置信度低于此值，则认为该点不属于该mask区域内
classes = []  # mask-coco数据集包括的种类
colors = []  # 用于分类的颜色集合


def choose_run_mode():
    """
    choose image or video or webcam as input
    """
    global out_file_path
    if args.image:
        # Open the image file
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.image)

        out_file_path = str(out_file_path / (args.image[:-4] + '_out.jpg'))
    elif args.video:
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        out_file_path = str(out_file_path / (args.video[:-4] + '_out.mp4'))
    else:
        # Webcam input
        cap = cv.VideoCapture(0)
        out_file_path = str(out_file_path / 'webcam_out.mp4')
    return cap


def load_coco_classes():
    """
    Load names of classes
    """
    classes_file = str(file_path/"model/mscoco_labels.names")
    global classes
    with open(classes_file, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')


def load_colors():
    """
    load the coco-colors
    """
    colors_file = str(file_path/"model/colors.txt")
    with open(colors_file, 'rt') as f:
        colors_str = f.read().rstrip('\n').split('\n')

    global colors
    for i in range(len(colors_str)):
        rgb = colors_str[i].split(' ')
        color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
        colors.append(color)


def load_pretrain_model():
    """
    load the pre-trained graph models and set net
    """
    text_graph = str(file_path/'model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
    model_weights = str(file_path/"model/frozen_inference_graph.pb")
    net = cv.dnn.readNetFromTensorflow(model_weights, text_graph)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
    return net


def set_video_writer(cap, write_fps=25):
    """
    Get the video writer initialized to save the output video
    """
    vid_writer = cv.VideoWriter(out_file_path, cv.VideoWriter_fourcc(*'mp4v'), write_fps,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                 round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    return vid_writer


def visualize(boxes, masks):
    """
    For each frame, extract the bounding box and pre_mask for each detected object
    the format of pre_mask is NxCxHxW where
    N - number of detected boxes,
    C - number of classes (excluding background)
    H x W is the segmentation shape (输出的原始图是15*15的map)
    """
    num_detections = boxes.shape[2]

    for i in range(num_detections):
        # 获取第i个detection的bbox和mask
        box = boxes[0, 0, i]
        pre_mask = masks[i]
        score = box[2]

        if score > confThreshold:
            class_id = int(box[1])
            # 还原bounding box坐标
            left = int(origin_w * box[3])
            top = int(origin_h * box[4])
            right = int(origin_w * box[5])
            bottom = int(origin_h * box[6])

            left = max(0, min(left, origin_w-1))
            top = max(0, min(top, origin_h-1))
            right = max(0, min(right, origin_w-1))
            bottom = max(0, min(bottom, origin_h-1))

            # 根据detection的object id提取对应的mask输出
            pre_mask = pre_mask[class_id]

            # Draw bounding box, colorize and show the pre_mask on the image
            draw_box_mask(frame, class_id, score, left, top, right, bottom, pre_mask)


def draw_box_mask(frame, class_id, conf, left, top, right, bottom, pre_mask):
    """
    Draw the predicted bounding box and mask on the image
    """
    # Draw bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    # Print a label of class.
    class_label = '%.2f' % conf
    if classes:
        assert(class_id < len(classes))
        class_label = '%s:%s' % (classes[class_id], class_label)
    # Display the label at the top of the bounding box
    label_size, base_line = cv.getTextSize(class_label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv.rectangle(frame, (left, top - round(1.5*label_size[1])), (left + round(1.5*label_size[0]), top + base_line),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, class_label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # Draw the contours of detections
    # 将mask从原始输出的15*15尺度 根据bounding box的大小进行还原
    mask = cv.resize(pre_mask, (right - left + 1, bottom - top + 1))
    # 其中 大于threshold值的区域置为1，其余置为0
    mask_bool = (mask > maskThreshold)
    # 取出mask包围为区域，认为是region of interest，roi存储的是mask的原始像素值
    roi = frame[top: bottom+1, left: right+1][mask_bool]

    color = colors[class_id % len(colors)]
    # mask内部的区域 用浅色填充
    frame[top:bottom+1, left:right+1][mask_bool] = ([0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(np.uint8)
    # 根据mask范围 画contour包络线
    mask = mask_bool.astype(np.uint8)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame[top: bottom+1, left: right+1], contours, -1, color, 3, cv.LINE_8, hierarchy, 2)


def show_status(frame):
    """
    show the Inference time and real-time FPS
    """
    # Put efficiency information. The function getPerfProfile returns the
    # overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    inf_label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, inf_label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if not args.image:
        fps.update()
        fps.stop()
        fps_label = "FPS: {:.2f}".format(fps.fps())
        cv.putText(frame, fps_label, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mask-RCNN object detection and segmentation')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--video', help='Path to video file.')
    args = parser.parse_args()

    cap = choose_run_mode()
    load_coco_classes()
    load_colors()
    net = load_pretrain_model()
    fps = FPS().start()
    video_writer = set_video_writer(cap)
    while cv.waitKey(1) < 0:
        has_frame, frame = cap.read()
        # Stop the program if reached end of video
        if not has_frame:
            print("Output file is stored as ", out_file_path)
            cv.waitKey(3000)
            break

        # 原始图像的像素尺度
        origin_h, origin_w = frame.shape[:2]
        # 不同算法及训练模型的blobFromImage参数不同，可访问opencv的github地址查询
        # https://github.com/opencv/opencv/tree/master/samples/dnn
        blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)
        # Set the input to the network
        net.setInput(blob)
        # Run the forward pass to get output from the output layers
        boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
        # Extract the bounding box and mask for each of the detected objects
        visualize(boxes, masks)
        # 计算并显示Inference time及实时FPS
        show_status(frame)

        # Write the frame with the detection boxes
        if args.image:
            cv.imwrite(out_file_path, frame.astype(np.uint8))
        else:
            video_writer.write(frame.astype(np.uint8))

        winName = 'Mask-RCNN Object-detection and Segmentation'
        cv.imshow(winName, frame)

    if not args.image:
        video_writer.release()
        cap.release()
    cv.destroyAllWindows()


