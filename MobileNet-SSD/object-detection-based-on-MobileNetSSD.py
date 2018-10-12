from imutils.video import FPS
import numpy as np
import sys
import time
import argparse
import os
import cv2 as cv


file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
outputFile = file_path + "test_out" + os.sep
threshold = 0.5  # objects' confidence threshold
inpWidth, inpHeight = 300, 300  # Width & height of network's input image
# MobileNetSSD预训练的label classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
		   "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# 随机生成各类别的bounding box颜色(选择较暗的颜色)
COLORS = np.random.uniform(0, 180, size=(len(CLASSES), 3))

# Usage example:  python object-detection-based-on-YOLOv3.py --video=run.mp4
parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

def choose_run_mode(out_path):
    if args.image:
        # Open the image file
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.image)
        global outputFile
        outputFile += args.image[:-4] + '_out.jpg'
    elif args.video:
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        outputFile += args.video[:-4] + '_out.mp4'
    else:
        # Webcam input
        cap = cv.VideoCapture(0)
        outputFile += 'webcam_out.mp4'


    return cap


def load_pretrain_model():
    # Give the configuration and weight files for the model and load the network using them.
    prototxt_file = file_path + 'Model' + os.sep + 'MobileNetSSD_deploy.prototxt'
    caffemodel_file = file_path + 'Model' + os.sep + 'MobileNetSSD_deploy.caffemodel'
    net = cv.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)
    print('MobileNetSSD caffe model loaded successfully')
    # cv.dnn.DNN_TARGET_OPENCL to run it on a GPU
    # current OpenCV version is tested only with Intel’s GPUs, it would automatically switch to CPU
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
    return net


def show_status(frame):
    # Put efficiency information. The function getPerfProfile returns the
    # overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    inf_label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, inf_label, (0, origin_h-50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if not args.image:
        fps.update()
        fps.stop()
        fps_label = "FPS: {:.2f}".format(fps.fps())
        cv.putText(frame, fps_label, (0, origin_h-25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


if __name__ == '__main__':
    # -----main process---------------------------------------------------
    # choosing image/video/webcam 并保存输出文件
    cap = choose_run_mode(outputFile)
    # load YOLOv3 darknet model
    net = load_pretrain_model()

    fps = FPS().start()
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc(*'mp4v'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                 round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            break
        # 原始图像的像素尺度
        origin_h, origin_w = frame.shape[:2]
        # 不同算法及训练模型的blobFromImage参数不同，可访问opencv的github地址查询
        # https://github.com/opencv/opencv/tree/master/samples/dnn
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0 / 127.5, (300, 300), 127.5)
        net.setInput(blob)
        # 前向传播计算输出，YOLOv3有三个输出层
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                idx = int(detections[0, 0, i, 1])
                bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
                x_start, y_start, x_end, y_end = bounding_box.astype('int')
                # 显示image中的object类别及其置信度
                label = '{0}: {1:.2f}%'.format(CLASSES[idx], confidence * 100)
                # 画bounding box
                cv.rectangle(frame, (x_start, y_start), (x_end, y_end), COLORS[idx], 2)
                # 画文字的填充矿底色
                cv.rectangle(frame, (x_start, y_start - 18), (x_end, y_start), COLORS[idx], -1)
                # detection result的文字显示
                cv.putText(frame, label, (x_start + 2, y_start - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # 计算并显示Inference time及实时FPS
        show_status(frame)
        # Write the frame with the detection boxes
        if (args.image):
            cv.imwrite(outputFile, frame)
        else:
            vid_writer.write(frame)

        winName = 'SSD object detection in OpenCV'
        # cv.namedWindow(winName, cv.WINDOW_NORMAL)
        cv.imshow(winName, frame)

    if not args.image:
        vid_writer.release()
        cap.release()
    cv.destroyAllWindows()