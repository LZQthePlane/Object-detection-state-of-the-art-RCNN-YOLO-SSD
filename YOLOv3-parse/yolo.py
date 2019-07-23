# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

"""设置"""
score_thres = 0.6
iou_thres = 0.5
model_image_size = (416, 416)
# model_image_size = (None, None)
gpu_num = 1


class YOLO(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)  # update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.colors = self._get_colors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        """获取classes，默认为coco_classes"""
        classes_path = os.path.expanduser('model_data/coco_classes.txt')
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """获取anchors，[[10, 13], [16, 30], ...]"""
        # 框1~3在大尺度52x52特征图中使用，框4~6是中尺度26x26，框7~9是小尺度13x13；
        # 大尺度特征图用于检测小物体，小尺度检测大物体；
        anchors_path = os.path.expanduser('model_data/yolo_anchors.txt')
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def _get_colors(self):
        """为不同object生成不同颜色的bounding box"""
        # 将HSV的第0位H值，按1等分，其余SV值，均为1，生成一组HSV列表；
        # 选择HSV划分，而不是RGB的原因是，HSV的颜色值偏移更好，画出的框，颜色更容易区分。
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        return colors

    def generate(self):
        model_path = os.path.expanduser('model_data/yolov3.h5')
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        try:
            # 加载模型
            self.yolo_model = load_model(model_path, compile=False)
        except:
            # 如果模型文件有问题，通过构建模型并加入参数的方式进行加载
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights('model_data/yolov3.h5')  # make sure model, anchors and classes match
        else:
            # 如果模型最后一层的输出参数数量不等于预设值的数量，则进行报警
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        # 求出input_image中的boxes, scores和classes
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=score_thres, iou_threshold=iou_thres)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        # 将图像等比例转换为检测尺寸，检测尺寸需要是32的倍数，若不是，则对图像周围进行填充；
        if model_image_size != (None, None):
            assert model_image_size[0] % 32 == 0, '必须为32的倍数'
            assert model_image_size[1] % 32 == 0, '必须为32的倍数'
            boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.  # 转换0~1
        image_data = np.expand_dims(image_data, 0)  # 添加批次维度，将图片增加1维 (?, 416, 416, 3)

        # Feed数据
        out_boxes, out_scores, out_classes = \
            self.sess.run([self.boxes, self.scores, self.classes],
                          feed_dict={self.yolo_model.input: image_data,
                                     self.input_image_shape: [image.size[1], image.size[0]],
                                     K.learning_phase(): 0  # 只是一个flag：0表示测试模式，1表示训练模式
                                     })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # 设置边框宽度，类别文字格式等
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            # 防止边框跑出图片外，重新定义一下
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            # 定义标签文字的位置
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 画框，画文字背景框，写文字标签
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print('程序运行时长：{}'.format(end - start))
        return image

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


def detect_img(yolo, img_path):
    try:
        image = Image.open(img_path)
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
    yolo.close_session()

