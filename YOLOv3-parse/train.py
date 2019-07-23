"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    # 创建model，默认9个anchor的为yolo，6个anchor的为tiny-yolo3
    is_tiny_version = len(anchors) == 6
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2, weights_path='model_data/yolo.h5')
        # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    # 只存储权重（save_weights_only）；只存储最优结果（save_best_only）；每隔3个epoch存储一次（period）
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # 数据集划分
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)  # 验证集10%
    num_train = len(lines) - num_val  # 训练集90%

    # 训练过程分为两个阶段
    # 阶段一：在部分网络被冻结的情况下训练，只训练输出层的参数.这一步能获得效果不错的model
    if True:
        model.compile(optimizer=Adam(lr=1e-3),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # use custom yolo_loss Lambda layer.

        # 把y_true当成输入，作为模型的多输入，把loss封装为层，作为输出；
        # 在模型中，最终输出的y_pred就是loss；
        # 在编译时，将loss设置为y_pred即可，无视y_true；
        # 在训练时，随意添加一个符合结构的y_true即可。

        batch_size = 16
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # 训练阶段2：将全部的权重都设置为可训练，对模型进行进一步微调，学习率减小。
    # 如果效果不理想，增长训练时间
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 16  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=100,
                            initial_epoch=50,
                            # reduce_lr：当评价指标不提升时，减少学习率，每次减少10%，当验证损失值，持续3次未减少时，则终止训练
                            # early_stopping：当验证集损失值，连续增加小于0时，持续10个epoch，则终止训练
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape,
                 anchors,
                 num_classes,
                 load_pretrained=True,
                 freeze_body=2,  # 冻结模式
                 weights_path='model_data/yolo_weights.h5'  # 预训练模型的权重
                 ):
    """create the training model"""
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    # ground truth
    # 第1位是输入样本数，第2~3位是特征图的尺寸，第4位是anchor数，第5位是类别(n)+4个框值+框的置信度(是否含有物体)
    y_true = [Input(shape=(h//{0: 32, 1: 16, 2: 8}[l], w//{0: 32, 1: 16, 2: 8}[l],
                           num_anchors//3, num_classes+5)) for l in range(3)]
    # [ < tf.Tensor'input_1:0' shape = (?, 13, 13, 3, 85) dtype = float32 >,
    # < tf.Tensor'input_2:0'shape = (?, 26, 26, 3, 85) dtype = float32 >,
    # < tf.Tensor'input_3:0'shape = (?, 52, 52, 3, 85) dtype = float32 >]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        # 1是冻结DarkNet53的层（185）；
        # 2是冻结全部，只保留最后3个1*1的用于预测的卷积层（249）；
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # 构建loss function
    # 损失函数yolo_loss封装自定义Lambda的损失层中，作为模型的最后一层，参于训练。
    # 损失层Lambda的输入是已有模型的输出model_body.output和真值y_true，输出是1个值，即损失值。
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})\
        ([*model_body.output, *y_true])
    # 其中，model_body.output是已有模型的预测值，y_true是真实值，两者的格式相同，如下：
    # model_body: [(?, 13, 13, 255), (?, 26, 26, 255), (?, 52, 52, 255)]
    # y_true: [(?, 13, 13, 255), (?, 26, 26, 255), (?, 52, 52, 255)]
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0: 32, 1: 16}[l], w//{0: 32, 1: 16}[l],
                           num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """data generator for fit_generator"""
    # 在数据生成器data_generator中，数据的总行数是n，
    # 循环输出固定批次数batch_size的图片数据image_data和标注框数据box_dat
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                # 在第0次时，将数据洗牌shuffle，
                np.random.shuffle(annotation_lines)
            # 调用get_random_data解析annotation_lines[i]，
            # 生成图片image和标注框box，添加至各自的列表image_data和box_data中
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        # 将image_data和box_data都转换为np数组
        image_data = np.array(image_data)  # (16, 416, 416, 3)
        box_data = np.array(box_data)   # (16, 20, 5) 每个图片最多含有20个框

        # y_true是3个预测特征的列表
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        # 最终输出：图片数据image_data、真值y_true、每个图片的损失值np.zeros。
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """验证输入参数是否正确，再调用data_generator，这也是wrapper函数的常见用法
    ------------
    annotation_lines：标注数据的行，每行数据包含图片路径，和框的位置信息
    """
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()
