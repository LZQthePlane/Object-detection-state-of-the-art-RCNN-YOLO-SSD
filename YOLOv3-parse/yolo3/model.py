"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose

"""for training-------------------------------------------------------"""


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    # 将核权重矩阵的正则化，使用L2正则化，参数是5e - 4，操作对象是w参数；
    # Padding一般使用same模式，当步长为(2, 2)时，使用valid模式。避免在降采样中引入无用的边界信息；
    # 其余参数不变，都与二维卷积操作Conv2D()一致；
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    # 在第1个卷积操作DarknetConv2D_BN_Leaky中，是3个操作的组合，即：
    # 1个Darknet的2维卷积Conv2D层，即DarknetConv2D；
    # 1个批正则化层，即BatchNormalization()，操作对象是网络层的输入数据X；
    # 1个LeakyReLU层，斜率是0.1，LeakyReLU是ReLU的变换；
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(DarknetConv2D(*args, **no_bias_kwargs),
                   BatchNormalization(),
                   LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    """残差模块，与标准残差模块不同
    A series of resblocks starting with a downsampling Convolution2D"""
    # ZeroPadding2D：填充x的边界为0，由(?, 416, 416, 32)转换为(?, 417, 417, 32)。因为下一步卷积操作的步长为2，所以图的边长需要是奇数；
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    # DarknetConv2D_BN_Leaky：DarkNet的2维卷积操作，核是(3, 3)，
    # 注意步长是(2,2)，这是降采样，由于此函数一共只被调用5次，即darknet-53共有5次下采样操作
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    # compose：输出预测图y，功能是组合函数，先执行1x1的卷积操作，再执行3x3的卷积操作，
    # filter先降低2倍后恢复，最后与输入相同，都是64；
    for i in range(num_blocks):
        y = compose(DarknetConv2D_BN_Leaky(num_filters//2, (1, 1)),
                    DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        # x = Add()([x, y])：残差操作，将x的值与y的值相加。残差操作可以避免，在网络较深时所产生的梯度弥散问题。
        x = Add()([x, y])
    return x


def darknet_body(x):
    """全卷积网络，采用了darknet53层中的前52层，除去最后一层的平均池化层"""
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)  # 416*416*3
    x = resblock_body(x, 64, 1)  # 1+1*2=3
    x = resblock_body(x, 128, 2)  # 1+2*2=5
    x = resblock_body(x, 256, 8)  # 1+8*2=17
    x = resblock_body(x, 512, 8)  # 1+8*2=17
    x = resblock_body(x, 1024, 4)  # 1+4*2=9
    return x


def make_last_layers(x, num_filters, out_filters):
    """用于构造3个尺度的输出"""
    # 第1步，x执行多组1x1的卷积操作和3x3的卷积操作，filter先扩大再恢复，
    # 最后与输入的filter保持不变，仍为512，则x由(?, 13, 13, 1024)转变为(?, 13, 13, 512)
    x = compose(DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    # x先执行3x3的卷积操作，再执行不含BN和Leaky的1x1的卷积操作，
    # 作用类似于全连接操作，生成预测矩阵y
    y = compose(DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
                DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    # 先建立基本的darknet网络结构，进行降采样
    # 输出维度（?,13,13,1024），yolo3详细网络结构中的第74层
    darknet = Model(inputs, darknet_body(inputs))

    # 接下来输出3个不同尺度的检测图，用于检测不同大小的物体。
    # 调用3次make_last_layers，产生3个检测图，即y1、y2和y3

    # 输出的x是(?, 13, 13, 512)，输出的y是(?, 13, 13, 255)。
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    # 输出的x是(?, 26, 26, 256)，输出的y是(?, 26, 26, 255)
    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)),  # 将x由512的通道数，转换为256的通道数
                UpSampling2D(2))(x)  # 将x由13x13的结构，转换为26x26的结构
    x = Concatenate()([x, darknet.layers[152].output])  # 在拼接之后，输出的x的格式是(?, 26, 26, 768)
    # 这样做的目的是：将Darknet最底层的高级抽象信息darknet.output，
    # 经过若干次转换之后，除了输出给第1个检测部分，还被用于第2个检测部分，
    # 经过上采样，与Darknet骨干中，倒数第2次降维的数据拼接，共同作为第2个检测部分的输入。
    # 底层抽象特征含有全局信息，中层抽象特征含有局部信息，这样拼接，两者兼顾，用于检测较小的物体。
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    # 生成y3是(?, 52, 52, 255)，忽略x的输出
    x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)),
                UpSampling2D(2))(x)  # x经过128个filter的卷积，再执行上采样，输出为(?, 52, 52, 128)；
    x = Concatenate()([x, darknet.layers[92].output])  # 两者拼接之后，x是(?, 52, 52, 384)
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1, y2, y3])


def tiny_yolo_body(inputs, num_anchors, num_classes):
    """Create Tiny YOLO_v3 model CNN body in keras."""
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3, 3)),
            DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3, 3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1, 1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1, 1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3, 3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1, 1)))([x2, x1])

    return Model(inputs, [y1, y2])


# 此函数，当calc_loss为True时，用于训练过程，当calc_loss为False时，用于推理过程
def olo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """
    将预测图yolo_outputs[l]，拆分为边界框的起始点xy、宽高wh、置信度confidence和类别概率class_probs

    parameters:
    -----------------------
    feats或yolo_outputs[l]：是模型的输出，第i个预测图，如(?, 13, 13, 255)
    anchors或anchors[anchor_mask[l]]：第l个anchor box，如[(116, 90), (156,198), (373,326)]
    """
    num_anchors = len(anchors)
    # 将anchors转换为与预测图feats维度相同的Tensor，即anchors_tensor的结构是(1, 1, 1, 3, 2)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # 获取网格的尺寸grid_shape，即预测图feats的第1~2位，如13x13；
    grid_shape = K.shape(feats)[1:3]  # height, width
    # grid_y和grid_x用于生成网格grid，通过arange、reshape、tile的组合，
    # 创建y轴的0~12的组合grid_y，再创建x轴的0~12的组合grid_x，将两者拼接concatenate，就是grid
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    # grid是遍历二元数值组合的数值，结构是(13, 13, 1, 2)；
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # 下一步，将feats的最后一维展开，将anchors与其他数据（类别数+4个框值+框置信度）分离:(?, 13, 13, 3, 85)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 计算起始点xy、宽高wh、框置信度box_confidence和类别置信度box_class_probs
    # 这4个值box_xy, box_wh, confidence, class_probs的范围均在0~1之间

    # 起始点xy：将feats中xy的值，经过sigmoid归一化，再加上相应的grid的二元组，再除以网格边长，归一化
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    # 宽高wh：将feats中wh的值，经过exp正值化，再乘以anchors_tensor的anchor box，再除以图片宽高，归一化
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    # 框置信度box_confidence：将feats中confidence值，经过sigmoid归一化
    box_confidence = K.sigmoid(feats[..., 4:5])
    # 类别置信度box_class_probs：将feats中class_probs值，经过sigmoid归一化
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss is True:
        # 网格grid：结构是(13, 13, 1, 2)，数值为0~12的全遍历二元组；
        # 预测值feats：经过reshape变换，将255维数据分离出3维anchors，结构是(?, 13, 13, 3, 85)
        # box_xy和box_wh归一化的起始点xy和宽高wh，xy的结构是(?, 13, 13, 3, 2)，wh的结构是(?, 13, 13, 3, 2)；
        # box_xy的范围是(0~1)，box_wh的范围是(0~1)；即bx、by、bw、bh计算完成之后，再进行归一化
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """将原始box 信息 转换成训练时输入的格式

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
    检测框，批次数16，最大框数20，每个框5个值，4个边界点和1个类别序号，如(16, 20, 5)；

    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    """

    # 检查有无异常数据 即txt提供的box id 是否存在大于 num_class的情况
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    # 真值框，左上和右下2个坐标值和1个类别，如[184, 299, 191, 310, 0.0]，结构是(16, 20, 5)
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    # 得到中心点坐标，结构是(16, 20, 2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    # 得到box宽高，结构是(16, 20, 2)
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 中心坐标 和 宽高 都变成 相对于input_shape的比例
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    # 现在true_boxes 中的数据成了 x,y,w,h 如[0.449, 0.730, 0.016, 0.026, 0.0]

    # 这个m是batch的大小 即是输入图片的数量
    m = true_boxes.shape[0]
    # grid_shape是input_shape等比例降低，即[[13, 13], [26, 26], [52, 52]]
    grid_shapes = [input_shape//{0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    # y_true是全0矩阵（np.zeros）列表，即[(16, 13, 13, 3, 85), (16, 26, 26, 3, 85), (16, 52, 52, 3, 85)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5+num_classes),
                       dtype='float32') for l in range(num_layers)]

    # 将anchors增加1维expand_dims，由(9,2)转为(1,9,2)
    anchors = np.expand_dims(anchors, 0)

    anchor_maxes = anchors / 2.  # 是anchors值除以2
    anchor_mins = -anchor_maxes  # 是负的anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0  # 将boxes_wh中宽w大于0的位，设为True，即含有box，结构是(16,20)

    # 循环m处理批次中的每个图像和标注框
    for b in range(m):
        # 只选择存在标注框的wh，例如：wh的shape是(7,2) 7个有效box，2个维度(w h)
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # wh倒数第2个添加1位，即(7,2)->(7,1,2)
        wh = np.expand_dims(wh, -2)
        # box_maxes和box_mins，与anchor_maxes和anchor_mins的操作类似
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # 下面就是在计算 ground_true与anchor box的交并比，计算方式很巧妙
        # box_mins的shape是(7,1,2)，anchor_mins的shape是(1,9,2)，intersect_mins的shape是(7,9,2)，即两两组合的值
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        # intersect_area的shape是(7,9)；
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # box_area的shape是(7, 1)；
        box_area = wh[..., 0] * wh[..., 1]
        # anchor_area的shape是(1, 9)
        anchor_area = anchors[..., 0] * anchors[..., 1]
        # iou的shape是(7, 9)，即anchor box与检测框box，两两匹配的iou值
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # 对于每个真实box 找到最匹配的anchor
        # best_anchor 的格式为 bounding_box id -> anchor_id
        best_anchor = np.argmax(iou, axis=-1)

        # 遍历所有 匹配的anchor
        # t是bounding box id，n是anchor_id
        for t, n in enumerate(best_anchor):
            # 遍历anchor 尺寸 3个尺寸
            # 因为此时box 已经和一个anchor box匹配上，看这个anchor box属于那一层，小，中，大，然后将其box分配到那一层
            for l in range(num_layers):
                # 如果匹配的这个n即 anchor id在 l这一层，那么进行赋值，否则保持默认值0
                if n in anchor_mask[l]:
                    # np.floor 返回不大于输入参数的最大整数。 即对于输入值 x ，将返回最大的整数 i ，使得 i <= x。
                    # true_boxes x,y,w,h, 此时x y w h都是相对于整张图像的
                    # 第b个图像 第 t个 bounding box的 x 乘以 第l个grid shape的x（grid shape 格式是hw，因为input_shape格式是hw）

                    # 找到这个bounding box落在哪个cell的中心
                    i = np.floor(true_boxes[b, t, 0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1]*grid_shapes[l][0]).astype('int32')
                    # 找到n 在 anchor_box的索引位置
                    k = anchor_mask[l].index(n)
                    # 得到box的id
                    c = true_boxes[b, t, 4].astype('int32')
                    # 第b个图像 第j行 i列 第k个anchor x，y，w，h,confindence,类别概率
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    # 置信度是1 因为含有目标
                    y_true[l][b, j, i, k, 4] = 1
                    # 类别的one-hot编码
                    y_true[l][b, j, i, k, 5+c] = 1
    # y_true的第0和1位是中心点xy，范围是(0~1)，第2和3位是宽高wh，范围是0~1，第4位是置信度1或0，第5~n位中某一个位置是1其余为0
    return y_true


def box_iou(b1, b2):
    """Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    """

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    """参考
    https://blog.csdn.net/weixin_42078618/article/details/85005428

    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    返回值
    -------
    loss: tensor, shape=(1,)

    """
    num_layers = len(anchors)//3  # 层的数量，是anchors数量的3分之1
    # 分离args，前3个是yolo_outputs预测值，后3个是y_true真值
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    # 678对应13x13，345对应26x26，012对应52x52
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    # K.shape(yolo_outputs[0])[1:3]，第1个预测矩阵yolo_outputs[0]的结构（shape）的第1~2位，即(?, 13, 13, 18)中的(13, 13)。
    # 再x32，就是YOLO网络的输入尺寸，即(416, 416)，因为在网络中，含有5个步长为(2, 2)的卷积操作，降维32=5^2倍
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    # grid_shapes：即[(13, 13), (26, 26), (52, 52)]
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]

    m = K.shape(yolo_outputs[0])[0]  # 输入模型的图片总量，即批次数 batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))  # m的float类型

    loss = 0
    for l in range(num_layers):
        # 获取物体置信度object_mask，最后1个维度的第4位，第0~3位是框，第4位是物体置信度
        object_mask = y_true[l][..., 4:5]
        # 类别置信度true_class_probs，最后1个维度的第5位
        true_class_probs = y_true[l][..., 5:]
        # 接着，调用yolo_head重构预测图
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l], anchors[anchor_mask[l]],
                                                     num_classes, input_shape, calc_loss=True)
        # 再将xy和wh组合成预测框pred_box，结构是(?, 13, 13, 3, 4)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # 生成真值数据
        # 在网格中的中心点xy，偏移数据，值的范围是0~1；y_true的第0和1位是中心点xy的相对位置，范围是0~1
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        # 在网络中的wh针对于anchors的比例，再转换为log形式，范围是有正有负；y_true的第2和3位是宽高wh的相对位置，范围是0~1
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        # box_loss_scale = 2 - w * h，计算wh权重，取值范围(1~2)
        box_loss_scale = 2 - y_true[l][..., 2:3]*y_true[l][..., 3:4]

        # 接着，根据IoU忽略阈值生成ignore_mask，将预测框pred_box和真值框true_box计算IoU，
        # 抑制不需要的anchor框的值，即IoU小于最大阈值的anchor框。
        # ignore_mask的shape是(?, ?, ?, 3, 1)，第0位是批次数，第1~2位是特征图尺寸
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        # 中心点的损失值。object_mask是y_true的第4位，即是否含有物体，含有是1，不含是0。
        # box_loss_scale的值，与物体框的大小有关，2减去相对面积，值得范围是(1~2)。binary_crossentropy是二值交叉熵
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
        # 宽高的损失值。除此之外，额外乘以系数0.5，平方K.square()
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[..., 2:4])
        # 框的损失值。两部分组成，第1部分是存在物体的损失值，
        # 第2部分是不存在物体的损失值，其中乘以忽略掩码ignore_mask，忽略预测框中IoU大于阈值的框
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask
        # class_loss：类别损失值
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        # 将各部分损失值的和，除以均值，累加，作为最终的图片损失值
        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss


"""for inference------------------------------------------------------"""


def yolo_eval(yolo_outputs,  # YOLO模型的输出，3个尺度的列表，即13-26-52，最后1维是预测值，由255=3x(5+80)组成，
                             # 3是每层的anchor数，5是4个框值xywh和1个框中含有物体的置信度，80是COCO的类别数
                             # 格式[(?, 13, 13, 255), (?, 26, 26, 255), (?, 52, 52, 255)]
              anchors,  # [(10,13), (16,30) ...)]
              num_classes,
              image_shape,  # 图像真实尺寸大小,placeholder类型的TF参数
              max_boxes=20,  # 图中 每个class的最多的检测框数，20个
              score_threshold=.6,
              iou_threshold=.5):
    """求出input_image中的boxes, scores和classes"""
    num_layers = len(yolo_outputs)
    # 将anchors划分为3个层，第1层13x13是678，第2层26x26是345，第3层52x52是012
    # 特征图越大，13->52，检测的物体越小，需要的anchors越小，所以anchors列表以倒序赋值。
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    # 输入图像的尺寸，也就是第0个特征图的尺寸乘以32，即13x32 = 416，这与Darknet的网络结构有关。
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32

    boxes = []
    box_scores = []
    # 在YOLO的第l层输出yolo_outputs中，调用yolo_boxes_and_scores()，提取框_boxes和置信度_box_scores，
    # 将3个层的框数据放入列表boxes和box_scores，再拼接concatenate展平，输出的数据就是所有的框和置信度。
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]],
                                                    num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # concatenate将相同维度的数据元素连接到一起
    boxes = K.concatenate(boxes, axis=0)  # (?, 4)  ?是框数
    box_scores = K.concatenate(box_scores, axis=0)  # (?, 80)

    # mask，过滤小于置信度阈值的框，只保留大于置信度的框，mask掩码；
    # 如果threshold的设置大于0.5，即可保证每个特征图cell中最多有一个object被识别
    mask = box_scores >= score_threshold  # (?, 80)
    # 每张图片的最大检测框数，max_boxes是20；
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        # 通过掩码mask和类别c，筛选框class_boxes和置信度class_box_scores；
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # 分别对每一类object进行非极大抑制
        # 通过极大值抑制，筛选出框boxes的NMS索引nms_index；
        index_after_nms = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        # gather函数以索引选择列表元素，筛选出框class_boxes和置信度class_box_scores
        class_boxes = K.gather(class_boxes, index_after_nms)
        class_box_scores = K.gather(class_box_scores, index_after_nms)
        # 再生成类别信息classes；
        classes = K.ones_like(class_box_scores, 'int32') * c

        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    # 将多个类别的数据组合，生成最终的检测数据框，并返回。
    # 这里的?即为最终输出的box的个数，虽然进行了num_classes次循环，但其中很多次循环得到的是空的tensor
    boxes_ = K.concatenate(boxes_, axis=0)  # (?, 4)
    scores_ = K.concatenate(scores_, axis=0)  # (?,)
    classes_ = K.concatenate(classes_, axis=0)  # (?,)

    return boxes_, scores_, classes_


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """提取框boxes和其置信度"""
    # box_xy是box的中心坐标 (0~1)相对位置；box_wh是box的宽高 (0~1)相对值；
    # box_confidence是框中是否存在物体的置信度；box_class_probs是类别置信度；
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # 将box_xy和box_wh的(0~1)相对值，转换为真实坐标，输出boxes是(y_min, x_min, y_max, x_max)的值
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # 将不同网格的值展平为框的列表，即(?,13,13,3,4)->(?,4)；
    boxes = K.reshape(boxes, [-1, 4])
    # box_scores是框置信度与类别置信度的乘积，再reshape展平，(?, 80)；
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """# 将box_xy和box_wh的(0~1)相对值，转换为真实坐标"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes