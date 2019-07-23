
def bb_iou(box_a, box_b):
    """
    要点1：在图像坐标系中，原点是在左上角
    """
    # 获取各个box的左上角及右下角坐标
    a_x1, a_y1, a_x2, a_y2 = box_a
    b_x1, b_y1, b_x2, b_y2 = box_b

    # 计算intersection面积
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    """
    要点2：之所以要+1，是因为像素点是一个个cell而不是point
    假设有一个box为（0，0，2，2），其实际上是一个3*3的area
    """
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    # 计算各个box面积，以求得union面积
    box_a_area = (a_x2 - a_x1 + 1) * (a_y2 - a_y1 + 1)
    box_b_area = (b_x2 - b_x1 + 1) * (b_y2 - b_y1 + 1)
    union = box_a_area + box_b_area - inter_area

    # 计算交并比
    iou = inter_area / float(union)
    return iou


box1 = (661, 27.8, 879, 47)
box2 = (700, 32, 709, 50)
res_ = bb_iou(box_a=box1, box_b=box2)
print(res_)
