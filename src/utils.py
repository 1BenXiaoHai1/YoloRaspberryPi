import numpy as np
import cv2
import torch
import random
import os

# 将图片变成形状为(640,640)的正方形
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    将图片变成形状为(640,640)的正方形
    模型输入图片的尺寸为正方形，而数据集中的图片一般为长方形，粗暴的resize会使得图片失真，
    采用letterbox可以较好的解决这个问题。该方法可以保持图片的长宽比例，剩下的部分采用灰色填充。
    :param img:输入图片
    :param new_shape:想要的图片形状
    :param color:指定了添加边框时使用的颜色，默认为灰色 (114,114,114)
    :param auto:是否使用自动模式来决定如何调整图像尺寸。在自动模式下，函数将计算最小矩形填充区域。
    :param scaleFill:指示是否拉伸图像以填充整个目标尺寸，忽略原始图像的宽高比。
    :param scaleup:指示是否允许对图像进行放大。如果设置为 False，则只会对图像进行缩小操作。
    :param stride:指定了调整图像尺寸时的步长。步长通常用于确保图像的尺寸是该值的整数倍
    :return:目标大小的img, 比例因子ratio(数组，高、宽), 填充大小(dw, dh)
    """
    shape = img.shape[:2]  # 获取当前图片的高宽，img的类型为(1080, 810, 3)，最后的3是通道数

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算比例因子r = new / old，用于将原始形状shape缩放到目标形状new_shape
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择小的，如果过大则可能需要裁剪，这可能导致目标丢失
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # 计算padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # round函数是四舍五入
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算出高宽的padding大小
    # 根据不同的模式（自动或拉伸）来调整图像的尺寸
    if auto:  # 自动模式，minimum rectangle,最小填充
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding,确定填充的最小矩形区域，确保处理后的图像尺寸是步长的整数倍
    elif scaleFill:  # 拉伸模式，stretch，直接使用目标尺寸，并计算宽高比以进行适当的缩放
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    # divide padding into 2 sides,将总的填充量平均分配到图像的两侧（左边和右边，上边和下边）
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # cv2.INTER_LINEAR表示使用线性插值方法进行缩放
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 为图像img添加边框
    return img, ratio, (dw, dh)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None): # 将坐标coords(x1y1x2y2)从img1_shape尺寸缩放到img0_shape尺寸，将预测坐标从feature map映射回原图
    """
    Rescale coords (xyxy) from img1_shape to img0_shape
    :param img1_shape: coords相对于的shape大小
    :param coords: 要进行缩放的box坐标信息 x1y1x2y2
    :param img0_shape: 要将coords缩放到相对的目标shape大小
    :param ratio_pad: 缩放比例gain和pad值   None就先计算gain和pad值再pad+scale  不为空就直接pad+scale
    :return:缩放后的预测框
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new，取高宽比例较小的。如果不够还可以pad，直接取大的进行裁剪有可能减去目标
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0] # 指定比例
        pad = ratio_pad[1] # 指定pad值

    # 因为pad = img1_shape - img0_shape 所以要把尺寸从img1 -> img0 就同样也需要减去pad
    # 如果img1_shape>img0_shape  pad>0   coords从大尺寸缩放到小尺寸 减去pad 符合
    # 如果img1_shape<img0_shape  pad<0   coords从小尺寸缩放到大尺寸 减去pad 符合
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape) # 防止放缩后的坐标过界 边界处直接剪切
    return coords

def clip_coords(boxes, img_shape):
    """
    Clip bounding xyxy bounding boxes to image shape (height, width)
    :param boxes:边界框(x1, y1, x2, y2)
    :param img_shape:
    :return:无返回值，clamp_()是一个原地操作
    """
    # clamp_ 是一个原地操作（in-place operation），它将 boxes 中 x1 坐标小于 0 的值设为 0，
    # 大于 img_shape[1]（宽度减 1）的值设为 img_shape[1]
    boxes[:, 0].clamp_(0, img_shape[1])  # x1,将所有边界框的 x1 坐标限制在 [0, width-1] 的范围内。
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def to_numpy(tensor):
    # return tensor.detach().cuda().numpy() if tensor.requires_grad else tensor.cuda().numpy()
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def xywh2xyxy(x):
    """
    预测框从[x, y, w, h]描述方式转换为 [x1, y1, x2, y2]描述方式。
    xy1=top-left, xy2=bottom-right
    :param x: 预测框[x, y, w, h]
    :return: 预测框[x1, y1, x2, y2]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # 左上角坐标
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x_left
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y_left
    # 右下角坐标
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x_right
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y_right

    return y


def box_iou(box1, box2):
    """
    计算两个批次的边界框之间的交并比（Intersection over Union, IoU）
    :param box1:预测框1[x1, y1, x2, y2]
    :param box2:预测框2[x1, y1, x2, y2]
    :return: IoU
    """

    def box_area(box):  # 计算box预测框的面积
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    # 计算两个预测框的面积
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # 计算两个批次边界框之间的交集面积
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def plot_one_box(x,i, img,t1,t2, color=None, label=None, line_thickness=3,save_path='/result'):
    str_FPS = "FPS: %.2f" % (1. / (t2 - t1)) # 计算FPS
    cv2.putText(img, str_FPS, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255 , 0), 3)
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA) # 绘制矩形框
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if save_path:
        cv2.imwrite(os.path.join(save_path, f'{i}.jpg'), img)

    return img
