import os
import cv2
import sys
import time
import numpy as np
import onnx
import onnxruntime

import torch
import torchvision

from PyQt5 import QtCore, QtGui, QtWidgets
import random

import utils


class Model:  # 前向推理模型
    def __init__(self):
        self.image_size = 640  # 默认图片大小为640
        self.device = 'cuda:0'  # 默认训练设备为cuda:0
        # 默认训练文件为Yolov5s.onnx
        self.onnx_model_path = 'models/yolov5.onnx'
        self.onnx_model = onnx.load_model(self.onnx_model_path)
        self.session = onnxruntime.InferenceSession(self.onnx_model_path)
        # Get names and colors
        self.names = ['Apple','Banana','Grape','Orange','Pineapple','Watermelon']
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
        self.drawn_image=None

    def inference(self, img_BGR):
        """
        对输入的图片进行图片预处理、前向推理、非极大值抑制NMS、输出
        :param img_BGR: 图片，BGR类型
        :return: 无返回值
        """
        t1 = time.time()

        img0 = img_BGR.copy()

        # 图像预处理
        # print(img0.shape)
        img = self.image_process(img_BGR)
        # print(img.shape)
        # 前向推理-将预处理后的图像数据输入模型进行推理，得到预测结果pred
        inputs = {'images': utils.to_numpy(img)}
        # print(inputs)
        pred = self.session.run(None, inputs)
        # print(pred)
        # 非极大值抑制NMS，对预测结果 pred 应用非极大值抑制，过滤掉重叠较多的边界框
        filterd_pred = self.non_max_suppression(torch.tensor(pred[0]), conf_thres=0.25, iou_thres=0.45)
        filterd_pred[0][:, :4] = utils.scale_coords(img.shape[2:], filterd_pred[0][:, :4],
                                                          img0.shape).round()
        # print(filterd_pred)
        t2 = time.time()
        # Process detections
        for i, det in enumerate(filterd_pred):
            if det is not None and len(det):
                # det[:, :4] = utils.scale_coords(img.shape[2:], filterd_pred[0][:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    print(label)
                    self.drawn_image = utils.plot_one_box(xyxy, i, img0, t1,t2,label=label, color=self.colors[int(cls)],
                                                          line_thickness=2)

        return self.drawn_image

    def image_process(self, img_BGR):
        """
        对输入的图片进行处理，包括缩放、调整为RBG模型、归一化操作。
        :param img_BGR:
        :return: img
        """
        img = utils.letterbox(img_BGR, self.image_size, stride=32, auto=False, scaleFill=False, scaleup=True)[
            0]  # 自适应图片缩放

        img = img[:, :, ::-1].transpose(2, 0, 1)  # 颠倒通道顺序(BGR to RGB)，然后调换维度通道、高度和宽度
        img = np.ascontiguousarray(img)  # 将image放到一个连续内存的Numpy数组
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # fp32
        img /= 255.0  # 像素标准化到0.0-1.0
        if img.ndimension() == 3:  # 如果图片为三维，则需要添加一个维度-批处理大小
            img = img.unsqueeze(0)

        return img

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False, labels=()):
        """
        Runs Non-Maximum Suppression (NMS) on inference results
        :param prediction: [batch, num_boxes(3个yolo预测层), (x+y+w+h+1+num_classes)] = [1, 18900, 25]  3个anchor的预测结果总和
        :param conf_thres: 先进行一轮筛选，将分数过低的预测框（<conf_thres）删除（分数置0）
        :param iou_thres: iou阈值, 如果其余预测框与target的iou>iou_thres, 就将那个预测框置0
        :param classes: classes 通常是一个包含目标类别名称或索引的列表
        :param agnostic: 进行nms是否也去除不同类别之间的框 默认False
        :param multi_label: 是否是多标签  nc>1  一般是True
        :param labels:
        :return:output,经过NMS之后的结果
        """
        nc = prediction.shape[2] - 5  # number of classes类别数,[batch, num_anchors(3个yolo预测层), (x+y+w+h+1+num_classes)]
        xc = prediction[
                 ..., 4] > conf_thres  # 从prediction数组中选择所有可能的前维度组合，并从最后一个维度中选择索引为4的元素，即置信度分数，通过与conf_thres比较得到一个布尔数组xc

        min_wh, max_wh = 2, 4096  # 设置了预测物体的宽度和高度的最小值 min_wh 和最大值 max_wh。这些值用于排除那些太小或太大的检测框，因为这些框可能是不准确的
        max_det = 300  # 每张图像上最大检测数量 max_det
        max_nms = 30000  # NMS（非极大值抑制）之前的最大检测框数量
        time_limit = 10.0  # seconds to quit after, nms执行时间阈值 超过这个时间就退出了
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index,image inference
            x = x[xc[xi]]  # 通过布尔索引从x中选择出那些置信度较高的检测框，剔除小于置信度阈值的grid cell，更新x为只包含这些检测框的信息

            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]
                v[:, 4] = 1.0
                v[range(len(l)), l[:, 0].long() + 5] = 1.0
                x = torch.cat((x, v), 0)
            # If none remain process next image
            if not x.shape[0]:  # 如果长度为0，意味着没有检测框剩下，则跳过当前图像，直接处理下一张
                continue

            # 置信度计算-置信度，最终计算结果表示模型对于检测框内特定类别对象的置信度
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf，x[:, 5:]选择所有检测框的类别置信度分数，x[:, 4:5]选择所有检测框的对象置信度分数。

            # 将检测框的描述方式进行转换,Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = utils.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            # 在多标签情况下，一个检测框可以同时属于多个类别；而在单标签情况下，每个检测框只属于一个最可能的类别。
            if multi_label:  # 多标签模式
                """
                首先检查模型输出的类别置信度（x[:, 5:]）是否大于设定的置信度阈值conf_thres。
                nonzero函数返回所有满足条件的索引，as_tuple=False表示返回的是一个二维数组而不是元组，.T是转置操作，
                得到两个数组i和j，分别代表检测框的索引和满足置信度阈值的类别索引。
                """
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)  # 包含了检测框的坐标、类别概率和类别索引
            else:  # best class only，单标签模式
                """
                使用max函数找到每个检测框对于所有类别的最大置信度conf和对应的类别索引j。
                1表示沿着类别维度查找最大值，keepdim = True保持输出的维度不变。
                """
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # 检查形状
            n = x.shape[0]  # 检测框的个数
            if not n:  # 当前图片没有检测框则直接跳过
                continue
            elif n > max_nms:  # 检测框的数量n超过了预设的最大值max_nms。需要剔除一部分，以免浪费计算资源
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 将根据检测框的置信度分数进行排序，并选择前 max_nms 个检测框

            # 处理多个图像批次的非极大值抑制算法
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            # x[:, :4] 选择每个检测框的坐标（中心点 x, y 和宽高 w, h），然后加上类别偏移量 c。
            # x[:, 4] 选择每个检测框的置信度得分。
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS-移除重叠的检测框，只保留最佳的一些检测框
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            # 合并NMS（Merge NMS）的技术
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = utils.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]  # 将处理后的检测结果 x[i] 存储到输出列表 output 中的相应位置 xi
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output


# UI界面
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        # 初始化
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()  # 定时器，用于在指定的时间间隔触发特定的操作，单位是毫秒
        self.setupUi(self)  # UI初始化，调用setupUi
        self.init_logo()  # logo初始化
        self.init_slots()  # 槽函数初始化
        # video = "http://192.168.184.219:4747/vedio"
        self.cap = cv2.VideoCapture(0)  # 默认摄像头,后面也可以改用手机摄像头
        self.out = None
        self.model = Model() # 模型

    def setupUi(self, MainWindow):
        # 主窗口
        MainWindow.setObjectName("轻量化目标检测")  # 设置主窗口名
        MainWindow.resize(1100, 1000)  # 主窗口大小
        # 布局器设置，创建了一个QWidget对象作为主窗口的中央部件，并使用水平布局和垂直布局进行布局管理
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        # 设置Label框-用来显示图片、视频
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)  # 将Label添加到垂直布局器中(Label与按钮组是垂直布局)
        # 创建水平布局器，用来安排按钮组
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        # 模型选择按钮
        self.pushButton_model = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_model.setMinimumSize(QtCore.QSize(0, 100))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.pushButton_model.setFont(font)
        self.pushButton_model.setObjectName("pushButton_model")
        self.horizontalLayout.addWidget(self.pushButton_model)
        # 图片输入按钮
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_img.setMinimumSize(QtCore.QSize(80, 100))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.pushButton_img.setFont(font)
        self.pushButton_img.setObjectName("pushButton_img")
        self.horizontalLayout.addWidget(self.pushButton_img)
        # 视频输入按钮
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_video.setMinimumSize(QtCore.QSize(0, 100))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.horizontalLayout.addWidget(self.pushButton_video)
        # 摄像头输入按钮
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_camera.setEnabled(True)
        self.pushButton_camera.setMinimumSize(QtCore.QSize(0, 100))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.pushButton_camera.setFont(font)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.horizontalLayout.addWidget(self.pushButton_camera)
        # Label和按钮组为垂直布局
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.setStretch(0, 3)
        self.verticalLayout.setStretch(1, 1)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 28))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)  # 更新界面上的文本内容
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "轻量化目标检测"))
        self.label.setText(_translate("MainWindow", "目标检测结果"))
        self.pushButton_model.setText(_translate("MainWindow", "模型选择"))
        self.pushButton_img.setText(_translate("MainWindow", "图片输入"))
        self.pushButton_video.setText(_translate("MainWindow", "视频输入"))
        self.pushButton_camera.setText(_translate("MainWindow", "摄像头输入"))

    def init_logo(self): # 加载logo图片
        img_path = './resources/main.jpg'
        pix = QtGui.QPixmap(img_path)  # 加载图片
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    def init_slots(self):  # 槽函数初始化，绑定按钮触发事件到对应的函数
        self.pushButton_model.clicked.connect(self.model_open)
        self.pushButton_img.clicked.connect(self.image_open)
        self.pushButton_video.clicked.connect(self.video_open)
        self.pushButton_camera.clicked.connect(self.camera_open)
        self.timer_video.timeout.connect(self.show_video_frame)

    def model_open(self):  # 选择不同的模型导入
        print("button_model_open")
        model_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择模型","models/", "*.onnx"
        )
        if not model_name:
            return
        # 更换模型
        self.model.onnx_model_path = model_name
        self.model.onnx_model = onnx.load_model(self.model.onnx_model_path)
        self.model.session = onnxruntime.InferenceSession(self.model.onnx_model_path)
        # 根据模型类别，更改输入图片尺寸
        print(model_name)

    def image_open(self):  # 图片输入
        print("button_image_open")
        name_list = []

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "dataset/","*.jpg;;*.png;;All Files(*)"
        )
        if not img_name:
            return

        img = cv2.imread(img_name)  # 读取图片
        print(img_name)
        # 模型推理阶段
        with torch.no_grad():
            drawn_image = self.model.inference(img)

        cv2.imwrite('predict_result.jpg', drawn_image)
        # 将OpenCV的图像矩阵转换为Qt可以处理的 QImage 格式
        self.result = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(
            self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                  QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def video_open(self):  # 视频输入
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频输入文件", "","*.mp4;;*.avi;;All Files(*)")

        if not video_name:
            return

        flag = self.cap.open(video_name)
        if not flag:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.out = cv2.VideoWriter('predict_result.avi', cv2.VideoWriter_fourcc(
                *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer_video.start()
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setDisabled(True)
            self.pushButton_model.setDisabled(True)

    def camera_open(self):  # 摄像头输入
        if not self.timer_video.isActive():  # 如果 QTimer 对象正在运行（已经启动并且尚未停止），则该触发器再次发生时关闭软件
            # 默认使用本地摄像头
            flag = self.cap.open(0)
            if not flag:  # 打开摄像头失败
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('predict_result.avi', cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))  # 通过将图像帧写入视频文件，你可以将一系列图像帧保存为视频文件
                self.timer_video.start()
                self.pushButton_video.setDisabled(True)  # 设置图片、视频输入为禁止
                self.pushButton_img.setDisabled(True)
                self.pushButton_model.setDisabled(True)
                self.pushButton_camera.setText(u"关闭摄像头")
        else:  # 下一个触发事件发生(点击关闭摄像头按钮)，释放摄像头，并且设置按钮为可点击
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.init_logo()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_model.setDisabled(False)
            self.pushButton_camera.setText(u"摄像头输入")

    def show_video_frame(self):
        flag, img = self.cap.read()  # 连续读取视频流中的帧图像，并对每一帧进行处理或展示
        if img is not None:
            showimg = img
            with torch.no_grad():
                # 图像预处理
                drawn_image = self.model.inference(img)
            self.out.write(drawn_image)
            # show = cv2.resize(drawn_image, (640, 480))
            self.result = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.pushButton_model.setDisabled(False)
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setDisabled(False)
            self.init_logo()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 创建了一个Qt应用程序
    ui = Ui_MainWindow()  # 使用由用户界面设计器生成的类定义了应用程序的界面布局和组件
    ui.show()
    sys.exit(app.exec_())
