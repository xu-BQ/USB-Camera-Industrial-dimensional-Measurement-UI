
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap


# ==========================================
# 1. 算法处理层 (Image Processor)
# 职责：只负责纯图像处理，不关心界面长什么样
# ==========================================
class ImageProcessor:
    def __init__(self):
        # 工程硬编码参数：假设通过标定，1个像素代表 0.1 mm
        # (实际应用中，你需要根据相机离桌面的高度，用一张A4纸去标定这个比例)
        self.pixels_per_mm = 1.72

    def process(self, frame, canny_thresh1, canny_thresh2):
        # 1. 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. 光线暗的救星：限制对比度自适应直方图均衡化 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        # 3. 双边滤波 (Bilateral Filter) - 工业上保边去噪的神器
        # 相比高斯滤波，它能在去噪的同时，更好地保留物体的锐利边缘
        blurred = cv2.bilateralFilter(gray_enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        # 4. Canny
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

        # （可选）如果边缘还是断裂，可以加一个闭运算
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        '''
        # 1. 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2. 高斯滤波去噪 (工业上常用 5x5 或 7x7 核)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 3. Canny边缘检测 (阈值由UI滑块动态传入)
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

        '''
        # 4. 寻找轮廓
        # RETR_EXTERNAL 只找最外层轮廓，CHAIN_APPROX_SIMPLE 压缩轮廓点以节省内存
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_frame = frame.copy()
        measure_text = "未检测到目标"

        if contours:
            # 5. 过滤噪点：找面积最大的轮廓（假设硬币是画面中最大的物体）
            c = max(contours, key=cv2.contourArea)

            # 过滤太小的噪点（比如面积小于500像素的）
            if cv2.contourArea(c) > 500:
                # 6. 计算最小外接圆
                ((x, y), radius) = cv2.minEnclosingCircle(c)

                # 7. 核心：像素尺寸转物理尺寸
                pixel_diameter = radius * 2
                physical_diameter = pixel_diameter / self.pixels_per_mm

                # 8. 绘制UI (画圆和圆心)
                cv2.circle(result_frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(result_frame, (int(x), int(y)), 2, (0, 0, 255), 3)

                # 更新显示文本
                measure_text = f"直径: {physical_diameter:.2f} mm"

        return result_frame, edges, measure_text


# ==========================================
# 2. 多线程层 (Camera Thread)
# 职责：死循环读摄像头、跑算法，并通过信号发送给UI
# ==========================================
class CameraThread(QThread):
    # 定义信号：传递给主线程 (QImage原图, QImage边缘图, 测量文本)
    change_pixmap_signal = pyqtSignal(QImage, QImage, str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.processor = ImageProcessor()
        # 默认Canny阈值
        self.canny_thresh1 = 50
        self.canny_thresh2 = 150

    def run(self):
        cap = cv2.VideoCapture(0)  # 0是默认电脑摄像头

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # 1. 调用算法层处理图像
                result_frame, edges_frame, text = self.processor.process(
                    frame, self.canny_thresh1, self.canny_thresh2
                )

                # 2. OpenCV的格式是BGR，PyQt需要RGB
                rgb_image = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                edges_rgb = cv2.cvtColor(edges_frame, cv2.COLOR_GRAY2RGB)

                # 3. 转换为QImage (注意：必须调用.copy()，否则会因为内存释放导致花屏)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

                h_e, w_e, ch_e = edges_rgb.shape
                q_edges = QImage(edges_rgb.data, w_e, h_e, ch_e * w_e, QImage.Format_RGB888).copy()

                # 4. 发射信号给UI
                self.change_pixmap_signal.emit(q_img, q_edges, text)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    # 接收UI滑块的值
    def update_thresholds(self, t1, t2):
        self.canny_thresh1 = t1
        self.canny_thresh2 = t2


# ==========================================
# 3. UI 层 (主窗口)
# 职责：只负责布局、显示、接收用户输入
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工业级尺寸测量系统 V1.0")
        self.resize(1000, 600)

        # 初始化UI控件
        self.init_ui()

        # 初始化并启动相机线程
        self.thread = CameraThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        # 连接滑块信号槽
        self.slider_t1.valueChanged.connect(self.slider_changed)
        self.slider_t2.valueChanged.connect(self.slider_changed)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左侧：视频显示 ---
        left_layout = QVBoxLayout()
        self.image_label = QLabel("正在启动摄像头...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: white; border: 2px solid gray;")
        self.image_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.image_label)

        # 将边缘检测图也显示出来，方便调试
        self.edge_label = QLabel("边缘检测视图")
        self.edge_label.setAlignment(Qt.AlignCenter)
        self.edge_label.setStyleSheet("background-color: black; color: white;")
        self.edge_label.setMinimumSize(640, 200)
        left_layout.addWidget(self.edge_label)

        # --- 右侧：控制面板 ---
        right_layout = QVBoxLayout()

        # 1. 测量结果面板
        group_result = QGroupBox("测量结果")
        result_layout = QVBoxLayout()
        self.result_label = QLabel("等待测量...")
        self.result_label.setStyleSheet("font-size: 24px; color: red; font-weight: bold;")
        result_layout.addWidget(self.result_label)
        group_result.setLayout(result_layout)

        # 2. 参数调节面板
        group_ctrl = QGroupBox("参数调节 (Canny阈值)")
        ctrl_layout = QVBoxLayout()

        self.slider_t1 = QSlider(Qt.Horizontal)
        self.slider_t1.setRange(0, 255)
        self.slider_t1.setValue(50)
        self.slider_t2 = QSlider(Qt.Horizontal)
        self.slider_t2.setRange(0, 255)
        self.slider_t2.setValue(150)

        ctrl_layout.addWidget(QLabel("低阈值 (Thresh1):"))
        ctrl_layout.addWidget(self.slider_t1)
        ctrl_layout.addWidget(QLabel("高阈值 (Thresh2):"))
        ctrl_layout.addWidget(self.slider_t2)
        group_ctrl.setLayout(ctrl_layout)

        right_layout.addWidget(group_result)
        right_layout.addWidget(group_ctrl)
        right_layout.addStretch()  # 把控件顶上去

        # --- 组合布局 ---
        main_layout.addLayout(left_layout, stretch=3)  # 左侧占比大
        main_layout.addLayout(right_layout, stretch=1)  # 右侧占比小

    def update_image(self, q_img, q_edges, text):
        """接收子线程发来的图像并更新UI"""
        self.image_label.setPixmap(QPixmap.fromImage(q_img))
        self.edge_label.setPixmap(QPixmap.fromImage(q_edges))
        self.result_label.setText(text)

    def slider_changed(self):
        """滑块变动时，把新参数传给子线程"""
        self.thread.update_thresholds(self.slider_t1.value(), self.slider_t2.value())

    def closeEvent(self, event):
        """窗口关闭时，安全清理线程"""
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())