from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from ImageProcessor import ImageProcessor
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
        try:
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
        except Exception as e:
            print("子线程崩溃了！错误原因：",e)

    def stop(self):
        self._run_flag = False
        self.wait()

    # 接收UI滑块的值
    def update_thresholds(self, t1, t2):
        self.canny_thresh1 = t1
        self.canny_thresh2 = t2