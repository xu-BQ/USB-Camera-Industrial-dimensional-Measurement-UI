from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

from CameraThread import CameraThread
from ImageProcessor import ImageProcessor
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