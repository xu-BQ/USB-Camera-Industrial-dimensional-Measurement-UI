import cv2
import numpy as np
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