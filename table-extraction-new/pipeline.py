import cv2
import torch

print('TABLE STRUCTURE DETECTION')
str_model = torch.hub.load(
    'yolov5',
    'custom',
    source='local',
    path = 'model/best.pt',
    force_reload = True)

image = cv2.imread('/Users/trananhvu/Documents/GitHub/CTCP Thiết Kế Xây Lắp Viễn Đông-page-7.jpg')
pred = str_model(image, size=640)


