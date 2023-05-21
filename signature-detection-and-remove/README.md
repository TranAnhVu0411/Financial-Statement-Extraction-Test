# Nâng cấp sign detection và cleaning (preprocess document)

Phần này tập trung vào việc nâng cấp khả năng detect chữ ký và xử lý loại bỏ chữ ký

⚠️ Chạy trong folder signature-detection-and-remove (Trong terminal nhập cd signature-detection-and-remove)

Cài đặt:
- YOLOv5:
    - Tải model từ trang https://drive.google.com/drive/folders/1397Q9nqMqEsSesB9UvlIL1u1IGzrUOIK (best.pth) và lưu vào folder model
    - git clone https://github.com/ultralytics/yolov5
    - cd yolov5
    - pip install -r requirements.txt

- Signver
    - git clone https://github.com/victordibia/signver.git
    - cd signver
    - pip install -r requirements.txt

Hướng dẫn sử dụng:
- Chạy độc lập
    - Chạy signature detection: chạy file signature-detection.py, đầu vào mặc định là các ảnh có trong folder preprocess-document/result-preprocess có 'removestamp' trong tên, đầu ra là ảnh chữ ký được crop được lưu trong folder signature-crop
    - Chạy signature cleanding: chạy file signature-cleaning.py (Trước khi chạy đảm bảo trong folder signature-crop có dữ liệu ảnh đầu ra của phần chạy signature detection), đầu ra là ảnh chữ ký được clean được lưu trong folder signature-clean

- Chạy luồng xử lý loại bỏ signature
    - Trong file pipeline.py, sửa đổi img_name thành tên ảnh trong folder preprocess-document/result-preprocess (Tốt nhất sử dụng những ảnh nào có removestamp ở trong tên)
    - Chạy file pipeline.py, đầu ra là ảnh đã được loại bỏ chữ ký được lưu trong folder result/preprocess, ngoài ra còn lưu thêm cả mask của ảnh chứa chữ ký được lưu trong folder result/mask