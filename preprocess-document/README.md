# Tiền xử lý dữ liệu ảnh

Phần này tập trung vào tiền xử lý dữ liệu ảnh báo cáo tài chính, cụ thể bao gồm các bước:
- Deskew ảnh đầu vào sử dụng Wand
- Tăng contrast cho ảnh sử dụng Pillow
- Loại bỏ vùng chứa dấu sử dụng OpenCV
- Loại bỏ chữ ký (Tham khảo https://github.com/EnzoSeason/signature_detection)

Hướng dẫn sử dụng:
- Cài thư viện Wand, matplotlib (pip install Wand)
- Lưu ảnh muốn xử lý vào folder image dưới dạng test(index).jpg (Ví dụ: test1.jpg, test20.jpg)
- Vào terminal, nhập lệnh cd preprocess-document
- Chỉnh sửa img_idx theo index trong file preprocess.py, chạy file để thu được kết quả, kết quả được lưu trong result-preprocess

Mô hình kiểm nghiệm kết quả: CRAFT - Text detection model

Đánh giá kết quả thông qua số lượng từ mà mô hình CRAFT bắt được (Mô hình bắt được càng ít nhiễu càng tốt)

Cài đặt CRAFT - Text detection: 
- clone https://github.com/clovaai/CRAFT-pytorch và tải pretrained model và lưu vào folder model (Đã có sẵn) (⚠️ Lưu ý chỉnh sửa code theo trang: https://github.com/clovaai/CRAFT-pytorch/issues/191), đổi tên CRAFT-pytorch thành CRAFT_pytorch (Phục vụ cho table-extraction-old)
- Cài các thư viện theo requirements.txt (pip install lần lượt, không được chạy pip install -r requirements.txt)

Chạy CRAFT - Text detection
- Chạy cd preprocess-document
- Chạy script: python CRAFT_pytorch/test.py --trained_model=(Path to train model) --test_folder=result-preprocess --cuda=False (Ví dụ: python CRAFT_pytorch/test.py --trained_model=/Users/trananhvu/Documents/GitHub/Financial-Statement-Extraction-Test/model/craft_mlt_25k.pth --test_folder=result-preprocess --cuda=False)
- Kết quả ở trong folder result

⚠️ Loại bỏ chữ ký đang có một số nhược điểm như:
- Detect không hiệu quả
- Có thể sẽ loại bỏ vùng text chứa trong box được detect
Sẽ kiểm nghiệm yolov5 model cho sign detection (https://medium.com/red-buffer/signature-detection-and-localization-using-yolov5-algorithm-7176ed19fc8b) và Signver cho sign cleaning (https://github.com/victordibia/signver) (Hiện đang ở trong phần sign-detection-and remove)