# Tiền xử lý dữ liệu ảnh

Phần này tập trung vào tiền xử lý dữ liệu ảnh báo cáo tài chính, cụ thể bao gồm các bước:
- Deskew ảnh đầu vào
- Tăng contrast cho ảnh
- Loại bỏ vùng chứa dấu
- Loại bỏ chữ ký (Tham khảo https://github.com/EnzoSeason/signature_detection)

Hướng dẫn sử dụng:
- Cài thư viện Wand, matplotlib (pip install Wand)
- Lưu ảnh muốn xử lý vào folder image dưới dạng test<index>.jpg (Ví dụ: test1.jpg, test20.jpg)
- Chỉnh sửa img_idx theo index trong file preprocess.py, chạy file để thu được kết quả

Mô hình kiểm nghiệm kết quả: CRAFT - Text detection model

Đánh giá kết quả thông qua số lượng từ mà mô hình CRAFT bắt được (Mô hình bắt được càng ít nhiễu càng tốt)

Cài đặt CRAFT - Text detection: 
- clone https://github.com/clovaai/CRAFT-pytorch và tải pretrained model và lưu vào folder model (Đã có sẵn) (⚠️ Lưu ý chỉnh sửa code theo trang: https://github.com/clovaai/CRAFT-pytorch/issues/191)
- Cài các thư viện theo requirements.txt (pip install lần lượt, không được chạy pip install -r requirements.txt)

Chạy CRAFT - Text detection
- Chạy script: python CRAFT-pytorch/test.py --trained_model=model/craft_mlt_25k.pth --test_folder=preprocess-document/result --cuda=False
- Kết quả ở trong folder result

Sẽ kiểm nghiệm sign detection sử dụng Signver sau (https://github.com/victordibia/signver)