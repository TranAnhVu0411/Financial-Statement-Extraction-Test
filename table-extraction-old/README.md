# Trích xuất dữ liệu bảng từ ảnh

## ⚠️ Lưu ý: Đây là bản demo cũ, độ chính xác không cao, sẽ có bản cập nhật mới trong tương lai

Phần này tập trung vào chuyển đổi ảnh bảng (Ảnh bảng đã crop trước, bảng không có border) sang dạng CSV. Quá trình này sẽ đi qua các bước như sau:
- Tiền xử lý ảnh, Chuyển đổi ảnh sang PDF
- Sử dụng thư viện Camelot để lấy ra thông tin bounding box của các cells trong bảng
- Sử dụng thư viện OpenCV để sắp xếp lại các bounding box theo trật tự có cấu trúc, sử dụng CRAFT model để xác định xem bounding box đó có chứa text hay không, cuối cùng dùng VietOCR để OCR text có trong bounding box đó

Cài đặt thư viện: pytesseract, camelot (camelot-py), vietocr, PyPDF2, yaml (pyyaml)

### Hướng dẫn sử dụng: Chạy thử nghiệm thư viện Camelot thuần
- Vào terminal, nhập lệnh cd table-extraction-old
- Lưu ảnh muốn trích xuất vào trong folder image/preprocess dưới dạng dưới dạng test(index).jpg (Ví dụ: test1.jpg, test20.jpg)
- Chỉnh sửa img_idx theo index trong file tesseract-table-test.py, chạy file để thu được ảnh đã được xử lý được lưu trong folder image/preprocess và file pdf chuyển đổi được lưu trong folder pdf
- Chỉnh sửa img_idx theo index trong file camelot-test.py, chạy file để thu được csv được trích xuất từ thư viện Camelot được lưu trong folder csv/camelot và file metadata chứa bounding box của cell được lưu trong folder metadata

### Hướng dẫn sử dụng: Chạy thử nghiệm theo pipeline đề xuất
- Vào terminal, nhập lệnh cd table-extraction-old
- Chỉnh sửa img_idx và craft_model (absolute path) trong pipeline.py
- Chạy pipeline.py, kết quả được lưu trong folder csv/pipeline