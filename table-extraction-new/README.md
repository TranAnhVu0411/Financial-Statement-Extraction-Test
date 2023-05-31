# Trích xuất dữ liệu bảng từ ảnh (Bản mới)

Phần này tập trung vào chuyển đổi ảnh bảng (Ảnh bảng đã crop trước, bảng không có border) sang dạng CSV. Quá trình này sẽ đi qua các bước như sau:
- Tiền xử lý ảnh (Deskew ảnh)
- Sử dụng mô hình YOLOv5 đã huấn luyện để lấy được các thành phần cấu trúc của bảng
- Sử dụng kết quả của YOLOv5 kết hợp với text detection để làm tinh chỉnh các cell trong bảng
- Từ thông tin các cell trong bảng tiến hành OCR sử dụng VietOCR

Cài đặt YOLOv5 (pretrained weight hiện chưa public), tham khảo trong README.md của signature-detection-and-remove
Cài đặt CRAFT và pretrained weight (lưu pretrained weight vào folder model), tham khảo trong README.md của preprocess-document
Cài đặt thêm thư viện PyMuPDF và html2excel

### Hướng dẫn sử dụng
- Vào terminal, nhập lệnh cd table-extraction-new
- Thay đổi đường dẫn file ảnh trong tableYOLO.py và chạy file, kết quả tiền xử lý nghiêng, loại bỏ đường, file html và excel tương ứng sẽ được lưu trong folder result