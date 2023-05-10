# Financial-Statement-Extraction-Test

Github thu thập dữ liệu và thử nghiệm Trích xuất dữ liệu báo cáo tài chính (Bản Test), gồm các phần:

### Thu thập dữ liệu
- selenium-crawler: Thu thập và lưu trữ các link báo cáo tài chính từ trang https://data.kreston.vn/tra-cuu-bao-cao-viet-nam/
- pdf-image-extraction: Tải PDF từ các link PDF báo cáo tài chính được thu thập từ selenium-crawler và chuyển đổi các trang PDF về dạng ảnh

### Tiền xử lý báo cáo tài chính
- preprocess-document: Tiền xử lý ảnh báo cáo tài chính nhằm tăng độ chính xác cho OCR

### Trích rút dữ liệu bảng
- table-extraction-old: (⚠️ Lưu ý: bản cũ) Chuyển đổi dữ liệu ảnh bảng (Đã được crop sẵn) về dạng CSV