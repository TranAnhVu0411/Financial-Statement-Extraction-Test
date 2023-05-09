# Tải file PDF và chuyển đổi PDF thành danh sách ảnh trang

## I, Tải file PDF
Chạy file pdf_downloader.ipynb để tiến hành tải dữ liệu PDF ta vừa mới crawl được từ selenium

⚠️ Lưu ý:
- Nên chạy pdf_downloader.ipynb trên Colab vì dữ liệu PDF rất nặng
- Do tài nguyên Drive có hạn, nên chỉ lấy khoảng 500-800 file PDF trong 1 tài khoản Drive (tối đa khoảng 1000 PDF)
## II, Chuyển đổi PDF thành danh sách ảnh trang
- Cài thư viện pdf2image và poppler
- Download PDF từ drive về máy và lưu trong folder pdf-image-extraction/pdf
- Tạo folder pdf-image-extraction/image
- Vào terminal, nhập lệnh cd pdf-image-extraction 
- Chạy file image-extraction.py để tiến hành chuyển PDF thành danh sách ảnh, dữ liệu được lưu trữ theo dạng {tên công ty}/{số trang}.jpg

⚠️ Lưu ý:
- Cân nhắc số lượng PDF sử dụng để chuyển thành ảnh (100 PDF chuyển thành ảnh sử dụng khoảng 1GB)