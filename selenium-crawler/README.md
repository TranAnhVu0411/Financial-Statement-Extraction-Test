# Crawl dữ liệu báo cáo tài chính

Dữ liệu báo cáo tài chính được lấy trên trang https://data.kreston.vn/tra-cuu-bao-cao-viet-nam/ sử dụng selenium để lấy dữ liệu

Cài đặt selenium: pip install selenium (Môi trường python 3.9)

Tải chromedriver.exe để chạy chương trình

Quá trình lấy dữ liệu sẽ đi qua 2 bước:
- **Bước 1**: Chọn filter như hình dưới:\
![filter](selenium-crawler/image/filter.png)\
và lấy các đường dẫn các trang báo cáo tài chính chi tiết (Vào terminal, nhập lệnh cd selenium-crawler và chạy extract-report-link.py):\
![list](selenium-crawler/image/list.png)\
Dữ liệu được lưu trữ trong metadata/link_data.json
- **Bước 2**: Duyệt từng đường dẫn trang báo cáo tài chính chi tiết, thu thập thông tin như trong hình dưới (Chạy crawl-report.py)\
![detail](selenium-crawler/image/detail.png)\
Dữ liệu được lưu trữ thông tin trong metadata/crawl_data{}.py