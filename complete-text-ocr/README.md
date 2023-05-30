# Pipeline hoàn chỉnh cho trích xuất dữ liệu text

Phần này là tổng hợp từ các phần preprocess-document, signature-detection-and-remove và thêm phần OCR (sử dụng vietOCR, tham khảo trong phần table-extraction-old), Pipeline sẽ gồm các bước như sau:
- Tiền xử lý dữ liệu ảnh (Loại bỏ con dấu, loại bỏ chữ ký)
- Lấy vùng text trong ảnh (Sử dụng CRAFT để detect text và masking vùng text, sử dụng masking đó để loại bỏ hết background của ảnh, chỉ giữ lại vùng text)
- Xác định các contours các đoạn sẽ extract
- Từ các đoạn đã xác định từ bước trên, sử dụng histogram ảnh để lấy ra các dòng (Tham khảo: https://www.kaggle.com/code/irinaabdullaeva/text-segmentation)
- Từ các dòng, tiến hành OCR bằng vietOCR, cuối cùng lưu trữ kết quả dưới dạng JSON

Hướng dẫn chạy code
- Trước khi chạy:
    - Cài CRAFT, YOLO, Signver và tải pretrained model (Tham khảo trong preprocess-document và signature-detection-and-remove)
- Chạy visualize:
    - Chọn 1 ảnh trong signature-detection-and-remove/result/preprocess, thay đổi img_idx trong create_word_metadata.py, cd complete-text-ocr và chạy file đó, thu được file npy chứa bounding box của các text trong ảnh được phát hiện bởi CRAFT
    - Chạy notebook Visualize.ipynb (Lưu ý nhớ sửa lại các đường dẫn file vì hiện tại nó đang hơi sai)
- Chạy pipeline text ocr:
    - Lưu ảnh đầu vào trong folder image dưới dạng test{index}.jpg
    - cd complete-text-ocr
    - Thay đổi img_idx trong preprocess.py và chạy file đó, kết quả đầu ra là ảnh đã được xử lý sẽ được lưu vào trong result/preprocess
    - Thay đổi img_idx trong ocr.py và chạy file đó, kết quả đầu ra bao gồm
        - Ảnh vùng text và ảnh các dòng tương ứng được lưu lần lượt trong folder region và lines
        - Thông tin metadata ocr được lưu vào trong folder result/ocr