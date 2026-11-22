# Hệ Thống Phân Loại Email Spam Tiếng Việt (Vietnamese Spam Email Detector)
> **Nhánh: `vietnamese-demo`** — Phiên bản tinh gọn, thuần Tiếng Việt phục vụ trình chiếu đồ án và kiểm nghiệm trực tiếp.

---

## 1. Giới thiệu Tổng quan

Hệ thống phát hiện và phân loại email/tin nhắn rác (Spam) và thư thường (Ham) tối giản dành riêng cho ngữ cảnh Tiếng Việt:
- **TF-IDF Vectorizer**: Trích xuất đặc trưng cụm từ tiếng Việt với `ngram_range=(1, 2)`, giữ nguyên ký tự có dấu.
- **Multinomial Naive Bayes**: Mô hình phân loại xác suất với hệ số làm mịn tối ưu cho tiếng Việt `alpha=0.1`.
- **Dữ liệu huấn luyện**: Hơn 2,900 mẫu email/tin nhắn thực tế tại Việt Nam (lừa đảo tuyển dụng, mạo danh ngân hàng/shipper, vay tín dụng, v.v.).

---

## 2. Cấu trúc Dự án (Nhánh `vietnamese-demo`)

```text
Email-Spam-Ham/
├── data/
│   ├── vi/                             # 3 tập dữ liệu chuẩn: train.csv, val.csv, test.csv
│   └── vietnamese_spam.csv             # Dữ liệu gốc tiếng Việt đã làm sạch
├── models/
│   └── model_vi.joblib                 # Mô hình Pipeline tiếng Việt đã huấn luyện
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Tải dữ liệu tiếng Việt & Stratified Split 70/15/15
│   └── classifier.py                   # Pipeline TF-IDF + MultinomialNB thuần Việt
├── tests/
│   ├── test_data_loader.py             # Unit test cho data loader tiếng Việt
│   └── test_classifier.py              # Unit test cho pipeline và dự đoán
├── app.py                              # Giao diện Web demo tương tác (Streamlit)
├── main.py                             # Giao diện dòng lệnh demo (CLI)
├── requirements.txt
└── README.md
```

---

## 3. Hướng dẫn Khởi chạy Demo

### Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

### Cách 1: Khởi chạy Giao diện Web Streamlit (Khuyến nghị cho thuyết trình / báo cáo)
```bash
streamlit run app.py
```
- Giao diện có sẵn 8 mẫu email thực tế tại Việt Nam (Trúng thưởng, Ngân hàng, Tuyển dụng, Vay tiền, Giả shipper, Họp nhóm, Học tập, Công việc).
- Hiển thị xác suất Spam/Ham, thẻ kết luận và phân tích các từ khóa nghi vấn trực quan.
- Bảng KPI và Ma trận nhầm lẫn (Confusion Matrix) trên tập kiểm thử độc lập.

### Cách 2: Khởi chạy Demo dòng lệnh (CLI)
```bash
# Mặc định: Chạy đánh giá Test set và mở prompt nhập tương tác:
python main.py

# Hoặc dự đoán nhanh 1 câu:
python main.py --predict "Chúc mừng bạn trúng thưởng 50 triệu bấm vào link nhận quà ngay"
```

---

## 4. Kiểm thử Tự động (Testing)

Chạy bộ kiểm thử tự động với pytest:
```bash
pytest -v
```
Toàn bộ test cases đều được tối ưu hóa cho dữ liệu và quy trình tiếng Việt.
