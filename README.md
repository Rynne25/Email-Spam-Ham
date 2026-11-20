# Hệ Thống Phân Loại Email Spam (Spam Email Classification System)
## Dành cho Tiếng Anh (English) và Tiếng Việt (Vietnamese)

Dự án Machine Learning tối giản, rõ ràng, được thiết kế chuyên biệt cho sinh viên đại học học tập các môn **Nhập môn Học máy (Machine Learning)** và **Khai phá dữ liệu (Data Mining)**.

---

## 1. Giới thiệu Tổng quan

Hệ thống cung cấp giải pháp phát hiện và phân loại email/tin nhắn rác (Spam) và thư thường (Ham) dựa trên phương pháp Học máy cổ điển kết hợp:
- **TF-IDF (Term Frequency - Inverse Document Frequency)**: Biểu diễn văn bản dưới dạng các vector đặc trưng số học.
- **Multinomial Naive Bayes (Đa thức Naive Bayes)**: Thuật toán phân loại xác suất dựa trên định lý Bayes, đặc biệt hiệu quả và nhanh chóng cho dữ liệu văn bản.

Hệ thống hỗ trợ **2 mô hình độc lập**:
1. **Mô hình Tiếng Anh (`model_en.joblib`)**: Huấn luyện trên tập dữ liệu Enron Spam với hơn 33,000 email.
2. **Mô hình Tiếng Việt (`model_vi.joblib`)**: Tự động tải từ Hugging Face Hub, huấn luyện trên tập dữ liệu tin nhắn SMS/phishing thực tế tại Việt Nam.

---

## 2. Kiến trúc Hệ thống & Luồng Xử lý

Quy trình xử lý dữ liệu tuân thủ nghiêm ngặt nguyên lý Machine Learning chuẩn:

```text
[Dữ liệu Thô CSV]
       │
       ▼
[Tiền Xử Lý & Làm Sạch] ──► Loại bỏ mẫu rỗng, chuẩn hóa nhãn (0: Ham, 1: Spam)
       │
       ▼
[Phân Chia Dữ Liệu]     ──► Stratified Split: 70% Train - 15% Validation - 15% Test
       │
       ▼
[Scikit-learn Pipeline]
   ├── Bước 1: TF-IDF Vectorizer (Trích xuất đặc trưng từ vựng & N-gram)
   └── Bước 2: Multinomial Naive Bayes (Huấn luyện phân loại xác suất)
       │
       ▼
[Đánh Giá & Lưu Trữ]    ──► Accuracy, Precision, Recall, F1, Confusion Matrix ──► .joblib
```

### Điểm nổi bật về kiến trúc dành cho sinh viên:
- **Thiết kế hàm độc lập (Functional Modular Architecture)**: Phân tách rõ ràng giữa `train()`, `val()`, `validation()`, và `test()`.
- **Không gây rò rỉ dữ liệu (No Data Leakage)**: Sử dụng `Pipeline` của `scikit-learn` giúp gom bộ trích xuất đặc trưng và thuật toán học máy vào một khối duy nhất.
- **Tối ưu tiếng Việt không cần cài đặt thư viện ngoài**: Sử dụng `ngram_range=(1, 2)` và `token_pattern=r'(?u)\b\w+\b'` để bắt trọn cụm từ ghép tiếng Việt (ví dụ: *khuyến mãi, trúng thưởng, tài khoản, xác thực*), loại bỏ hoàn toàn rủi ro lỗi biên dịch C++ của các gói NLP phức tạp.

---

## 3. Cấu trúc Thư mục Dự án

```text
d:\Phân loại email spam\
├── data/                               # Thư mục lưu trữ dữ liệu
│   └── vietnamese_sms.csv              # Dữ liệu tiếng Việt (tự động tải và cache tại đây)
├── models/                             # Thư mục lưu trữ mô hình sau khi huấn luyện
│   ├── model_en.joblib                 # Mô hình Pipeline tiếng Anh
│   └── model_vi.joblib                 # Mô hình Pipeline tiếng Việt
├── src/                                # Mã nguồn chính
│   ├── __init__.py
│   ├── data_loader.py                  # Nạp dữ liệu & phân chia 70/15/15 stratified
│   └── classifier.py                   # Pipeline, train, val, validation, test, save/load, predict
├── tests/                              # Bộ kiểm thử tự động với pytest
│   ├── __init__.py
│   ├── test_data_loader.py             # Unit tests cho nạp và phân chia dữ liệu
│   └── test_classifier.py              # Unit tests cho pipeline, metrics và persistence
├── main.py                             # Giao diện dòng lệnh chính (CLI Entrypoint)
├── requirements.txt                    # Danh sách thư viện phụ thuộc
└── README.md                           # Tài liệu hướng dẫn sử dụng
```

---

## 4. Hướng dẫn Cài đặt & Chuẩn bị Môi trường

### Yêu cầu hệ thống:
- Python 3.10 trở lên (tương thích hoàn hảo với cả Python 3.14).
- Không yêu cầu card đồ họa GPU rời.

### Cài đặt các thư viện cần thiết:
Mở terminal hoặc PowerShell tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

Nội dung file `requirements.txt`:
- `scikit-learn>=1.4.0`: Thư viện học máy cốt lõi (TF-IDF, Naive Bayes, metrics, pipeline).
- `pandas>=2.0.0`: Thư viện xử lý bảng dữ liệu.
- `numpy>=1.24.0`: Thư viện tính toán ma trận.
- `joblib>=1.3.0`: Thư viện lưu trữ và nạp mô hình.
- `pytest>=8.0.0`: Khung kiểm thử tự động.

---

## 5. Hướng dẫn Sử dụng (CLI)

### 5.1 Huấn luyện và Đánh giá Toàn bộ (Cả Tiếng Anh & Tiếng Việt)
Lệnh mặc định sẽ thực hiện nạp dữ liệu, phân chia 70/15/15, huấn luyện và đánh giá chi tiết trên tập test:
```bash
python main.py
```

### 5.2 Huấn luyện Riêng Từng Ngôn ngữ
- **Chỉ chạy mô hình Tiếng Anh**:
  ```bash
  python main.py --lang en
  ```
- **Chỉ chạy mô hình Tiếng Việt**:
  ```bash
  python main.py --lang vi
  ```

### 5.3 Thử nghiệm Dự đoán Email / Tin nhắn Mới
Bạn có thể nhập trực tiếp chuỗi văn bản bất kỳ để kiểm tra phân loại:
- **Dự đoán Tiếng Việt**:
  ```bash
  python main.py --predict "Chúc mừng bạn đã trúng thưởng 50 triệu đồng liên hệ nhận quà ngay" --predict_lang vi
  ```
  *Kết quả in ra*:
  ```text
  ========================================================
                   KẾT QUẢ DỰ ĐOÁN
  ========================================================
    - Nội dung text   : "Chúc mừng bạn đã trúng thưởng 50 triệu..."
    - Dự đoán         : SPAM (Nhãn nhị phân: 1)
    - Độ tin cậy      : 92.50%
    - Xác suất chi tiết:
        + Ham  (Email thường): 7.50%
        + Spam (Thư rác)     : 92.50%
  ========================================================
  ```

- **Dự đoán Tiếng Anh**:
  ```bash
  python main.py --predict "Urgent: Your bank account will be suspended unless you verify credentials now" --predict_lang en
  ```

---

## 6. Giải thích Các Chỉ Số Đánh Giá (Evaluation Metrics)

Trong bài toán phân loại email spam nhị phân, quy ước:
- **Lớp 0 (Âm tính - Negative)**: Email thường (Ham).
- **Lớp 1 (Dương tính - Positive)**: Email rác/lừa đảo (Spam).

### 6.1 Ma trận Nhầm lẫn (Confusion Matrix)
Ma trận vuông kích thước $2 \times 2$:
$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix}$$
- **TN (True Negative)**: Email thường và mô hình đoán đúng là thường.
- **FP (False Positive)**: Email thường nhưng mô hình đoán nhầm là spam (*báo động giả - rất nguy hiểm vì người dùng bị lỡ email quan trọng*).
- **FN (False Negative)**: Email spam nhưng mô hình bỏ sót và đoán là thường (*lọt lưới rác*).
- **TP (True Positive)**: Email spam và mô hình bắt đúng là spam.

### 6.2 Các Chỉ số Đo lường Chính
1. **Accuracy (Độ chính xác tổng quan)**:
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
   Tỷ lệ dự đoán chính xác trên toàn bộ tập dữ liệu kiểm thử.

2. **Precision (Độ chuẩn xác của Spam)**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   Trong số tất cả các email bị mô hình gán nhãn là Spam, có bao nhiêu phần trăm thực sự là Spam. Precision càng cao thì tỷ lệ chặn oan email thường càng thấp.

3. **Recall (Độ thu hồi / Độ nhạy)**:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   Trong số tất cả các email spam thực tế, mô hình đã bắt trúng được bao nhiêu phần trăm.

4. **F1-Score (Trung bình điều hòa giữa Precision và Recall)**:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   Chỉ số cân bằng đại diện khi tập dữ liệu bị mất cân bằng nhãn.

---

## 7. Kết quả Thực nghiệm Thực tế

Được đo lường độc lập trên tập **Test độc lập (15%)** không tham gia vào quá trình huấn luyện:

| Chỉ số Đánh giá | Mô hình Tiếng Anh (Enron Spam) | Mô hình Tiếng Việt (SMS Phishing) |
| :--- | :---: | :---: |
| **Quy mô toàn bộ tập** | 33,665 mẫu (50.9% Spam / 49.1% Ham) | 2,991 mẫu (26.7% Spam / 73.3% Ham) |
| **Tập Train (70%)** | 23,565 mẫu | 2,093 mẫu |
| **Tập Validation (15%)** | 5,050 mẫu | 449 mẫu |
| **Tập Test (15%)** | 5,050 mẫu | 449 mẫu |
| **Accuracy (Tổng)** | **98.46%** | **93.76%** |
| **Precision (Spam)** | **98.11%** | **88.33%** |
| **Recall (Spam)** | **98.87%** | **88.33%** |
| **F1-Score (Spam)** | **98.49%** | **88.33%** |
| **Confusion Matrix** | $\begin{pmatrix} 2433 & 49 \\ 29 & 2539 \end{pmatrix}$ | $\begin{pmatrix} 315 & 14 \\ 14 & 106 \end{pmatrix}$ |
| **Thời gian huấn luyện** | ~4.0 giây | ~0.35 giây |

---

## 8. Hướng dẫn Chạy Kiểm thử Tự động (Testing)

Dự án đi kèm bộ unit test và integration test hoàn chỉnh, tuân thủ tiêu chuẩn kỹ nghệ phần mềm.

Chạy kiểm thử với lệnh:
```bash
pytest -v
```

Kết quả kỳ vọng: **14/14 tests PASSED (100%)** bao gồm:
- Kiểm tra tính hợp lệ và cấu trúc schema 2 cột `['text', 'label']`.
- Kiểm tra tỷ lệ phân chia chính xác 70% Train, 15% Val, 15% Test.
- Kiểm tra bảo toàn tỷ lệ nhãn qua Stratified Split.
- Kiểm tra sự độc lập và tính nhất quán của `train()`, `val()`, `validation()`, và `test()`.
- Kiểm tra tính đầy đủ của 5 chỉ số đánh giá và kích thước $2 \times 2$ của Ma trận nhầm lẫn.
- Kiểm tra độ bền lưu trữ và nạp mô hình (`save_model`, `load_model`) qua `joblib`.
- Kiểm tra hàm dự đoán đơn lẻ `predict_single()` với tổng xác suất chuẩn hóa $100\%$.
