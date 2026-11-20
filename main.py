"""
Module: main.py
Chương trình chính (CLI Entrypoint) của Hệ thống Phân loại Email Spam Tiếng Anh & Tiếng Việt.

Hướng dẫn sử dụng cho sinh viên:
  1. Huấn luyện và đánh giá cả 2 mô hình (Mặc định):
     python main.py

  2. Huấn luyện và đánh giá riêng mô hình Tiếng Anh:
     python main.py --lang en

  3. Huấn luyện và đánh giá riêng mô hình Tiếng Việt:
     python main.py --lang vi

  4. Dự đoán nhanh một văn bản email/tin nhắn mới:
     python main.py --predict "Chúc mừng quý khách đã trúng thưởng 50 triệu!" --predict_lang vi
     python main.py --predict "Urgent: Your account is suspended, click here to verify" --predict_lang en
"""

import os
import sys
import time
import argparse
from src.data_loader import load_english_data, load_vietnamese_data, split_dataset, save_splits
from src.classifier import (
    train,
    val,
    validation,
    test,
    load_model,
    predict_single
)

# Đường dẫn mặc định lưu trữ 2 mô hình
MODEL_PATHS = {
    'en': os.path.join('models', 'model_en.joblib'),
    'vi': os.path.join('models', 'model_vi.joblib')
}


def run_pipeline(language: str) -> None:
    """
    Thực thi toàn bộ luồng huấn luyện, kiểm định và kiểm thử cho một ngôn ngữ:
      Bước 1: Nạp và chuẩn hóa dữ liệu
      Bước 2: Phân chia 70% Train, 15% Validation, 15% Test
      Bước 3: Khởi tạo và huấn luyện Pipeline TF-IDF + Multinomial Naive Bayes
      Bước 4: Đánh giá trên tập Validation
      Bước 5: Đánh giá toàn diện trên tập Test (xuất đầy đủ Confusion Matrix)
      Bước 6: Lưu mô hình hoàn chỉnh vào thư mục models/
    """
    lang = language.lower()
    lang_name = "TIẾNG ANH (ENGLISH)" if lang == 'en' else "TIẾNG VIỆT (VIETNAMESE)"
    
    print("\n" + "=" * 64)
    print(f"   QUY TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH: {lang_name}")
    print("=" * 64)

    # 1. Nạp dữ liệu
    print(f"[Bước 1/5] Đang nạp và làm sạch dữ liệu {lang_name}...")
    start_time = time.time()
    if lang == 'en':
        df = load_english_data()
    else:
        df = load_vietnamese_data()
    load_time = time.time() - start_time

    total_samples = len(df)
    spam_count = int(df['label'].sum())
    ham_count = total_samples - spam_count
    spam_pct = (spam_count / total_samples) * 100
    ham_pct = (ham_count / total_samples) * 100

    print(f"  -> Tổng số mẫu hợp lệ: {total_samples:,} mẫu ({load_time:.2f} giây)")
    print(f"  -> Phân bố nhãn: Spam = {spam_count:,} ({spam_pct:.1f}%) | Ham = {ham_count:,} ({ham_pct:.1f}%)")

    # 2. Phân chia tập dữ liệu 70/15/15 và lưu ra các file train.csv, val.csv, test.csv
    print("\n[Bước 2/5] Phân chia dữ liệu theo tỷ lệ chuẩn 70% Train - 15% Val - 15% Test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, random_state=42)
    split_dir = os.path.join('data', lang)
    split_paths = save_splits(X_train, X_val, X_test, y_train, y_val, y_test, split_dir)
    print(f"  -> Tập Huấn luyện (Train) : {len(X_train):>6,} mẫu ({len(X_train)/total_samples*100:.1f}%) | Spam: {y_train.sum():>5,} | File: {split_paths['train']}")
    print(f"  -> Tập Kiểm định  (Val)   : {len(X_val):>6,} mẫu ({len(X_val)/total_samples*100:.1f}%) | Spam: {y_val.sum():>5,} | File: {split_paths['val']}")
    print(f"  -> Tập Kiểm thử   (Test)  : {len(X_test):>6,} mẫu ({len(X_test)/total_samples*100:.1f}%) | Spam: {y_test.sum():>5,} | File: {split_paths['test']}")

    # 3. Huấn luyện mô hình
    save_path = MODEL_PATHS[lang]
    print(f"\n[Bước 3/5] Đang huấn luyện mô hình TF-IDF + Multinomial Naive Bayes...")
    train_start = time.time()
    model = train(X_train, y_train, language=lang, model_save_path=save_path)
    train_duration = time.time() - train_start
    print(f"  -> Huấn luyện và lưu mô hình hoàn tất trong {train_duration:.2f} giây!")
    print(f"  -> Vị trí file mô hình: {save_path}")

    # 4. Kiểm định trên tập Validation
    print("\n[Bước 4/5] Đánh giá trên tập Kiểm định (Validation Set)...")
    val_metrics = val(model, X_val, y_val)
    print(f"  -> Validation Accuracy : {val_metrics['accuracy'] * 100:.2f}%")
    print(f"  -> Validation Precision: {val_metrics['precision'] * 100:.2f}%")
    print(f"  -> Validation Recall   : {val_metrics['recall'] * 100:.2f}%")
    print(f"  -> Validation F1-Score : {val_metrics['f1'] * 100:.2f}%")

    # 5. Kiểm thử toàn diện trên tập Test
    print("\n[Bước 5/5] Đánh giá toàn diện trên tập Kiểm thử Độc lập (Test Set)...")
    test_metrics = test(model, X_test, y_test, display=True)

    return model, val_metrics, test_metrics


def handle_prediction(text: str, language: str) -> None:
    """
    Xử lý dự đoán cho một văn bản email/tin nhắn mới nhập từ người dùng.
    """
    model_path = MODEL_PATHS.get(language)
    if not model_path or not os.path.exists(model_path):
        print(f"\n[LỖI] Chưa tìm thấy file mô hình đã huấn luyện tại: {model_path}")
        print(f"Vui lòng chạy lệnh huấn luyện trước:")
        print(f"   python main.py --lang {language}\n")
        sys.exit(1)

    print(f"\n[INFO] Đang nạp mô hình {language.upper()} từ: {model_path}...")
    model = load_model(model_path)
    result = predict_single(model, text)

    print("\n" + "=" * 56)
    print("                 KẾT QUẢ DỰ ĐOÁN")
    print("=" * 56)
    print(f"  - Nội dung text   : \"{result['text']}\"")
    print(f"  - Dự đoán         : {result['label'].upper()} (Nhãn nhị phân: {result['prediction']})")
    print(f"  - Độ tin cậy      : {result['confidence'] * 100:.2f}%")
    print("  - Xác suất chi tiết:")
    print(f"      + Ham  (Email thường): {result['probabilities']['Ham'] * 100:.2f}%")
    print(f"      + Spam (Thư rác)     : {result['probabilities']['Spam'] * 100:.2f}%")
    print("=" * 56 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Hệ thống phân loại email spam tối giản tiếng Anh & tiếng Việt cho sinh viên."
    )
    parser.add_argument(
        '--lang',
        choices=['en', 'vi', 'all'],
        default='all',
        help="Chọn mô hình cần huấn luyện và đánh giá: 'en' (Tiếng Anh), 'vi' (Tiếng Việt), hoặc 'all' (Cả hai)."
    )
    parser.add_argument(
        '--predict',
        type=str,
        default=None,
        help="Chuỗi văn bản email/tin nhắn cần kiểm tra phân loại."
    )
    parser.add_argument(
        '--predict_lang',
        choices=['en', 'vi'],
        default='vi',
        help="Ngôn ngữ của văn bản cần dự đoán ('en' hoặc 'vi', mặc định là 'vi')."
    )

    args = parser.parse_args()

    # Chế độ 1: Dự đoán văn bản đơn lẻ
    if args.predict:
        handle_prediction(args.predict, args.predict_lang)
        return

    # Chế độ 2: Huấn luyện và đánh giá mô hình
    if args.lang in ['en', 'all']:
        run_pipeline('en')
    if args.lang in ['vi', 'all']:
        run_pipeline('vi')

    print("\n[HOÀN TẤT] Tất cả các tiến trình huấn luyện và kiểm thử đã thực thi thành công!")


if __name__ == '__main__':
    main()
