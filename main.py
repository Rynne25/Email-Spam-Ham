"""
Module: main.py
Chương trình dòng lệnh (CLI Demo) tối giản của Hệ thống Phân loại Email Spam Tiếng Việt.

Cách dùng:
  1. Chạy demo tương tác (Tự động nạp/đánh giá mô hình & nhập thử văn bản trực tiếp):
     python main.py

  2. Dự đoán nhanh một câu văn bản:
     python main.py --predict "Chúc mừng quý khách đã trúng thưởng 50 triệu!"

  3. Huấn luyện lại mô hình từ đầu:
     python main.py --train
"""

import os
import sys
import argparse
from src.data_loader import load_vietnamese_data, split_dataset, save_splits, load_splits
from src.classifier import (
    train,
    test,
    load_model,
    predict_single,
    explain_prediction
)

MODEL_PATH = os.path.join('models', 'model_vi.joblib')
SPLIT_DIR = os.path.join('data', 'vi')


def ensure_data_and_model():
    """Đảm bảo dữ liệu và mô hình tiếng Việt đã sẵn sàng."""
    splits_exist = all(os.path.exists(os.path.join(SPLIT_DIR, f"{s}.csv")) for s in ['train', 'val', 'test'])
    if not splits_exist:
        print("[1/3] Đang chuẩn bị tập dữ liệu tiếng Việt (70% Train - 15% Val - 15% Test)...")
        df = load_vietnamese_data()
        splits = split_dataset(df, random_state=42)
        save_splits(*splits, output_dir=SPLIT_DIR)

    if not os.path.exists(MODEL_PATH):
        print(f"[2/3] Đang huấn luyện mô hình tiếng Việt và lưu tại {MODEL_PATH}...")
        X_train, _, _, y_train, _, _ = load_splits(SPLIT_DIR)
        train(X_train, y_train, language='vi', model_save_path=MODEL_PATH)

    return load_model(MODEL_PATH)


def display_prediction(model, text: str) -> None:
    """In kết quả dự đoán kèm độ tin cậy và từ khóa giải thích."""
    res = predict_single(model, text)
    is_spam = res['prediction'] == 1
    tag = "[🚨 SPAM / THƯ RÁC]" if is_spam else "[✅ EMAIL AN TOÀN / HAM]"

    print("\n" + "─" * 56)
    print(f"  Kết luận     : {tag}")
    print(f"  Độ tin cậy   : {res['confidence'] * 100:.2f}%")
    print(f"  Xác suất     : Spam: {res['probabilities']['Spam'] * 100:.2f}% | Ham: {res['probabilities']['Ham'] * 100:.2f}%")

    keywords = explain_prediction(model, text, top_k=5)
    if keywords:
        kw_str = ", ".join([f"{k['word']} ({k['score']:+.2f})" for k in keywords])
        print(f"  Từ khóa chính: {kw_str}")
    print("─" * 56 + "\n")


def interactive_loop(model) -> None:
    """Vòng lặp tương tác cho phép người dùng gõ văn bản test trực tiếp."""
    print("Mời bạn nhập nội dung email/tin nhắn cần kiểm tra (hoặc gõ 'q' để thoát):")
    while True:
        try:
            user_input = input("✉️  Nội dung: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Tạm biệt!")
                break
            display_prediction(model, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Demo tối giản Hệ thống Phân loại Email Spam Tiếng Việt (TF-IDF + Naive Bayes)."
    )
    parser.add_argument(
        '--predict',
        type=str,
        default=None,
        help="Nội dung văn bản email cần phân loại trực tiếp."
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help="Ép buộc huấn luyện lại mô hình tiếng Việt từ đầu."
    )

    args = parser.parse_args()

    # Chế độ huấn luyện lại
    if args.train:
        print("[INFO] Đang huấn luyện lại mô hình tiếng Việt...")
        df = load_vietnamese_data()
        splits = split_dataset(df, random_state=42)
        save_splits(*splits, output_dir=SPLIT_DIR)
        X_train, _, _, y_train, _, _ = splits
        train(X_train, y_train, language='vi', model_save_path=MODEL_PATH)
        print(f"[HOÀN TẤT] Mô hình đã được cập nhật tại: {MODEL_PATH}")
        return

    # Tải / Chuẩn bị mô hình
    model = ensure_data_and_model()

    # Nếu truyền cờ --predict
    if args.predict:
        display_prediction(model, args.predict)
        return

    # Chế độ demo mặc định: Kiểm thử nhanh trên tập test + Mở tương tác
    print("=" * 60)
    print("   HỆ THỐNG PHÂN LOẠI EMAIL SPAM TIẾNG VIỆT (DEMO)")
    print("   Thuật toán: TF-IDF n-gram (1, 2) + Multinomial Naive Bayes")
    print("=" * 60)

    _, _, X_test, _, _, y_test = load_splits(SPLIT_DIR)
    print("\n[ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP]")
    test(model, X_test, y_test, display=True)

    interactive_loop(model)


if __name__ == '__main__':
    main()
