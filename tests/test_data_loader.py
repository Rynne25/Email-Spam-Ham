"""
Module: tests/test_data_loader.py
Kiểm thử tự động cho module nạp dữ liệu và phân chia tập dữ liệu.
"""

import os
import pytest
import pandas as pd
from src.data_loader import (
    load_english_data,
    load_vietnamese_data,
    split_dataset,
    EN_DEFAULT_PATH,
    VN_LOCAL_PATH
)


def test_load_english_data_success():
    """Kiểm tra nạp dữ liệu tiếng Anh từ file Enron thực tế."""
    if not os.path.exists(EN_DEFAULT_PATH):
        pytest.skip("File dữ liệu tiếng Anh không tồn tại trên máy test.")

    df = load_english_data(EN_DEFAULT_PATH)
    assert isinstance(df, pd.DataFrame), "Kết quả phải là DataFrame của pandas."
    assert list(df.columns) == ['text', 'label'], "DataFrame phải có đúng 2 cột ['text', 'label']."
    assert len(df) > 30000, f"Số lượng dòng tiếng Anh phải > 30,000, thực tế nhận: {len(df)}"
    assert set(df['label'].unique()).issubset({0, 1}), "Nhãn phải là nhị phân {0, 1}."
    assert (df['text'].str.len() > 0).all(), "Không được có mẫu văn bản nào rỗng."


def test_load_english_data_file_not_found():
    """Kiểm tra ngoại lệ FileNotFoundError khi đường dẫn không tồn tại."""
    with pytest.raises(FileNotFoundError):
        load_english_data("invalid/path/to/nonexistent_dataset.csv")


def test_load_vietnamese_data_success():
    """Kiểm tra nạp dữ liệu tiếng Việt (tự động nạp từ cache cục bộ hoặc tải về)."""
    df = load_vietnamese_data()
    assert isinstance(df, pd.DataFrame), "Kết quả phải là DataFrame của pandas."
    assert list(df.columns) == ['text', 'label'], "DataFrame phải có đúng 2 cột ['text', 'label']."
    assert len(df) >= 2000, f"Số lượng dòng tiếng Việt phải >= 2000, thực tế: {len(df)}"
    assert set(df['label'].unique()).issubset({0, 1}), "Nhãn phải là nhị phân {0, 1}."
    assert (df['text'].str.len() > 0).all(), "Không được có mẫu văn bản nào rỗng."


def test_split_dataset_exact_proportions_synthetic():
    """Kiểm tra tỷ lệ phân chia chuẩn 70% Train - 15% Val - 15% Test trên tập mẫu 1000 dòng."""
    sample_df = pd.DataFrame({
        'text': [f"Sample message {i}" for i in range(1000)],
        'label': [1 if i % 4 == 0 else 0 for i in range(1000)]  # 25% spam, 75% ham
    })

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(sample_df, random_state=42)

    total = len(sample_df)
    assert len(X_train) == 700, f"Kỳ vọng 700 mẫu train (70%), thực tế: {len(X_train)}"
    assert len(X_val) == 150, f"Kỳ vọng 150 mẫu val (15%), thực tế: {len(X_val)}"
    assert len(X_test) == 150, f"Kỳ vọng 150 mẫu test (15%), thực tế: {len(X_test)}"
    assert len(y_train) == 700
    assert len(y_val) == 150
    assert len(y_test) == 150


def test_split_dataset_stratification():
    """Kiểm tra kỹ thuật Stratified Split bảo toàn tỷ lệ nhãn trên cả 3 tập."""
    sample_df = pd.DataFrame({
        'text': [f"Email content {i}" for i in range(2000)],
        'label': [1 if i < 600 else 0 for i in range(2000)]  # 30% spam (600/2000)
    })

    original_spam_ratio = sample_df['label'].mean()  # 0.30

    _, _, _, y_train, y_val, y_test = split_dataset(sample_df, random_state=42)

    train_spam_ratio = y_train.mean()
    val_spam_ratio = y_val.mean()
    test_spam_ratio = y_test.mean()

    # Độ sai lệch tỷ lệ spam giữa các tập không được vượt quá 1% (0.01)
    assert abs(train_spam_ratio - original_spam_ratio) < 0.01
    assert abs(val_spam_ratio - original_spam_ratio) < 0.01
    assert abs(test_spam_ratio - original_spam_ratio) < 0.01


def test_split_dataset_reproducibility():
    """Kiểm tra tính tái lập (reproducible) khi cố định random_state."""
    sample_df = pd.DataFrame({
        'text': [f"Text sample {i}" for i in range(500)],
        'label': [i % 2 for i in range(500)]
    })

    split1 = split_dataset(sample_df, random_state=42)
    split2 = split_dataset(sample_df, random_state=42)

    for s1, s2 in zip(split1, split2):
        pd.testing.assert_series_equal(s1, s2)
