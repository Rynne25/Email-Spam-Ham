"""
Module: tests/test_data_loader.py
Kiểm thử tự động cho module nạp dữ liệu tiếng Việt và phân chia tập dữ liệu.
Tương thích cả pytest và python -m unittest (stdlib).
"""

import unittest
import pandas as pd
from src.data_loader import (
    load_vietnamese_data,
    split_dataset,
    VN_LOCAL_PATH
)


class TestDataLoader(unittest.TestCase):

    def test_load_vietnamese_data_success(self):
        """Kiểm tra nạp dữ liệu tiếng Việt (tự động nạp từ cache cục bộ hoặc tải về)."""
        df = load_vietnamese_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ['text', 'label'])
        self.assertGreaterEqual(len(df), 2000)
        self.assertTrue(set(df['label'].unique()).issubset({0, 1}))
        self.assertTrue((df['text'].str.len() > 0).all())

    def test_split_dataset_exact_proportions_synthetic(self):
        """Kiểm tra tỷ lệ phân chia chuẩn 70% Train - 15% Val - 15% Test trên tập mẫu 1000 dòng."""
        sample_df = pd.DataFrame({
            'text': [f"Sample message {i}" for i in range(1000)],
            'label': [1 if i % 4 == 0 else 0 for i in range(1000)]
        })

        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(sample_df, random_state=42)

        self.assertEqual(len(X_train), 700)
        self.assertEqual(len(X_val), 150)
        self.assertEqual(len(X_test), 150)
        self.assertEqual(len(y_train), 700)
        self.assertEqual(len(y_val), 150)
        self.assertEqual(len(y_test), 150)

    def test_split_dataset_stratification(self):
        """Kiểm tra kỹ thuật Stratified Split bảo toàn tỷ lệ nhãn trên cả 3 tập."""
        sample_df = pd.DataFrame({
            'text': [f"Email content {i}" for i in range(2000)],
            'label': [1 if i < 600 else 0 for i in range(2000)]
        })

        original_spam_ratio = sample_df['label'].mean()

        _, _, _, y_train, y_val, y_test = split_dataset(sample_df, random_state=42)

        train_spam_ratio = y_train.mean()
        val_spam_ratio = y_val.mean()
        test_spam_ratio = y_test.mean()

        self.assertLess(abs(train_spam_ratio - original_spam_ratio), 0.01)
        self.assertLess(abs(val_spam_ratio - original_spam_ratio), 0.01)
        self.assertLess(abs(test_spam_ratio - original_spam_ratio), 0.01)

    def test_split_dataset_reproducibility(self):
        """Kiểm tra tính tái lập (reproducible) khi cố định random_state."""
        sample_df = pd.DataFrame({
            'text': [f"Text sample {i}" for i in range(500)],
            'label': [i % 2 for i in range(500)]
        })

        split1 = split_dataset(sample_df, random_state=42)
        split2 = split_dataset(sample_df, random_state=42)

        for s1, s2 in zip(split1, split2):
            pd.testing.assert_series_equal(s1, s2)


if __name__ == '__main__':
    unittest.main()
