"""
Module: tests/test_classifier.py
Kiểm thử tự động cho module xây dựng Pipeline tiếng Việt, huấn luyện (train), 
kiểm định (val/validation), kiểm thử (test), lưu/nạp và dự đoán.
Tương thích cả pytest và python -m unittest (stdlib).
"""

import os
import unittest
import tempfile
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.classifier import (
    build_pipeline,
    train,
    val,
    validation,
    test as evaluate_model,
    save_model,
    load_model,
    predict,
    predict_single
)


class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.texts = [
            "Chúc mừng quý khách đã trúng thưởng 50 triệu đồng từ chương trình tri ân",
            "Chào bạn, ngày mai nhóm mình có lịch họp báo cáo lúc 8 giờ sáng nhé",
            "Cảnh báo: Tài khoản ngân hàng Vietcombank bị khóa, truy cập link để mở",
            "Thầy gửi cho cả lớp tài liệu tham khảo môn học máy qua email",
            "Cơ hội việc làm online tại nhà thu nhập 20 triệu một tháng không cần cọc",
            "Tối nay đi ăn lẩu liên hoan cùng phòng không Tuấn ơi"
        ] * 5
        self.labels = [1, 0, 1, 0, 1, 0] * 5

    def test_build_pipeline_vietnamese(self):
        """Kiểm tra cấu trúc và siêu tham số Pipeline tiếng Việt."""
        pipeline = build_pipeline(language='vi')
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIn('tfidf', pipeline.named_steps)
        self.assertIn('nb', pipeline.named_steps)

        tfidf = pipeline.named_steps['tfidf']
        nb = pipeline.named_steps['nb']

        self.assertIsInstance(tfidf, TfidfVectorizer)
        self.assertEqual(tfidf.max_features, 10000)
        self.assertEqual(tfidf.ngram_range, (1, 2))
        pattern = r'(?u)\b[a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]{2,}\b'
        self.assertEqual(tfidf.token_pattern, pattern)
        self.assertTrue(tfidf.sublinear_tf)

        self.assertIsInstance(nb, MultinomialNB)
        self.assertEqual(nb.alpha, 0.1)

    def test_train_val_and_validation_alias(self):
        """Kiểm tra hàm train, val và bí danh validation."""
        model = train(self.texts[:20], self.labels[:20], language='vi')

        val_res = val(model, self.texts[20:], self.labels[20:])
        self.assertIsInstance(val_res, dict)
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            self.assertIn(key, val_res)
            self.assertTrue(0.0 <= val_res[key] <= 1.0)

        validation_res = validation(model, self.texts[20:], self.labels[20:])
        self.assertEqual(val_res, validation_res, "validation() phải trả về kết quả giống val().")

    def test_test_function_metrics_and_confusion_matrix(self):
        """Kiểm tra hàm test xuất đầy đủ Accuracy, Precision, Recall, F1 và Confusion Matrix."""
        model = train(self.texts[:20], self.labels[:20], language='vi')
        metrics = evaluate_model(model, self.texts[20:], self.labels[20:], display=False)

        for key in ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix', 'classification_report']:
            self.assertIn(key, metrics, f"Thiếu khóa '{key}' trong kết quả kiểm thử.")

        cm = metrics['confusion_matrix']
        self.assertIsInstance(cm, np.ndarray)
        self.assertEqual(cm.shape, (2, 2))
        self.assertTrue(np.issubdtype(cm.dtype, np.integer))

        self.assertIsInstance(metrics['classification_report'], str)
        self.assertGreater(len(metrics['classification_report']), 0)

    def test_model_persistence(self):
        """Kiểm tra lưu mô hình (save_model) và nạp lại mô hình (load_model)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model_path = os.path.join(tmp_dir, "test_pipeline.joblib")
            model = train(self.texts, self.labels, language='vi', model_save_path=tmp_model_path)
            self.assertTrue(os.path.exists(tmp_model_path))

            loaded_model = load_model(tmp_model_path)
            self.assertIsInstance(loaded_model, Pipeline)

            test_samples = ["Cơ hội việc làm kiếm tiền nhanh", "Hôm nay có bài tập mới không"]
            orig_preds = model.predict(test_samples)
            loaded_preds = loaded_model.predict(test_samples)
            np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_load_model_file_not_found(self):
        """Kiểm tra nạp file mô hình không tồn tại sẽ raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_model("nonexistent_path/model_never_exists.joblib")

    def test_predict_and_predict_single(self):
        """Kiểm tra hàm predict() và predict_single()."""
        model = train(self.texts, self.labels, language='vi')

        preds = predict(model, ["Trúng thưởng tiền mặt lớn", "Họp nhóm tối nay"])
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), 2)
        self.assertTrue(all(p in [0, 1] for p in preds))

        single_res = predict_single(model, "Trúng thưởng 100 triệu bấm vào link sau")
        self.assertIsInstance(single_res, dict)
        self.assertIn('text', single_res)
        self.assertIn(single_res['prediction'], [0, 1])
        self.assertIn(single_res['label'], ['Ham', 'Spam'])
        self.assertTrue(0.0 <= single_res['confidence'] <= 1.0)
        self.assertIn('Ham', single_res['probabilities'])
        self.assertIn('Spam', single_res['probabilities'])

        total_prob = single_res['probabilities']['Ham'] + single_res['probabilities']['Spam']
        self.assertAlmostEqual(total_prob, 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
