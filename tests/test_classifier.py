"""
Module: tests/test_classifier.py
Kiểm thử tự động cho module xây dựng Pipeline, huấn luyện (train), 
kiểm định (val/validation), kiểm thử (test), lưu/nạp và dự đoán.
"""

import os
import tempfile
import numpy as np
import pytest
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


@pytest.fixture
def sample_en_dataset():
    """Tập dữ liệu mẫu nhỏ tiếng Anh phục vụ unit test nhanh."""
    texts = [
        "Congratulations! You won a $1,000 Walmart gift card, click here now",
        "Hi Professor, could you please review the attached draft?",
        "URGENT: Your bank account will be suspended unless you verify credentials",
        "Team, the weekly status meeting is rescheduled to Thursday 2 PM",
        "Claim your free bonus spins on our online casino platform today",
        "Attached is the revised project specification document for review"
    ] * 5
    labels = [1, 0, 1, 0, 1, 0] * 5
    return texts, labels


@pytest.fixture
def sample_vi_dataset():
    """Tập dữ liệu mẫu nhỏ tiếng Việt phục vụ unit test nhanh."""
    texts = [
        "Chúc mừng quý khách đã trúng thưởng 50 triệu đồng từ chương trình tri ân",
        "Chào bạn, ngày mai nhóm mình có lịch họp báo cáo lúc 8 giờ sáng nhé",
        "Cảnh báo: Tài khoản ngân hàng Vietcombank bị khóa, truy cập link để mở",
        "Thầy gửi cho cả lớp tài liệu tham khảo môn học máy qua email",
        "Cơ hội việc làm online tại nhà thu nhập 20 triệu một tháng không cần cọc",
        "Tối nay đi ăn lẩu liên hoan cùng phòng không Tuấn ơi"
    ] * 5
    labels = [1, 0, 1, 0, 1, 0] * 5
    return texts, labels


def test_build_pipeline_english():
    """Kiểm tra cấu trúc và siêu tham số Pipeline tiếng Anh."""
    pipeline = build_pipeline(language='en')
    assert isinstance(pipeline, Pipeline)
    assert 'tfidf' in pipeline.named_steps
    assert 'nb' in pipeline.named_steps

    tfidf = pipeline.named_steps['tfidf']
    nb = pipeline.named_steps['nb']

    assert isinstance(tfidf, TfidfVectorizer)
    assert tfidf.max_features == 10000
    assert tfidf.stop_words == 'english'

    assert isinstance(nb, MultinomialNB)
    assert nb.alpha == 1.0


def test_build_pipeline_vietnamese():
    """Kiểm tra cấu trúc và siêu tham số Pipeline tiếng Việt."""
    pipeline = build_pipeline(language='vi')
    assert isinstance(pipeline, Pipeline)
    assert 'tfidf' in pipeline.named_steps
    assert 'nb' in pipeline.named_steps

    tfidf = pipeline.named_steps['tfidf']
    nb = pipeline.named_steps['nb']

    assert isinstance(tfidf, TfidfVectorizer)
    assert tfidf.max_features == 10000
    assert tfidf.ngram_range == (1, 2)
    assert tfidf.token_pattern == r'(?u)\b\w+\b'
    assert tfidf.sublinear_tf is True

    assert isinstance(nb, MultinomialNB)
    assert nb.alpha == 0.1


def test_build_pipeline_invalid_language():
    """Kiểm tra ngoại lệ ValueError khi chọn ngôn ngữ không hỗ trợ."""
    with pytest.raises(ValueError):
        build_pipeline(language='french')


def test_train_val_and_validation_alias(sample_en_dataset):
    """Kiểm tra hàm train, val và bí danh validation."""
    texts, labels = sample_en_dataset
    model = train(texts[:20], labels[:20], language='en')

    # Kiểm tra val()
    val_res = val(model, texts[20:], labels[20:])
    assert isinstance(val_res, dict)
    for key in ['accuracy', 'precision', 'recall', 'f1']:
        assert key in val_res
        assert 0.0 <= val_res[key] <= 1.0

    # Kiểm tra bí danh validation()
    validation_res = validation(model, texts[20:], labels[20:])
    assert val_res == validation_res, "Hàm validation() phải trả về kết quả giống hệt val()."


def test_test_function_metrics_and_confusion_matrix(sample_vi_dataset):
    """Kiểm tra hàm test xuất đầy đủ Accuracy, Precision, Recall, F1 và Confusion Matrix."""
    texts, labels = sample_vi_dataset
    model = train(texts[:20], labels[:20], language='vi')

    metrics = evaluate_model(model, texts[20:], labels[20:], display=False)

    # 1. Kiểm tra các chỉ số bắt buộc
    for key in ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix', 'classification_report']:
        assert key in metrics, f"Thiếu khóa '{key}' trong kết quả kiểm thử."

    # 2. Kiểm tra Confusion Matrix
    cm = metrics['confusion_matrix']
    assert isinstance(cm, np.ndarray), "Confusion matrix phải là numpy.ndarray."
    assert cm.shape == (2, 2), f"Kích thước ma trận nhầm lẫn phải là 2x2, thực tế: {cm.shape}"
    assert np.issubdtype(cm.dtype, np.integer), "Các phần tử trong ma trận nhầm lẫn phải là số nguyên."

    # 3. Kiểm tra Classification Report
    assert isinstance(metrics['classification_report'], str)
    assert len(metrics['classification_report']) > 0


def test_model_persistence(sample_en_dataset):
    """Kiểm tra lưu mô hình (save_model) và nạp lại mô hình (load_model)."""
    texts, labels = sample_en_dataset

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_model_path = os.path.join(tmp_dir, "test_pipeline.joblib")

        # Huấn luyện và tự động lưu
        model = train(texts, labels, language='en', model_save_path=tmp_model_path)
        assert os.path.exists(tmp_model_path), "File mô hình phải được lưu tại đường dẫn chỉ định."

        # Nạp lại mô hình
        loaded_model = load_model(tmp_model_path)
        assert isinstance(loaded_model, Pipeline)

        # Kiểm tra tính nhất quán giữa mô hình gốc và mô hình nạp lại
        test_samples = ["Free bonus prize click here", "Hey, see you in class tomorrow"]
        orig_preds = model.predict(test_samples)
        loaded_preds = loaded_model.predict(test_samples)
        np.testing.assert_array_equal(orig_preds, loaded_preds)


def test_load_model_file_not_found():
    """Kiểm tra nạp file mô hình không tồn tại sẽ raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model("nonexistent_path/model_never_exists.joblib")


def test_predict_and_predict_single(sample_vi_dataset):
    """Kiểm tra hàm predict() và predict_single()."""
    texts, labels = sample_vi_dataset
    model = train(texts, labels, language='vi')

    # 1. Kiểm tra predict() với danh sách
    preds = predict(model, ["Trúng thưởng tiền mặt lớn", "Họp nhóm tối nay"])
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert all(p in [0, 1] for p in preds)

    # 2. Kiểm tra predict_single()
    single_res = predict_single(model, "Trúng thưởng 100 triệu bấm vào link sau")
    assert isinstance(single_res, dict)
    assert 'text' in single_res
    assert 'prediction' in single_res and single_res['prediction'] in [0, 1]
    assert 'label' in single_res and single_res['label'] in ['Ham', 'Spam']
    assert 'confidence' in single_res and 0.0 <= single_res['confidence'] <= 1.0
    assert 'probabilities' in single_res
    assert 'Ham' in single_res['probabilities']
    assert 'Spam' in single_res['probabilities']

    total_prob = single_res['probabilities']['Ham'] + single_res['probabilities']['Spam']
    assert abs(total_prob - 1.0) < 1e-5, f"Tổng xác suất phải bằng 1.0, thực tế: {total_prob}"
