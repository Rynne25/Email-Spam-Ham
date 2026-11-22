"""
Module: src/classifier.py
Chức năng:
  - Khởi tạo scikit-learn Pipeline (TfidfVectorizer + MultinomialNB) tối ưu cho tiếng Việt.
  - Các hàm độc lập: train(), val(), validation(), test().
  - Lưu và nạp mô hình qua joblib: save_model(), load_model().
  - Dự đoán email mới: predict(), predict_single().
Dành cho: Sinh viên học tập môn Nhập môn Học máy & Khai phá dữ liệu.
"""

import os
from typing import Any, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def build_pipeline(language: str = 'vi') -> Pipeline:
    """
    Khởi tạo scikit-learn Pipeline kết hợp trích xuất đặc trưng TF-IDF và thuật toán Multinomial Naive Bayes.
    Tối ưu hóa chuyên biệt cho văn bản tiếng Việt:
      + TfidfVectorizer: ngram_range=(1, 2), token_pattern bắt từ tiếng Việt >= 2 ký tự (loại bỏ số và ký tự đơn),
        stop_words loại bỏ các artifact kỹ thuật như 'lt', 'gt', 'amp'.
      + MultinomialNB: alpha=0.1, fit_prior=False (cân bằng xác suất tiên nghiệm tránh thiên vị class imbalance).
    """
    pattern = r'(?u)\b[a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]{2,}\b'
    stop_words = ['lt', 'gt', 'amp', 'fwiw', 'lor', 'pa', 'ill', 'da', 'ah']

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        token_pattern=pattern,
        stop_words=stop_words,
        sublinear_tf=True,
        lowercase=True
    )
    classifier = MultinomialNB(alpha=0.1, fit_prior=False)

    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('nb', classifier)
    ])
    return pipeline


def train(
    X_train: Union[pd.Series, list[str]],
    y_train: Union[pd.Series, list[int]],
    language: str = 'vi',
    model_save_path: str = None
) -> Pipeline:
    """
    HÀM HUẤN LUYỆN (train):
    Khởi tạo pipeline, khớp (fit) TF-IDF và huấn luyện Naive Bayes trên tập dữ liệu train.
    Tự động lưu mô hình ra đĩa nếu được truyền đường dẫn model_save_path.
    
    Args:
        X_train: Tập văn bản email/tin nhắn huấn luyện.
        y_train: Tập nhãn phân loại (0: Ham, 1: Spam).
        language: Ngôn ngữ (Mặc định 'vi').
        model_save_path: Đường dẫn file .joblib để lưu mô hình (tùy chọn).
        
    Returns:
        Pipeline: Mô hình đã được huấn luyện hoàn chỉnh.
    """
    model = build_pipeline(language=language)
    model.fit(X_train, y_train)

    if model_save_path:
        save_model(model, model_save_path)

    return model


def val(
    model: Pipeline,
    X_val: Union[pd.Series, list[str]],
    y_val: Union[pd.Series, list[int]]
) -> dict[str, float]:
    """
    HÀM KIỂM ĐỊNH (val):
    Đánh giá mô hình trên tập validation để kiểm tra độ khái quát và phát hiện overfitting.
    
    Args:
        model (Pipeline): Mô hình đã huấn luyện.
        X_val: Tập văn bản kiểm định.
        y_val: Nhãn thực tế kiểm định.
        
    Returns:
        dict: Chứa các chỉ số {'accuracy', 'precision', 'recall', 'f1'} kiểu float.
    """
    y_pred = model.predict(X_val)
    return {
        'accuracy': float(accuracy_score(y_val, y_pred)),
        'precision': float(precision_score(y_val, y_pred, zero_division=0)),
        'recall': float(recall_score(y_val, y_pred, zero_division=0)),
        'f1': float(f1_score(y_val, y_pred, zero_division=0))
    }


def validation(
    model: Pipeline,
    X_val: Union[pd.Series, list[str]],
    y_val: Union[pd.Series, list[int]]
) -> dict[str, float]:
    """
    Bí danh (alias) tương đương của hàm val(), giúp sinh viên linh hoạt khi gọi tên hàm.
    """
    return val(model, X_val, y_val)


def test(
    model: Pipeline,
    X_test: Union[pd.Series, list[str]],
    y_test: Union[pd.Series, list[int]],
    display: bool = True
) -> dict[str, Any]:
    """
    HÀM KIỂM THỬ (test):
    Đánh giá toàn diện mô hình trên tập kiểm thử độc lập (Test Set).
    Tính toán và trả về đầy đủ 5 chỉ số đánh giá tiêu chuẩn:
      1. Accuracy (Độ chính xác tổng quan)
      2. Precision (Độ chuẩn xác của nhãn Spam)
      3. Recall (Độ thu hồi / độ nhạy bắt trúng Spam)
      4. F1-Score (Trung bình điều hòa giữa Precision và Recall)
      5. Confusion Matrix (Ma trận nhầm lẫn kích thước 2x2)
      
    Args:
        model (Pipeline): Mô hình đã huấn luyện.
        X_test: Tập văn bản kiểm thử.
        y_test: Nhãn thực tế kiểm thử.
        display (bool): Nếu True, in kết quả trực quan dạng bảng ra console.
        
    Returns:
        dict[str, Any]: Chứa các chỉ số số học, ma trận nhầm lẫn numpy array, và báo cáo chuỗi.
    """
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    report = classification_report(
        y_test,
        y_pred,
        target_names=['Ham (0)', 'Spam (1)'],
        digits=4
    )

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }

    if display:
        print("\n" + "=" * 56)
        print("           KẾT QUẢ KIỂM THỬ (TEST METRICS)")
        print("=" * 56)
        print(f"  - Accuracy  (Độ chính xác tổng): {acc * 100:.2f}%")
        print(f"  - Precision (Độ chuẩn xác Spam): {prec * 100:.2f}%")
        print(f"  - Recall    (Độ thu hồi Spam)  : {rec * 100:.2f}%")
        print(f"  - F1-Score  (Điểm F1)          : {f1 * 100:.2f}%")
        print("\n  Ma trận nhầm lẫn (Confusion Matrix):")
        print("    ┌─────────────┬─────────────┐")
        print(f"    │ TN = {cm[0, 0]:<6} │ FP = {cm[0, 1]:<6} │  (Thực tế: Ham)")
        print("    ├─────────────┼─────────────┤")
        print(f"    │ FN = {cm[1, 0]:<6} │ TP = {cm[1, 1]:<6} │  (Thực tế: Spam)")
        print("    └─────────────┴─────────────┘")
        print("\n  Chi tiết Báo cáo Phân loại (Classification Report):")
        print(report)
        print("=" * 56 + "\n")

    return metrics


# Đánh dấu __test__ = False để pytest không nhầm lẫn hàm kiểm thử này là fixture/test case
test.__test__ = False


def save_model(model: Pipeline, filepath: str) -> None:
    """
    Lưu đối tượng Pipeline mô hình ra file .joblib.
    Tự động khởi tạo thư mục cha nếu chưa tồn tại.
    
    Args:
        model (Pipeline): Mô hình Pipeline cần lưu.
        filepath (str): Đường dẫn file lưu trữ.
    """
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[INFO] Đã lưu mô hình thành công tại: {filepath}")


def load_model(filepath: str) -> Pipeline:
    """
    Nạp mô hình Pipeline từ file .joblib đã lưu.
    
    Args:
        filepath (str): Đường dẫn tới file mô hình .joblib.
        
    Returns:
        Pipeline: Mô hình đã được phục hồi đầy đủ cả TF-IDF và MultinomialNB.
        
    Raises:
        FileNotFoundError: Nếu file mô hình không tồn tại.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {filepath}")
    model = joblib.load(filepath)
    return model


def predict(model: Pipeline, texts: list[str]) -> list[int]:
    """
    Dự đoán nhãn nhị phân cho một danh sách văn bản email.
    
    Args:
        model (Pipeline): Mô hình đã huấn luyện.
        texts (list[str]): Danh sách chuỗi văn bản cần phân loại.
        
    Returns:
        list[int]: Danh sách nhãn dự đoán (0: Ham, 1: Spam).
    """
    preds = model.predict(texts)
    return [int(p) for p in preds]


def predict_single(model: Pipeline, text: str) -> dict[str, Any]:
    """
    Dự đoán nhãn và xác suất chi tiết cho một văn bản email/tin nhắn đơn lẻ.
    
    Args:
        model (Pipeline): Mô hình đã huấn luyện.
        text (str): Chuỗi nội dung email cần kiểm tra.
        
    Returns:
        dict: Chứa {
            'text': văn bản đầu vào,
            'prediction': 0 hoặc 1,
            'label': 'Ham' hoặc 'Spam',
            'confidence': độ tin cậy (xác suất của nhãn dự đoán),
            'probabilities': {'Ham': float, 'Spam': float}
        }
    """
    proba = model.predict_proba([text])[0]
    pred = int(model.predict([text])[0])
    label_str = "Spam" if pred == 1 else "Ham"
    confidence = float(proba[pred])

    return {
        'text': text,
        'prediction': pred,
        'label': label_str,
        'confidence': confidence,
        'probabilities': {
            'Ham': float(proba[0]),
            'Spam': float(proba[1])
        }
    }


def explain_prediction(model: Pipeline, text: str, top_k: int = 8) -> list[dict[str, Any]]:
    """
    Giải thích các từ khóa quan trọng nhất trong email ảnh hưởng đến quyết định Spam/Ham.
    Tính điểm đóng góp = TF-IDF(từ) * (log P(từ | Spam) - log P(từ | Ham)).
      - Điểm dương (> 0): Từ kéo dự đoán về phía SPAM.
      - Điểm âm (< 0)   : Từ kéo dự đoán về phía HAM (Email thường).
    """
    vectorizer = model.named_steps['tfidf']
    nb = model.named_steps['nb']

    tfidf_vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    log_prob_diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]

    nonzero_indices = tfidf_vec.nonzero()[1]
    contributions = []
    for idx in nonzero_indices:
        score = float(tfidf_vec[0, idx] * log_prob_diff[idx])
        contributions.append({
            'word': str(feature_names[idx]),
            'score': score,
            'tendency': 'Spam' if score > 0 else 'Ham'
        })

    contributions.sort(key=lambda item: abs(item['score']), reverse=True)
    return contributions[:top_k]
