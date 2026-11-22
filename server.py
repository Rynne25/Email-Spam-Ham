import os
from typing import Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.data_loader import (
    load_vietnamese_data,
    split_dataset,
    save_splits,
    load_splits,
)
from src.classifier import (
    train,
    val,
    test,
    load_model,
    predict_single,
    explain_prediction,
)

MODEL_PATH = os.path.join("models", "model_vi.joblib")
SPLIT_DIR = os.path.join("data", "vi")

app = FastAPI(title="SpamGuard AI — Liquid Glass API", version="2.0.0")

# Cache in-memory
_cached_model = None
_cached_splits = None
_cached_metrics = None


def get_cached_splits():
    global _cached_splits
    if _cached_splits is not None:
        return _cached_splits

    train_file = os.path.join(SPLIT_DIR, "train.csv")
    val_file = os.path.join(SPLIT_DIR, "val.csv")
    test_file = os.path.join(SPLIT_DIR, "test.csv")

    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        _cached_splits = load_splits(SPLIT_DIR)
        return _cached_splits

    df = load_vietnamese_data()
    splits = split_dataset(df, random_state=42)
    save_splits(*splits, output_dir=SPLIT_DIR)
    _cached_splits = splits
    return _cached_splits


def get_cached_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if os.path.exists(MODEL_PATH):
        _cached_model = load_model(MODEL_PATH)
        return _cached_model

    X_train, _, _, y_train, _, _ = get_cached_splits()
    _cached_model = train(X_train, y_train, language="vi", model_save_path=MODEL_PATH)
    return _cached_model


def get_cached_metrics():
    global _cached_metrics
    if _cached_metrics is not None:
        return _cached_metrics

    model = get_cached_model()
    X_train, X_val, X_test, y_train, y_val, y_test = get_cached_splits()
    val_res = val(model, X_val, y_val)
    test_res = test(model, X_test, y_test, display=False)

    cm = test_res["confusion_matrix"].tolist()

    # Top keywords
    vectorizer = model.named_steps["tfidf"]
    nb = model.named_steps["nb"]
    fn = np.array(vectorizer.get_feature_names_out())
    diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]

    top_n = 8
    spam_idx = np.argsort(diff)[-top_n:][::-1]
    ham_idx = np.argsort(diff)[:top_n]

    top_spam = [{"word": str(fn[i]), "score": float(diff[i])} for i in spam_idx]
    top_ham = [{"word": str(fn[i]), "score": float(-diff[i])} for i in ham_idx]

    _cached_metrics = {
        "val": val_res,
        "test": {
            "accuracy": test_res["accuracy"],
            "precision": test_res["precision"],
            "recall": test_res["recall"],
            "f1": test_res["f1"],
            "confusion_matrix": cm,
            "classification_report": test_res["classification_report"],
        },
        "top_keywords": {
            "spam": top_spam,
            "ham": top_ham,
        },
        "splits": {
            "train": {"total": len(X_train), "ham": int((y_train == 0).sum()), "spam": int(y_train.sum())},
            "val": {"total": len(X_val), "ham": int((y_val == 0).sum()), "spam": int(y_val.sum())},
            "test": {"total": len(X_test), "ham": int((y_test == 0).sum()), "spam": int(y_test.sum())},
        },
    }
    return _cached_metrics


class PredictRequest(BaseModel):
    text: str


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "SpamGuard AI", "engine": "FastAPI + Liquid Glass"}


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung email.")

    model = get_cached_model()
    result = predict_single(model, text)
    explanation = explain_prediction(model, text, top_k=10)

    return {
        "prediction": result["prediction"],
        "label": result["label"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "explanation": explanation,
    }


@app.get("/api/metrics")
def api_metrics():
    return get_cached_metrics()


@app.get("/api/samples")
def api_samples():
    preset_spam = {
        "🚨 Trúng thưởng": "Chúc mừng quý khách đã trúng thưởng 50.000.000 VNĐ từ chương trình tri ân khách hàng! Vui lòng nhấn vào đường link ngay hôm nay để nhận quà miễn phí.",
        "🚨 Ngân hàng": "KHẨN CẤP: Tài khoản ngân hàng của bạn đang bị tạm khóa do giao dịch bất thường. Truy cập ngay đường dẫn xác minh mã OTP trong 24h để mở khóa tài khoản.",
        "🚨 Tuyển dụng": "Tuyển cộng tác viên online làm việc tại nhà, xử lý đơn hàng Shopee và Lazada. Thu nhập 500k - 1.5 triệu/ngày, nhận tiền ngay trong ngày, không cọc vốn. Nhắn Zalo 0987xxx để nhận việc ngay.",
        "🚨 Vay tín dụng": "Ngân hàng hỗ trợ vay tiêu dùng giải ngân từ 20 đến 100 triệu, lãi suất 0%, không thẩm định người thân, chấp nhận nợ xấu, nhận tiền sau 15 phút. Nhấn vào link vaytiennhanh.vn để đăng ký.",
        "🚨 Giả mạo Shipper": "GiaoHangTietKiem: Đơn hàng #VN89214 của bạn không thể phát do sai địa chỉ nhận hàng. Vui lòng truy cập ghtk-xacminh.com để cập nhật lại thông tin trong 12 giờ tránh bị hoàn đơn.",
    }
    preset_ham = {
        "✅ Họp nhóm": "Chào các bạn, sáng thứ 5 tuần này lúc 9h nhóm mình họp tại phòng B203 để rà soát tiến độ đồ án cuối kỳ nhé. Nhớ mang theo slide báo cáo.",
        "✅ Học tập": "Thưa thầy, em gửi thầy file báo cáo và mã nguồn bài tập tuần 6 ạ. Em cảm ơn thầy đã xem bài.",
        "✅ Công việc": "Kính gửi phòng ban liên quan, tôi xin gửi biên bản cuộc họp tổng kết tuần vừa qua cùng danh sách công việc cần hoàn thành trước thứ Sáu. Trân trọng.",
    }
    return {
        "spam": preset_spam,
        "ham": preset_ham,
    }


@app.get("/api/random-tests")
def api_random_tests(n: int = 10):
    _, _, X_test, _, _, y_test = get_cached_splits()
    model = get_cached_model()

    df_test = pd.DataFrame({"text": X_test, "actual": y_test}).sample(n=min(n, len(X_test)), random_state=None)
    preds = model.predict(df_test["text"])
    probas = model.predict_proba(df_test["text"])

    samples = []
    for idx, (_, row) in enumerate(df_test.iterrows()):
        pred = int(preds[idx])
        conf = float(probas[idx][pred])
        samples.append({
            "text": row["text"][:110] + ("..." if len(row["text"]) > 110 else ""),
            "full_text": row["text"],
            "actual": int(row["actual"]),
            "actual_label": "Spam" if row["actual"] == 1 else "Ham",
            "predicted": pred,
            "predicted_label": "Spam" if pred == 1 else "Ham",
            "confidence": conf,
            "is_correct": bool(pred == row["actual"]),
        })

    return {"samples": samples}


# Serve static web files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_file):
        return {"message": "index.html is being prepared"}
    return FileResponse(index_file)
