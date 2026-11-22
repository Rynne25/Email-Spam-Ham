"""
Module: app.py
Giao diện Web Streamlit Đồ án Phân loại Email Spam Tiếng Việt
(TF-IDF n-gram + Multinomial Naive Bayes).
Giao diện tối giản toàn màn hình, không sidebar, trang bị trọn bộ biểu đồ trực quan.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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

# ==================== CẤU HÌNH TRANG TOÀN MÀN HÌNH ====================
st.set_page_config(
    page_title="SpamGuard AI — Phân loại Email Spam Tiếng Việt",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================== GIAO DIỆN HIỆN ĐẠI (CLEAN SLATE & INDIGO) ====================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }

    /* Ẩn hoàn toàn sidebar thừa */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }

    /* Căn lề thoáng đãng */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Header thương hiệu */
    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 60%, #4338CA 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 18px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #E0E7FF;
        padding: 0.28rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .header-title {
        font-size: 1.95rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
        color: #FFFFFF;
    }
    .header-desc {
        margin: 0.45rem 0 0 0;
        font-size: 0.96rem;
        color: #CBD5E1;
        line-height: 1.5;
    }

    /* Tabs pill hiện đại */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 0.65rem 1.4rem;
        font-weight: 700;
        font-size: 0.95rem;
        color: #475569;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #4F46E5 !important;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.08);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    /* Card giao diện */
    .metric-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
        margin-bottom: 1rem;
    }

    /* Thẻ kết quả phân loại */
    .result-banner {
        border-radius: 16px;
        padding: 1.4rem 1.7rem;
        margin: 1rem 0 1.2rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .result-banner-spam {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 1px solid #FECDD3;
        border-left: 6px solid #E11D48;
    }
    .result-banner-ham {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1px solid #A7F3D0;
        border-left: 6px solid #059669;
    }
    .result-title-spam { color: #9F1239; font-size: 1.35rem; font-weight: 800; margin: 0; }
    .result-title-ham  { color: #065F46; font-size: 1.35rem; font-weight: 800; margin: 0; }
    .result-desc { color: #334155; font-size: 0.92rem; margin: 0.3rem 0 0 0; }

    .score-badge {
        padding: 0.55rem 1.2rem;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1.2rem;
        background: #FFFFFF;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    .score-spam { color: #BE123C; }
    .score-ham  { color: #047857; }

    /* Thanh đo xác suất */
    .gauge-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem 1.25rem;
    }
    .gauge-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.5rem;
    }
    .gauge-title { font-weight: 600; font-size: 0.9rem; color: #475569; }
    .gauge-pct   { font-weight: 800; font-size: 1.4rem; color: #0F172A; }
    .gauge-track {
        width: 100%;
        height: 11px;
        background: #F1F5F9;
        border-radius: 999px;
        overflow: hidden;
    }
    .gauge-fill-spam {
        height: 100%;
        background: linear-gradient(90deg, #FB7185 0%, #E11D48 100%);
        border-radius: 999px;
    }
    .gauge-fill-ham {
        height: 100%;
        background: linear-gradient(90deg, #34D399 0%, #059669 100%);
        border-radius: 999px;
    }

    /* KPI metric card */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }
    .kpi-name {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .kpi-val {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .kpi-highlight { color: #4F46E5; }

    /* Khung text area và button */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        font-size: 0.95rem !important;
        padding: 0.9rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12) !important;
    }
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== ĐƯỜNG DẪN & DỮ LIỆU MẪU ====================
MODEL_PATH = os.path.join('models', 'model_vi.joblib')
SPLIT_DIR = os.path.join('data', 'vi')

QUICK_SAMPLES = {
    '🚨 Spam Trúng thưởng': (
        "Chúc mừng quý khách đã trúng thưởng 50.000.000 VNĐ từ chương trình tri ân khách hàng! "
        "Vui lòng nhấn vào đường link ngay hôm nay để nhận quà miễn phí."
    ),
    '🚨 Spam Ngân hàng': (
        "KHẨN CẤP: Tài khoản ngân hàng của bạn đang bị tạm khóa do giao dịch bất thường. "
        "Truy cập ngay đường dẫn xác minh mã OTP trong 24h để mở khóa tài khoản."
    ),
    '🚨 Spam Tuyển dụng': (
        "Tuyển cộng tác viên online làm việc tại nhà, xử lý đơn hàng Shopee và Lazada. "
        "Thu nhập 500k - 1.5 triệu/ngày, nhận tiền ngay trong ngày, không cọc vốn. Nhắn Zalo 0987xxx để nhận việc ngay."
    ),
    '🚨 Spam Vay tín dụng': (
        "Ngân hàng hỗ trợ vay tiêu dùng giải ngân từ 20 đến 100 triệu, lãi suất 0%, không thẩm định người thân, "
        "chấp nhận nợ xấu, nhận tiền sau 15 phút. Nhấn vào link vaytiennhanh.vn để đăng ký."
    ),
    '🚨 Spam Giả mạo Giao hàng': (
        "GiaoHangTietKiem: Đơn hàng #VN89214 của bạn không thể phát do sai địa chỉ nhận hàng. "
        "Vui lòng truy cập ghtk-xacminh.com để cập nhật lại thông tin trong 12 giờ tránh bị hoàn đơn."
    ),
    '✅ Email Họp nhóm': (
        "Chào các bạn, sáng thứ 5 tuần này lúc 9h nhóm mình họp tại phòng B203 để rà soát tiến độ "
        "đồ án cuối kỳ nhé. Nhớ mang theo slide báo cáo."
    ),
    '✅ Email Học tập': (
        "Thưa thầy, em gửi thầy file báo cáo và mã nguồn bài tập tuần 6 ạ. "
        "Em cảm ơn thầy đã xem bài."
    ),
    '✅ Email Công việc': (
        "Kính gửi phòng ban liên quan, tôi xin gửi biên bản cuộc họp tổng kết tuần vừa qua "
        "cùng danh sách công việc cần hoàn thành trước thứ Sáu. Trân trọng."
    ),
}


def get_model():
    """Tải trực tiếp mô hình mới nhất từ đĩa."""
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    X_train, _, _, y_train, _, _ = get_splits()
    return train(X_train, y_train, language='vi', model_save_path=MODEL_PATH)


@st.cache_data
def get_splits():
    """Tải và phân chia tập dữ liệu 70/15/15."""
    train_file = os.path.join(SPLIT_DIR, 'train.csv')
    val_file = os.path.join(SPLIT_DIR, 'val.csv')
    test_file = os.path.join(SPLIT_DIR, 'test.csv')

    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        return load_splits(SPLIT_DIR)

    df = load_vietnamese_data()
    splits = split_dataset(df, random_state=42)
    save_splits(*splits, output_dir=SPLIT_DIR)
    return splits


@st.cache_data
def get_evaluation_metrics(_model, _X_val, _y_val, _X_test, _y_test):
    """Tính toán và cache các chỉ số đánh giá."""
    return val(_model, _X_val, _y_val), test(_model, _X_test, _y_test, display=False)


# ==================== CÁC HÀM VẼ BIỂU ĐỒ TRỰC QUAN ====================
def plot_metrics_bar(test_metrics: dict):
    """Vẽ biểu đồ cột trực quan cho 4 chỉ số hiệu năng Accuracy, Precision, Recall, F1."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=140)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        test_metrics['accuracy'] * 100,
        test_metrics['precision'] * 100,
        test_metrics['recall'] * 100,
        test_metrics['f1'] * 100,
    ]
    colors = ['#6366F1', '#3B82F6', '#06B6D4', '#10B981']

    bars = ax.bar(names, values, color=colors, width=0.48, edgecolor='none')
    ax.set_ylim(0, 115)
    ax.set_ylabel('Tỷ lệ (%)', fontsize=9.5, color='#475569', fontweight='semibold')
    ax.set_title('Hiệu năng Mô hình trên Tập Kiểm thử (Test Set)', fontsize=11, fontweight='bold', color='#0F172A', pad=12)

    for bar, val_num in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.5,
            f"{val_num:.2f}%",
            ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#1E293B'
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E2E8F0')
    ax.spines['bottom'].set_color('#E2E8F0')
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#94A3B8')

    fig.tight_layout()
    return fig


def plot_confusion_matrix_styled(cm):
    """Vẽ biểu đồ Ma trận nhầm lẫn hiện đại."""
    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=140)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)

    classes = ['Hợp lệ (0)', 'Spam (1)']
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=classes, yticklabels=classes)
    ax.set_title('Ma trận Nhầm lẫn (Confusion Matrix)', fontsize=11, fontweight='bold', color='#0F172A', pad=12)
    ax.set_ylabel('Nhãn thực tế (Actual)', fontsize=9.5, color='#334155', fontweight='semibold')
    ax.set_xlabel('Nhãn dự đoán (Predicted)', fontsize=9.5, color='#334155', fontweight='semibold')

    for spine in ax.spines.values():
        spine.set_visible(False)

    thresh = cm.max() / 2.0
    labels_desc = [['TN (Đúng Hợp lệ)', 'FP (Báo nhầm Spam)'], ['FN (Sót Spam)', 'TP (Bắt đúng Spam)']]
    for i in range(2):
        for j in range(2):
            color = '#FFFFFF' if cm[i, j] > thresh else '#0F172A'
            ax.text(
                j, i,
                f"{labels_desc[i][j]}\n{cm[i, j]:,}",
                ha='center', va='center',
                color=color, fontsize=10.5, fontweight='bold'
            )

    fig.tight_layout()
    return fig


def plot_dataset_split_chart(X_train, X_val, X_test, y_train, y_val, y_test):
    """Vẽ biểu đồ phân bổ số lượng mẫu Hợp lệ vs Spam trên cả 3 tập Train/Val/Test."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=140)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    splits = ['Tập Train (70%)', 'Tập Val (15%)', 'Tập Test (15%)']
    ham_counts = [int((y_train == 0).sum()), int((y_val == 0).sum()), int((y_test == 0).sum())]
    spam_counts = [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())]

    x = np.arange(len(splits))
    width = 0.35

    bar1 = ax.bar(x - width/2, ham_counts, width, label='Email Hợp lệ', color='#10B981')
    bar2 = ax.bar(x + width/2, spam_counts, width, label='Spam / Thư rác', color='#F43F5E')

    ax.set_ylabel('Số lượng mẫu', fontsize=9.5, color='#475569', fontweight='semibold')
    ax.set_title('Phân bổ Mẫu trên 3 Tập Dữ liệu (Stratified Split)', fontsize=11, fontweight='bold', color='#0F172A', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=9, fontweight='semibold')
    ax.legend(frameon=False, loc='upper right')

    for bar in list(bar1) + list(bar2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(ham_counts) * 0.02,
            f"{h:,}",
            ha='center', va='bottom',
            fontsize=8.5, fontweight='bold', color='#334155'
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E2E8F0')
    ax.spines['bottom'].set_color('#E2E8F0')
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#94A3B8')

    fig.tight_layout()
    return fig


def plot_top_keywords(model, top_n=10):
    """Vẽ biểu đồ Top từ khóa đặc trưng nhất của nhãn Spam và nhãn Hợp lệ."""
    vectorizer = model.named_steps['tfidf']
    nb = model.named_steps['nb']
    feature_names = np.array(vectorizer.get_feature_names_out())
    log_diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]

    spam_idx = np.argsort(log_diff)[-top_n:][::-1]
    ham_idx = np.argsort(log_diff)[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=140)
    fig.patch.set_facecolor('#FFFFFF')

    # Top Spam
    ax1.set_facecolor('#FFFFFF')
    y_pos1 = np.arange(top_n)
    ax1.barh(y_pos1, log_diff[spam_idx][::-1], color='#F43F5E', height=0.6)
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels(feature_names[spam_idx][::-1], fontsize=9.5, fontweight='semibold')
    ax1.set_title(f'Top {top_n} Từ khóa Đặc trưng SPAM', fontsize=11, fontweight='bold', color='#9F1239', pad=10)
    ax1.set_xlabel('Hệ số Log-Odds (Kéo về Spam)', fontsize=9, color='#475569')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Top Ham
    ax2.set_facecolor('#FFFFFF')
    y_pos2 = np.arange(top_n)
    ax2.barh(y_pos2, (-log_diff[ham_idx])[::-1], color='#10B981', height=0.6)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(feature_names[ham_idx][::-1], fontsize=9.5, fontweight='semibold')
    ax2.set_title(f'Top {top_n} Từ khóa Đặc trưng HỢP LỆ', fontsize=11, fontweight='bold', color='#065F46', pad=10)
    ax2.set_xlabel('Hệ số Log-Odds (Kéo về Hợp lệ)', fontsize=9, color='#475569')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


def plot_keyword_contribution_bar(keywords: list):
    """Vẽ mini bar chart thể hiện điểm đóng góp của các từ khóa trong câu được test."""
    if not keywords:
        return None

    words = [k['word'] for k in keywords][::-1]
    scores = [k['score'] for k in keywords][::-1]
    colors = ['#F43F5E' if s > 0 else '#10B981' for s in scores]

    fig, ax = plt.subplots(figsize=(8.2, max(2.6, len(keywords) * 0.35)), dpi=140)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    y_pos = np.arange(len(words))
    ax.barh(y_pos, scores, color=colors, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=9.5, fontweight='semibold')
    ax.axvline(0, color='#94A3B8', linewidth=1)
    ax.set_xlabel('Mức độ ảnh hưởng: > 0 (kéo về SPAM) | < 0 (kéo về HỢP LỆ)', fontsize=8.8, color='#475569', labelpad=8)
    ax.set_title('Điểm Trọng số Đóng góp của Từng Từ khóa trong Email', fontsize=10.5, fontweight='bold', color='#0F172A', pad=10)

    for i, s in enumerate(scores):
        align = 'left' if s >= 0 else 'right'
        offset = 0.03 if s >= 0 else -0.03
        ax.text(s + offset, i, f"{s:+.2f}", va='center', ha=align, fontsize=8.5, fontweight='bold', color='#1E293B')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E2E8F0')
    ax.spines['bottom'].set_color('#E2E8F0')
    ax.grid(axis='x', linestyle='--', alpha=0.3, color='#94A3B8')

    fig.tight_layout()
    return fig


# ==================== DỮ LIỆU & MÔ HÌNH ====================
model = get_model()
X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
total_samples = len(X_train) + len(X_val) + len(X_test)

# ==================== HEADER THƯƠNG HIỆU ====================
st.markdown(
    f"""
    <div class="app-header">
        <div class="header-badge">🛡️ TF-IDF (1, 2) + Multinomial Naive Bayes • Tiếng Việt</div>
        <h1 class="header-title">SpamGuard AI — Phân Loại Email & Tin Nhắn Spam</h1>
        <p class="header-desc">
            Hệ thống nhận diện thư rác và email lừa đảo tối giản dành riêng cho ngữ cảnh Tiếng Việt. 
            Quy mô dữ liệu: <b>{total_samples:,}</b> mẫu thực tế (Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== 2 TABS CHÍNH ====================
tab_demo, tab_report = st.tabs([
    "🔍 Kiểm thử Email trực tiếp",
    "📊 Báo cáo Thông số & Biểu đồ Trực quan",
])

# ==================== TAB 1: TRẢI NGHIỆM PHÂN LOẠI EMAIL ====================
with tab_demo:
    state_key = "email_input_vi"
    if state_key not in st.session_state:
        st.session_state[state_key] = ""

    st.markdown("##### ⚡ Chọn nhanh mẫu thử để trình chiếu demo:")
    sample_items = list(QUICK_SAMPLES.items())

    cols_row1 = st.columns(4)
    for col, (btn_label, sample_text) in zip(cols_row1, sample_items[:4]):
        with col:
            if st.button(btn_label, use_container_width=True, key=f"btn_{btn_label}"):
                st.session_state[state_key] = sample_text
                st.rerun()

    cols_row2 = st.columns(4)
    for col, (btn_label, sample_text) in zip(cols_row2, sample_items[4:]):
        with col:
            if st.button(btn_label, use_container_width=True, key=f"btn_{btn_label}"):
                st.session_state[state_key] = sample_text
                st.rerun()

    email_text = st.text_area(
        "✉️ Nội dung Email / Tin nhắn cần kiểm tra",
        key=state_key,
        height=150,
        placeholder="Dán nội dung email hoặc tin nhắn tiếng Việt vào đây để hệ thống phân tích...",
    )

    col_act1, col_act2, _ = st.columns([1.3, 1.0, 3.5])
    with col_act1:
        predict_clicked = st.button("🚀 Phân loại ngay", type="primary", use_container_width=True)
    with col_act2:
        if st.button("🧹 Xóa nội dung", use_container_width=True):
            st.session_state[state_key] = ""
            st.rerun()

    if predict_clicked:
        if not email_text.strip():
            st.warning("Vui lòng nhập nội dung email hoặc bấm chọn một mẫu thử nhanh ở trên.")
        else:
            result = predict_single(model, email_text)
            prob_spam = result['probabilities']['Spam'] * 100
            prob_ham = result['probabilities']['Ham'] * 100
            is_spam = result['prediction'] == 1

            # Thẻ kết luận
            if is_spam:
                st.markdown(
                    f"""
                    <div class="result-banner result-banner-spam">
                        <div>
                            <p class="result-title-spam">🚨 PHÁT HIỆN SPAM / THƯ RÁC</p>
                            <p class="result-desc">Nội dung chứa dấu hiệu thư rác, quảng cáo hoặc lừa đảo (Nhãn dự đoán: <b>1</b>).</p>
                        </div>
                        <div class="score-badge score-spam">Độ tin cậy: {result['confidence']*100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-banner result-banner-ham">
                        <div>
                            <p class="result-title-ham">✅ EMAIL HỢP LỆ / AN TOÀN (HAM)</p>
                            <p class="result-desc">Nội dung thư thông thường, không phát hiện dấu hiệu bất thường (Nhãn dự đoán: <b>0</b>).</p>
                        </div>
                        <div class="score-badge score-ham">Độ tin cậy: {result['confidence']*100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 2 Thanh đo xác suất
            g1, g2 = st.columns(2)
            with g1:
                st.markdown(
                    f"""
                    <div class="gauge-box">
                        <div class="gauge-header">
                            <span class="gauge-title">🚨 Xác suất SPAM (Thư rác)</span>
                            <span class="gauge-pct">{prob_spam:.2f}%</span>
                        </div>
                        <div class="gauge-track">
                            <div class="gauge-fill-spam" style="width: {min(max(prob_spam, 0), 100):.2f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with g2:
                st.markdown(
                    f"""
                    <div class="gauge-box">
                        <div class="gauge-header">
                            <span class="gauge-title">✅ Xác suất HỢP LỆ (Email thường)</span>
                            <span class="gauge-pct">{prob_ham:.2f}%</span>
                        </div>
                        <div class="gauge-track">
                            <div class="gauge-fill-ham" style="width: {min(max(prob_ham, 0), 100):.2f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Biểu đồ thanh phân tích đóng góp từ khóa
            st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
            st.markdown("##### 📈 Trực quan hóa Điểm Ảnh Hưởng của Từ Khóa")
            keywords = explain_prediction(model, email_text, top_k=8)
            if not keywords:
                st.info("Không tìm thấy từ khóa đặc trưng trùng khớp trong từ điển TF-IDF.")
            else:
                fig_kw = plot_keyword_contribution_bar(keywords)
                if fig_kw:
                    st.pyplot(fig_kw, use_container_width=True)
                    plt.close(fig_kw)


# ==================== TAB 2: BÁO CÁO THÔNG SỐ & BIỂU ĐỒ TRỰC QUAN ====================
with tab_report:
    val_metrics, test_metrics = get_evaluation_metrics(model, X_val, y_val, X_test, y_test)

    st.markdown("##### 🎯 Các Chỉ Số Đánh Giá Chuẩn Trên Tập Test (Test Metrics)")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-name">Accuracy (Độ chính xác)</div><div class="kpi-val">{test_metrics['accuracy']*100:.2f}%</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-name">Precision (Độ chuẩn xác)</div><div class="kpi-val">{test_metrics['precision']*100:.2f}%</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-name">Recall (Độ nhạy)</div><div class="kpi-val">{test_metrics['recall']*100:.2f}%</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-name">F1-Score</div><div class="kpi-val kpi-highlight">{test_metrics['f1']*100:.2f}%</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Cụm biểu đồ 1: Cột 4 chỉ số + Heatmap Ma trận nhầm lẫn
    col_chart1, col_chart2 = st.columns([1.1, 1])
    with col_chart1:
        fig_bar = plot_metrics_bar(test_metrics)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    with col_chart2:
        fig_cm = plot_confusion_matrix_styled(test_metrics['confusion_matrix'])
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # Cụm biểu đồ 2: Phân bổ dữ liệu 3 tập + Top từ khóa đặc trưng
    col_split, col_kw_all = st.columns([1, 1.5])
    with col_split:
        fig_split = plot_dataset_split_chart(X_train, X_val, X_test, y_train, y_val, y_test)
        st.pyplot(fig_split, use_container_width=True)
        plt.close(fig_split)

    with col_kw_all:
        fig_top_kw = plot_top_keywords(model, top_n=8)
        st.pyplot(fig_top_kw, use_container_width=True)
        plt.close(fig_top_kw)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown("##### 📋 Báo cáo Phân loại Chi tiết (Classification Report)")
    st.code(test_metrics['classification_report'], language="text")
