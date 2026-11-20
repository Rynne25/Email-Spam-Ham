"""
Module: app.py
Giao diện Web hiện đại (Streamlit) phục vụ trình chiếu đồ án và kiểm nghiệm
Hệ thống Phân loại Email Spam Tiếng Anh & Tiếng Việt (TF-IDF + Multinomial Naive Bayes).

Cách chạy:
    python -m streamlit run app.py
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.data_loader import (
    load_english_data,
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

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(
    page_title="SpamGuard AI — Phân loại Email Spam",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== GIAO DIỆN HIỆN ĐẠI (CSS) ====================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Giảm khoảng trắng thừa trên cùng */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Sidebar tối giản, sang trọng */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    .brand-box {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        padding: 1.15rem 1.1rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 20px -6px rgba(79, 70, 229, 0.45);
    }
    .brand-box h2 {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }
    .brand-box p {
        margin: 0.25rem 0 0 0;
        font-size: 0.82rem;
        color: rgba(255, 255, 255, 0.85);
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(125deg, #0F172A 0%, #1E1B4B 52%, #4338CA 100%);
        padding: 1.75rem 2rem;
        border-radius: 18px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 30px -10px rgba(15, 23, 42, 0.35);
    }
    .hero-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: #E0E7FF;
        padding: 0.28rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 0.65rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
        color: #FFFFFF;
    }
    .hero-sub {
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        color: #CBD5E1;
    }

    /* Tabs hiện đại dạng pill */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #EEF2F6;
        padding: 6px;
        border-radius: 12px;
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 0.55rem 1.15rem;
        font-weight: 600;
        font-size: 0.92rem;
        color: #475569;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #4F46E5 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    /* Card nội dung */
    .ui-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
        margin-bottom: 1rem;
    }

    /* Thẻ kết quả Spam / Ham */
    .result-box {
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0 1.2rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .result-spam {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 1px solid #FECDD3;
        border-left: 6px solid #E11D48;
    }
    .result-ham {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1px solid #A7F3D0;
        border-left: 6px solid #059669;
    }
    .result-title-spam { color: #9F1239; font-size: 1.3rem; font-weight: 800; margin: 0; }
    .result-title-ham  { color: #065F46; font-size: 1.3rem; font-weight: 800; margin: 0; }
    .result-desc { color: #334155; font-size: 0.93rem; margin: 0.3rem 0 0 0; }

    .conf-badge {
        padding: 0.55rem 1.1rem;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1.15rem;
        background: rgba(255, 255, 255, 0.85);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .conf-spam { color: #BE123C; }
    .conf-ham  { color: #047857; }

    /* Thanh xác suất hiện đại */
    .prob-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem 1.2rem;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.5rem;
    }
    .prob-label { font-weight: 600; font-size: 0.9rem; color: #475569; }
    .prob-value { font-weight: 800; font-size: 1.35rem; color: #0F172A; }
    .bar-track {
        width: 100%;
        height: 10px;
        background: #F1F5F9;
        border-radius: 999px;
        overflow: hidden;
    }
    .bar-fill-spam {
        height: 100%;
        background: linear-gradient(90deg, #FB7185 0%, #E11D48 100%);
        border-radius: 999px;
    }
    .bar-fill-ham {
        height: 100%;
        background: linear-gradient(90deg, #34D399 0%, #059669 100%);
        border-radius: 999px;
    }

    /* Keyword badges */
    .kw-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.38rem 0.8rem;
        margin: 0.25rem 0.35rem 0.25rem 0;
        border-radius: 999px;
        font-size: 0.86rem;
        font-weight: 600;
    }
    .kw-spam {
        background: #FFF1F2;
        color: #BE123C;
        border: 1px solid #FECDD3;
    }
    .kw-ham {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }

    /* KPI Metric cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .kpi-accent {
        color: #4F46E5;
    }

    /* Nút bấm bo tròn hiện đại */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease !important;
    }
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        font-size: 0.95rem !important;
        padding: 0.85rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== ĐƯỜNG DẪN & MẪU THỬ ====================
MODEL_PATHS = {
    'vi': os.path.join('models', 'model_vi.joblib'),
    'en': os.path.join('models', 'model_en.joblib'),
}

SPLIT_DIRS = {
    'vi': os.path.join('data', 'vi'),
    'en': os.path.join('data', 'en'),
}

QUICK_SAMPLES = {
    'vi': {
        '🚨 Spam Trúng thưởng': (
            "Chúc mừng quý khách đã trúng thưởng 50.000.000 VNĐ từ chương trình tri ân khách hàng! "
            "Vui lòng nhấn vào đường link ngay hôm nay để nhận quà miễn phí."
        ),
        '🚨 Spam Ngân hàng': (
            "KHẨN CẤP: Tài khoản ngân hàng của bạn đang bị tạm khóa do giao dịch bất thường. "
            "Truy cập ngay đường dẫn xác minh mã OTP trong 24h để mở khóa tài khoản."
        ),
        '✅ Email Họp nhóm': (
            "Chào các bạn, sáng thứ 5 tuần này lúc 9h nhóm mình họp tại phòng B203 để rà soát tiến độ "
            "đồ án cuối kỳ nhé. Nhớ mang theo slide báo cáo."
        ),
        '✅ Email Học tập': (
            "Thưa thầy, em gửi thầy file báo cáo và mã nguồn bài tập tuần 6 ạ. "
            "Em cảm ơn thầy đã xem bài."
        ),
    },
    'en': {
        '🚨 Lottery Spam': (
            "CONGRATULATIONS! You have been selected as our lucky winner of a $1,000,000 cash prize! "
            "Click here immediately to claim your free reward now."
        ),
        '🚨 Phishing Alert': (
            "URGENT: Your bank account has been suspended due to unauthorized login attempts. "
            "Verify your password and credit card details immediately to restore access."
        ),
        '✅ Work Meeting': (
            "Hi team, please find attached the agenda for tomorrow's quarterly project review at 10 AM. "
            "Let me know if you have any items to add."
        ),
        '✅ Academic Email': (
            "Dear Professor, I have submitted my assignment report for Week 5 on the portal. "
            "Please let me know if any additional files are needed. Best regards."
        ),
    },
}


@st.cache_resource
def get_model(lang: str):
    """Tải mô hình từ đĩa (nếu chưa có thì tự huấn luyện)."""
    model_path = MODEL_PATHS[lang]
    if os.path.exists(model_path):
        return load_model(model_path)
    X_train, _, _, y_train, _, _ = get_splits(lang)
    return train(X_train, y_train, language=lang, model_save_path=model_path)


@st.cache_data
def get_splits(lang: str):
    """Tải 3 tập train.csv, val.csv, test.csv (nếu chưa có thì tự tách và lưu)."""
    split_dir = SPLIT_DIRS[lang]
    train_file = os.path.join(split_dir, 'train.csv')
    val_file = os.path.join(split_dir, 'val.csv')
    test_file = os.path.join(split_dir, 'test.csv')

    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        return load_splits(split_dir)

    df = load_english_data() if lang == 'en' else load_vietnamese_data()
    splits = split_dataset(df, random_state=42)
    save_splits(*splits, output_dir=split_dir)
    return splits


def render_kpi_row(metrics: dict, highlight_f1: bool = True):
    """Hiển thị 4 thẻ KPI hiện đại cho Accuracy, Precision, Recall, F1-Score."""
    cols = st.columns(4)
    items = [
        ("Accuracy", metrics['accuracy'], False),
        ("Precision", metrics['precision'], False),
        ("Recall", metrics['recall'], False),
        ("F1-Score", metrics['f1'], highlight_f1),
    ]
    for col, (label, value, is_accent) in zip(cols, items):
        accent_cls = "kpi-accent" if is_accent else ""
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value {accent_cls}">{value * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def plot_confusion_matrix(cm, title: str):
    """Vẽ biểu đồ Ma trận nhầm lẫn (Confusion Matrix) phong cách tối giản hiện đại."""
    fig, ax = plt.subplots(figsize=(4.8, 3.9), dpi=140)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)

    classes = ['Ham (0)', 'Spam (1)']
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=classes,
        yticklabels=classes,
    )
    ax.set_title(title, fontsize=11.5, fontweight='bold', color='#0F172A', pad=12)
    ax.set_ylabel('Nhãn thực tế (Actual)', fontsize=9.5, color='#334155', fontweight='semibold')
    ax.set_xlabel('Nhãn dự đoán (Predicted)', fontsize=9.5, color='#334155', fontweight='semibold')

    for spine in ax.spines.values():
        spine.set_visible(False)

    thresh = cm.max() / 2.0
    labels_desc = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            color = '#FFFFFF' if cm[i, j] > thresh else '#0F172A'
            ax.text(
                j, i,
                f"{labels_desc[i][j]}\n{cm[i, j]:,}",
                ha='center', va='center',
                color=color, fontsize=11.5, fontweight='bold'
            )

    fig.tight_layout()
    return fig


# ==================== SIDEBAR TỐI GIẢN ====================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <h2>🛡️ SpamGuard AI</h2>
            <p>Phân loại Email Spam • Anh & Việt</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lang_label = st.radio(
        "🌐 Ngôn ngữ kiểm nghiệm",
        options=["🇻🇳 Tiếng Việt", "🇬🇧 Tiếng Anh"],
        index=0,
    )
    lang = 'vi' if "Tiếng Việt" in lang_label else 'en'

    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(lang)
    total = len(X_train) + len(X_val) + len(X_test)

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Huấn luyện lại mô hình", use_container_width=True):
        with st.spinner("Đang huấn luyện lại mô hình..."):
            t0 = time.time()
            train(X_train, y_train, language=lang, model_save_path=MODEL_PATHS[lang])
            elapsed = time.time() - t0
        get_model.clear()
        st.success(f"Hoàn tất trong {elapsed:.2f} giây!")


# Tải mô hình theo ngôn ngữ đang chọn
model = get_model(lang)
lang_display = "Tiếng Việt (Vietnamese)" if lang == 'vi' else "Tiếng Anh (English)"

# ==================== HERO BANNER ====================
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-pill">TF-IDF + Multinomial Naive Bayes • {lang_display}</div>
        <h1 class="hero-title">Hệ thống Nhận diện & Kiểm nghiệm Email Spam</h1>
        <p class="hero-sub">
            Phân tích nội dung email theo thời gian thực, hiển thị xác suất chi tiết, từ khóa nghi vấn và đánh giá chuẩn Train / Val / Test.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== TABS CHÍNH ====================
tab_demo, tab_eval, tab_data = st.tabs([
    "🔍 Kiểm thử Email trực tiếp",
    "📊 Đánh giá Train / Val / Test",
    "🗂️ Khám phá dữ liệu & Mẫu Test",
])

# ==================== TAB 1: LIVE DEMO ====================
with tab_demo:
    state_key = f"email_input_{lang}"
    if state_key not in st.session_state:
        st.session_state[state_key] = ""

    st.markdown("**⚡ Chọn nhanh mẫu thử để trình chiếu:**")
    sample_cols = st.columns(len(QUICK_SAMPLES[lang]))
    for col, (btn_label, sample_text) in zip(sample_cols, QUICK_SAMPLES[lang].items()):
        with col:
            if st.button(btn_label, use_container_width=True, key=f"btn_{lang}_{btn_label}"):
                st.session_state[state_key] = sample_text

    email_text = st.text_area(
        "✉️ Nội dung Email / Tin nhắn cần kiểm tra",
        value=st.session_state[state_key],
        height=165,
        placeholder=(
            "Dán nội dung email tiếng Việt vào đây để hệ thống phân tích..."
            if lang == 'vi'
            else "Paste English email content here for instant spam analysis..."
        ),
    )
    st.session_state[state_key] = email_text

    col_act1, col_act2, _ = st.columns([1.2, 0.9, 3])
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

            # Thẻ kết quả chính
            if is_spam:
                st.markdown(
                    f"""
                    <div class="result-box result-spam">
                        <div>
                            <p class="result-title-spam">🚨 PHÁT HIỆN SPAM / THƯ RÁC</p>
                            <p class="result-desc">Nội dung có dấu hiệu thư rác hoặc lừa đảo (Nhãn dự đoán: <b>1</b>).</p>
                        </div>
                        <div class="conf-badge conf-spam">Độ tin cậy {result['confidence']*100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box result-ham">
                        <div>
                            <p class="result-title-ham">✅ EMAIL AN TOÀN (HAM)</p>
                            <p class="result-desc">Nội dung bình thường, không phát hiện dấu hiệu spam (Nhãn dự đoán: <b>0</b>).</p>
                        </div>
                        <div class="conf-badge conf-ham">Độ tin cậy {result['confidence']*100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Thanh xác suất hiện đại
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(
                    f"""
                    <div class="prob-card">
                        <div class="prob-header">
                            <span class="prob-label">🚨 Xác suất SPAM (Thư rác)</span>
                            <span class="prob-value">{prob_spam:.2f}%</span>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill-spam" style="width: {min(max(prob_spam, 0), 100):.2f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with p2:
                st.markdown(
                    f"""
                    <div class="prob-card">
                        <div class="prob-header">
                            <span class="prob-label">✅ Xác suất HAM (Email thường)</span>
                            <span class="prob-value">{prob_ham:.2f}%</span>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill-ham" style="width: {min(max(prob_ham, 0), 100):.2f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Phân tích từ khóa nghi vấn
            st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)
            st.markdown("#### 🔎 Phân tích từ khóa ảnh hưởng đến quyết định")
            keywords = explain_prediction(model, email_text, top_k=10)
            if not keywords:
                st.info("Không tìm thấy từ khóa trùng khớp trong từ điển TF-IDF của mô hình.")
            else:
                spam_words = [k for k in keywords if k['tendency'] == 'Spam']
                ham_words = [k for k in keywords if k['tendency'] == 'Ham']

                kw1, kw2 = st.columns(2)
                with kw1:
                    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
                    st.markdown("**🚨 Từ khóa nghi vấn (kéo về SPAM):**")
                    if spam_words:
                        badges = "".join(
                            f'<span class="kw-badge kw-spam">{k["word"]} <span>+{k["score"]:.2f}</span></span>'
                            for k in spam_words
                        )
                        st.markdown(badges, unsafe_allow_html=True)
                    else:
                        st.caption("Không phát hiện từ khóa nghi vấn nổi bật.")
                    st.markdown('</div>', unsafe_allow_html=True)

                with kw2:
                    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
                    st.markdown("**✅ Từ khóa an toàn (kéo về HAM):**")
                    if ham_words:
                        badges = "".join(
                            f'<span class="kw-badge kw-ham">{k["word"]} <span>{k["score"]:.2f}</span></span>'
                            for k in ham_words
                        )
                        st.markdown(badges, unsafe_allow_html=True)
                    else:
                        st.caption("Không phát hiện từ khóa an toàn nổi bật.")
                    st.markdown('</div>', unsafe_allow_html=True)


# ==================== TAB 2: EVALUATION ====================
with tab_eval:
    val_metrics = val(model, X_val, y_val)
    test_metrics = test(model, X_test, y_test, display=False)

    st.markdown("#### 🎯 Kết quả trên tập Kiểm thử độc lập (`test()`)")
    render_kpi_row(test_metrics, highlight_f1=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🧪 Kết quả trên tập Kiểm định (`val()`)")
    render_kpi_row(val_metrics, highlight_f1=False)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    chart_col, report_col = st.columns([1, 1.25])
    with chart_col:
        st.markdown("#### 📐 Ma trận nhầm lẫn (Confusion Matrix)")
        fig = plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            title=f"Test Set ({'Tiếng Việt' if lang == 'vi' else 'Tiếng Anh'})",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with report_col:
        st.markdown("#### 📋 Báo cáo phân loại chi tiết")
        st.code(test_metrics['classification_report'], language="text")

        st.markdown("#### 📦 Phân bổ tập dữ liệu (70 / 15 / 15)")
        split_summary = pd.DataFrame([
            {
                'Tập': 'Train',
                'File': f'data/{lang}/train.csv',
                'Số mẫu': len(X_train),
                'Tỷ lệ': f"{len(X_train)/total*100:.1f}%",
                'Spam': int(y_train.sum()),
                'Ham': int((y_train == 0).sum()),
            },
            {
                'Tập': 'Validation',
                'File': f'data/{lang}/val.csv',
                'Số mẫu': len(X_val),
                'Tỷ lệ': f"{len(X_val)/total*100:.1f}%",
                'Spam': int(y_val.sum()),
                'Ham': int((y_val == 0).sum()),
            },
            {
                'Tập': 'Test',
                'File': f'data/{lang}/test.csv',
                'Số mẫu': len(X_test),
                'Tỷ lệ': f"{len(X_test)/total*100:.1f}%",
                'Spam': int(y_test.sum()),
                'Ham': int((y_test == 0).sum()),
            },
        ])
        st.dataframe(split_summary, use_container_width=True, hide_index=True)


# ==================== TAB 3: DATASET & RANDOM TEST CHECK ====================
with tab_data:
    st.markdown("#### 🎲 Đối chiếu nhanh 10 mẫu ngẫu nhiên từ tập `test.csv`")
    seed_key = f"sample_seed_{lang}"
    if seed_key not in st.session_state:
        st.session_state[seed_key] = 42

    if st.button("🔀 Đổi 10 mẫu Test ngẫu nhiên khác", key=f"rand_{lang}"):
        st.session_state[seed_key] += 1

    test_df = pd.DataFrame({'text': X_test.values, 'label': y_test.values})
    sample_df = test_df.sample(n=min(10, len(test_df)), random_state=st.session_state[seed_key]).copy()

    rows = []
    for _, row in sample_df.iterrows():
        pred_info = predict_single(model, row['text'])
        actual_str = "Spam" if row['label'] == 1 else "Ham"
        pred_str = pred_info['label']
        rows.append({
            'Nội dung (rút gọn)': row['text'][:140] + ('...' if len(row['text']) > 140 else ''),
            'Thực tế': actual_str,
            'Dự đoán': pred_str,
            'Độ tin cậy': f"{pred_info['confidence']*100:.1f}%",
            'Trạng thái': '✅ Đúng' if actual_str == pred_str else '❌ Sai',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📂 Xem dữ liệu trong các file CSV")
    split_choice = st.selectbox("Chọn tập dữ liệu:", ["train.csv", "val.csv", "test.csv"])
    file_to_view = os.path.join(SPLIT_DIRS[lang], split_choice)
    if os.path.exists(file_to_view):
        preview_df = pd.read_csv(file_to_view).head(20)
        preview_df['Nhãn'] = preview_df['label'].map({0: 'Ham (0)', 1: 'Spam (1)'})
        st.dataframe(preview_df[['text', 'Nhãn']], use_container_width=True, hide_index=True)
