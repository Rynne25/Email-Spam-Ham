"""
Module: app.py
Giao diện Web Streamlit Đồ án Phân loại Email Spam Tiếng Việt
(TF-IDF n-gram + Multinomial Naive Bayes).
Kiến trúc: Không gian làm việc 2 Cột (Split-screen Workspace), phong cách Technical Precision,
tích hợp trọn bộ Plotly Interactive Charts, tối ưu hóa 100% trải nghiệm người dùng và thuyết trình đồ án.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

# ==================== PHONG CÁCH KỸ THUẬT TINH XẢO (TECHNICAL PRECISION) ====================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
        background-color: #F8FAFC;
    }

    /* Ẩn sidebar và header mặc định */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* Căn lề layout rộng rãi */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px;
    }

    /* Top Bar thanh thoát chuẩn SaaS */
    .top-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 0.9rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-logo {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0F172A;
        margin: 0;
    }
    .brand-badge {
        background: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
    }
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #475569;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
    }

    /* Workspace Card Container */
    .ws-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
        margin-bottom: 1rem;
    }
    .ws-card-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Section Label */
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.45rem;
    }

    /* Thẻ kết luận kết quả */
    .verdict-box {
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }
    .verdict-spam {
        background: #FFF1F2;
        border: 1px solid #FECDD3;
        border-left: 5px solid #E11D48;
    }
    .verdict-ham {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 5px solid #059669;
    }
    .verdict-title-spam { font-size: 1.25rem; font-weight: 800; color: #9F1239; margin: 0; }
    .verdict-title-ham  { font-size: 1.25rem; font-weight: 800; color: #065F46; margin: 0; }
    .verdict-desc { font-size: 0.88rem; color: #475569; margin: 0.25rem 0 0 0; }

    .verdict-score {
        background: #FFFFFF;
        padding: 0.45rem 1rem;
        border-radius: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        font-size: 1.15rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .score-txt-spam { color: #BE123C; }
    .score-txt-ham  { color: #047857; }

    /* Thanh đo xác suất kép mini */
    .prob-metric-bar {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.8rem;
    }
    .prob-metric-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.35rem;
    }
    .prob-metric-name { font-size: 0.82rem; font-weight: 600; color: #475569; }
    .prob-metric-val  { font-size: 1.15rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #0F172A; }
    .prob-track {
        width: 100%;
        height: 7px;
        background: #E2E8F0;
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-fill-spam { height: 100%; background: #F43F5E; border-radius: 999px; }
    .prob-fill-ham  { height: 100%; background: #10B981; border-radius: 999px; }

    /* KPI Grid Cards */
    .kpi-stat-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .kpi-stat-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    .kpi-stat-value {
        font-size: 1.45rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #0F172A;
    }
    .kpi-accent-val { color: #4F46E5; }

    /* Textarea & Button Styling */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid #CBD5E1 !important;
        font-size: 0.92rem !important;
        padding: 0.85rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12) !important;
    }
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.15s ease !important;
    }

    /* Subtabs inside Right Column */
    .right-tabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 4px;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .right-tabs [data-baseweb="tab"] {
        border-radius: 7px;
        padding: 0.45rem 1rem;
        font-weight: 600;
        font-size: 0.86rem;
        color: #475569;
        border: none !important;
    }
    .right-tabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #4F46E5 !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
    }
    .right-tabs [data-baseweb="tab-highlight"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== ĐƯỜNG DẪN & DỮ LIỆU MẪU ====================
MODEL_PATH = os.path.join('models', 'model_vi.joblib')
SPLIT_DIR = os.path.join('data', 'vi')

PRESET_SPAM_SAMPLES = {
    '🚨 Trúng thưởng': "Chúc mừng quý khách đã trúng thưởng 50.000.000 VNĐ từ chương trình tri ân khách hàng! Vui lòng nhấn vào đường link ngay hôm nay để nhận quà miễn phí.",
    '🚨 Ngân hàng': "KHẨN CẤP: Tài khoản ngân hàng của bạn đang bị tạm khóa do giao dịch bất thường. Truy cập ngay đường dẫn xác minh mã OTP trong 24h để mở khóa tài khoản.",
    '🚨 Tuyển dụng': "Tuyển cộng tác viên online làm việc tại nhà, xử lý đơn hàng Shopee và Lazada. Thu nhập 500k - 1.5 triệu/ngày, nhận tiền ngay trong ngày, không cọc vốn. Nhắn Zalo 0987xxx để nhận việc ngay.",
    '🚨 Vay tín dụng': "Ngân hàng hỗ trợ vay tiêu dùng giải ngân từ 20 đến 100 triệu, lãi suất 0%, không thẩm định người thân, chấp nhận nợ xấu, nhận tiền sau 15 phút. Nhấn vào link vaytiennhanh.vn để đăng ký.",
    '🚨 Giả mạo Shipper': "GiaoHangTietKiem: Đơn hàng #VN89214 của bạn không thể phát do sai địa chỉ nhận hàng. Vui lòng truy cập ghtk-xacminh.com để cập nhật lại thông tin trong 12 giờ tránh bị hoàn đơn.",
}

PRESET_HAM_SAMPLES = {
    '✅ Họp nhóm': "Chào các bạn, sáng thứ 5 tuần này lúc 9h nhóm mình họp tại phòng B203 để rà soát tiến độ đồ án cuối kỳ nhé. Nhớ mang theo slide báo cáo.",
    '✅ Học tập': "Thưa thầy, em gửi thầy file báo cáo và mã nguồn bài tập tuần 6 ạ. Em cảm ơn thầy đã xem bài.",
    '✅ Công việc': "Kính gửi phòng ban liên quan, tôi xin gửi biên bản cuộc họp tổng kết tuần vừa qua cùng danh sách công việc cần hoàn thành trước thứ Sáu. Trân trọng.",
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


# ==================== CÁC HÀM VẼ BIỂU ĐỒ PLOTLY CHUYÊN NGHIỆP ====================
def create_plotly_metrics_chart(test_metrics: dict):
    """Vẽ biểu đồ cột tương tác Plotly cho 4 chỉ số hiệu năng."""
    names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    vals = [
        test_metrics['accuracy'] * 100,
        test_metrics['precision'] * 100,
        test_metrics['recall'] * 100,
        test_metrics['f1'] * 100,
    ]
    colors = ['#4F46E5', '#2563EB', '#0D9488', '#059669']

    fig = go.Figure(go.Bar(
        x=names,
        y=vals,
        text=[f"<b>{v:.2f}%</b>" for v in vals],
        textposition='outside',
        marker=dict(color=colors, cornerradius=5),
        hovertemplate="Chỉ số: <b>%{x}</b><br>Tỷ lệ: <b>%{y:.2f}%</b><extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="<b>4 Chỉ Số Hiệu Năng Mô Hình (Test Set)</b>", font=dict(size=13.5, color="#0F172A")),
        yaxis=dict(range=[0, 118], title="Tỷ lệ (%)", gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
        xaxis=dict(tickfont=dict(size=11, color="#334155")),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=35, r=15, t=45, b=35),
        height=320
    )
    return fig


def create_plotly_confusion_matrix(cm):
    """Vẽ Heatmap tương tác Plotly cho Ma trận nhầm lẫn bằng px.imshow."""
    labels_x = ['Dự đoán Hợp lệ (0)', 'Dự đoán Spam (1)']
    labels_y = ['Thực tế Hợp lệ (0)', 'Thực tế Spam (1)']

    annotations = [
        [f"<b>TN (Đúng Hợp lệ)</b><br>{cm[0, 0]:,} mẫu", f"<b>FP (Nhầm Spam)</b><br>{cm[0, 1]:,} mẫu"],
        [f"<b>FN (Sót Spam)</b><br>{cm[1, 0]:,} mẫu", f"<b>TP (Bắt đúng Spam)</b><br>{cm[1, 1]:,} mẫu"]
    ]

    fig = px.imshow(
        cm,
        x=labels_x,
        y=labels_y,
        color_continuous_scale=[[0, '#F5F3FF'], [0.5, '#C4B5FD'], [1, '#4F46E5']],
        labels=dict(x='Dự đoán', y='Thực tế', color='Số mẫu')
    )
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=labels_x[j],
                y=labels_y[i],
                text=annotations[i][j],
                showarrow=False,
                font=dict(size=12, color="#0F172A" if cm[i, j] < cm.max() * 0.6 else "#FFFFFF")
            )
    fig.update_layout(
        title=dict(text="<b>Ma Trận Nhầm Lẫn (Confusion Matrix Heatmap)</b>", font=dict(size=13.5, color="#0F172A")),
        xaxis=dict(tickfont=dict(size=11, color="#334155")),
        yaxis=dict(tickfont=dict(size=11, color="#334155")),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=45, r=30, t=50, b=45),
        height=320,
        coloraxis_showscale=False
    )
    return fig


def create_plotly_split_chart(X_train, X_val, X_test, y_train, y_val, y_test):
    """Vẽ biểu đồ phân bổ số lượng mẫu Hợp lệ vs Spam trên 3 tập dữ liệu."""
    splits = ['Train (70%)', 'Val (15%)', 'Test (15%)']
    ham_counts = [int((y_train == 0).sum()), int((y_val == 0).sum()), int((y_test == 0).sum())]
    spam_counts = [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Email Hợp lệ',
        x=splits,
        y=ham_counts,
        marker=dict(color='#10B981', cornerradius=5),
        text=[f"{c:,}" for c in ham_counts],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Hợp lệ: <b>%{y:,}</b> mẫu<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        name='Spam / Thư rác',
        x=splits,
        y=spam_counts,
        marker=dict(color='#F43F5E', cornerradius=5),
        text=[f"{c:,}" for c in spam_counts],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Spam: <b>%{y:,}</b> mẫu<extra></extra>"
    ))
    fig.update_layout(
        barmode='group',
        title=dict(text="<b>Phân Bổ Mẫu Dữ Liệu (Stratified Split)</b>", font=dict(size=13.5, color="#0F172A")),
        yaxis=dict(title="Số mẫu", gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
        xaxis=dict(tickfont=dict(size=11, color="#334155")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=35, r=15, t=55, b=35),
        height=320
    )
    return fig


def create_plotly_top_keywords(model, top_n=7):
    """Vẽ biểu đồ Top từ khóa đặc trưng nhất cho Spam và Hợp lệ."""
    vectorizer = model.named_steps['tfidf']
    nb = model.named_steps['nb']
    fn = np.array(vectorizer.get_feature_names_out())
    diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]

    spam_idx = np.argsort(diff)[-top_n:][::-1]
    ham_idx = np.argsort(diff)[:top_n]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"<b>Top {top_n} Từ Khóa SPAM</b>", f"<b>Top {top_n} Từ Khóa HỢP LỆ</b>"),
        horizontal_spacing=0.18
    )

    fig.add_trace(
        go.Bar(
            y=fn[spam_idx][::-1],
            x=diff[spam_idx][::-1],
            orientation='h',
            marker=dict(color='#F43F5E', cornerradius=4),
            text=[f"+{v:.2f}" for v in diff[spam_idx][::-1]],
            textposition='outside',
            hovertemplate="Từ khóa: <b>%{y}</b><br>Trọng số: +%{x:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            y=fn[ham_idx][::-1],
            x=(-diff[ham_idx])[::-1],
            orientation='h',
            marker=dict(color='#10B981', cornerradius=4),
            text=[f"{v:.2f}" for v in (-diff[ham_idx])[::-1]],
            textposition='outside',
            hovertemplate="Từ khóa: <b>%{y}</b><br>Trọng số: -%{x:.2f}<extra></extra>"
        ),
        row=1, col=2
    )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=55, r=35, t=45, b=35),
        height=320
    )
    fig.update_xaxes(title_text="Mức độ đặc trưng (Log-Odds)", gridcolor="#F1F5F9", tickfont=dict(color="#64748B", size=9.5))
    fig.update_yaxes(tickfont=dict(size=10.5, color="#1E293B", family="sans-serif"))
    return fig


# ==================== NẠP DỮ LIỆU & MÔ HÌNH ====================
model = get_model()
X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
total_samples = len(X_train) + len(X_val) + len(X_test)
val_metrics, test_metrics = get_evaluation_metrics(model, X_val, y_val, X_test, y_test)

# ==================== TOP NAVIGATION BAR ====================
st.markdown(
    f"""
    <div class="top-navbar">
        <div class="brand-group">
            <h1 class="brand-logo">🛡️ SpamGuard AI</h1>
            <span class="brand-badge">Bản Demo Tiếng Việt</span>
        </div>
        <div class="status-indicator">
            <span class="status-dot"></span>
            <span>Mô hình Sẵn sàng • TF-IDF + Naive Bayes • <b>{total_samples:,}</b> mẫu</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== BỐ CỤC 2 CỘT CÂN XỨNG (SPLIT-SCREEN WORKSPACE) ====================
col_workspace, col_analytics = st.columns([1.05, 1.15], gap="large")

# ==================== CỘT TRÁI: KHUNG ĐIỀU KHIỂN & PHÂN LOẠI EMAIL ====================
with col_workspace:
    state_key = "email_input_vi"
    if state_key not in st.session_state:
        st.session_state[state_key] = ""

    def set_sample_text(text: str):
        st.session_state[state_key] = text

    def clear_text_area():
        st.session_state[state_key] = ""

    st.markdown('<div class="section-label">⚡ Mẫu Thử Nhanh (Click để điền & test)</div>', unsafe_allow_html=True)

    # Dải nút chip gọn gàng phân nhóm rõ rệt
    st.caption("🔴 Mẫu Thư Rác / Lừa Đảo:")
    c_spam = st.columns(len(PRESET_SPAM_SAMPLES))
    for col, (label, text_val) in zip(c_spam, PRESET_SPAM_SAMPLES.items()):
        with col:
            st.button(
                label,
                on_click=set_sample_text,
                args=(text_val,),
                key=f"btn_sp_{label}",
                use_container_width=True,
            )

    st.caption("🟢 Mẫu Thư Hợp Lệ:")
    c_ham = st.columns(len(PRESET_HAM_SAMPLES))
    for col, (label, text_val) in zip(c_ham, PRESET_HAM_SAMPLES.items()):
        with col:
            st.button(
                label,
                on_click=set_sample_text,
                args=(text_val,),
                key=f"btn_hm_{label}",
                use_container_width=True,
            )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">✉️ Khung Nhập Nội Dung Email</div>', unsafe_allow_html=True)

    email_text = st.text_area(
        label="Nội dung Email cần kiểm tra",
        key=state_key,
        height=140,
        placeholder="Dán nội dung email hoặc tin nhắn tiếng Việt vào đây để hệ thống phân tích...",
        label_visibility="collapsed"
    )

    btn_col1, btn_col2, _ = st.columns([1.6, 1.2, 2.5])
    with btn_col1:
        predict_clicked = st.button("🚀 Phân loại ngay", type="primary", use_container_width=True)
    with btn_col2:
        st.button("🧹 Xóa nội dung", on_click=clear_text_area, use_container_width=True)

    # Hiển thị kết quả dự đoán
    if predict_clicked:
        if not email_text.strip():
            st.warning("Vui lòng nhập nội dung email hoặc bấm chọn một mẫu thử nhanh ở trên.")
        else:
            result = predict_single(model, email_text)
            prob_spam = result['probabilities']['Spam'] * 100
            prob_ham = result['probabilities']['Ham'] * 100
            is_spam = result['prediction'] == 1

            # Thẻ kết luận sắc nét
            if is_spam:
                st.markdown(
                    f"""
                    <div class="verdict-box verdict-spam">
                        <div>
                            <p class="verdict-title-spam">🚨 PHÁT HIỆN SPAM / THƯ RÁC</p>
                            <p class="verdict-desc">Có dấu hiệu lừa đảo, giả mạo hoặc quảng cáo độc hại (Nhãn: <b>1</b>).</p>
                        </div>
                        <div class="verdict-score score-txt-spam">{result['confidence']*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-box verdict-ham">
                        <div>
                            <p class="verdict-title-ham">✅ EMAIL HỢP LỆ / AN TOÀN (HAM)</p>
                            <p class="verdict-desc">Nội dung thư thông thường, an toàn không phát hiện bất thường (Nhãn: <b>0</b>).</p>
                        </div>
                        <div class="verdict-score score-txt-ham">{result['confidence']*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 2 Thanh đo xác suất mini mượt mà
            p_c1, p_c2 = st.columns(2)
            with p_c1:
                st.markdown(
                    f"""
                    <div class="prob-metric-bar">
                        <div class="prob-metric-header">
                            <span class="prob-metric-name">🚨 Xác suất SPAM</span>
                            <span class="prob-metric-val">{prob_spam:.1f}%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-fill-spam" style="width: {min(max(prob_spam, 0), 100):.1f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with p_c2:
                st.markdown(
                    f"""
                    <div class="prob-metric-bar">
                        <div class="prob-metric-header">
                            <span class="prob-metric-name">✅ Xác suất HỢP LỆ</span>
                            <span class="prob-metric-val">{prob_ham:.1f}%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-fill-ham" style="width: {min(max(prob_ham, 0), 100):.1f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Bảng mức độ tác động từ khóa Native Dataframe
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">🔎 Trọng số Tác Động Của Từ Khóa Trong Email</div>', unsafe_allow_html=True)
            keywords = explain_prediction(model, email_text, top_k=8)
            if not keywords:
                st.caption("Không tìm thấy từ khóa đặc trưng trùng khớp trong từ điển TF-IDF.")
            else:
                kw_spam = [k for k in keywords if k['tendency'] == 'Spam']
                kw_ham = [k for k in keywords if k['tendency'] == 'Ham']

                kw_col1, kw_col2 = st.columns(2)
                with kw_col1:
                    st.caption("🔴 Kéo về SPAM (+)")
                    if not kw_spam:
                        st.caption("Không có từ nghi vấn nổi bật.")
                    else:
                        df_sp = pd.DataFrame([
                            {'Từ khóa': k['word'], 'Điểm số': f"+{k['score']:.2f}", 'Tác động': min(max(abs(k['score']), 0.05), 1.0)}
                            for k in kw_spam
                        ])
                        st.dataframe(
                            df_sp,
                            column_config={
                                "Tác động": st.column_config.ProgressColumn("Tác động", format="%.2f", min_value=0, max_value=1.0),
                            },
                            use_container_width=True,
                            hide_index=True
                        )

                with kw_col2:
                    st.caption("🟢 Kéo về HỢP LỆ (-)")
                    if not kw_ham:
                        st.caption("Không có từ an toàn nổi bật.")
                    else:
                        df_hm = pd.DataFrame([
                            {'Từ khóa': k['word'], 'Điểm số': f"{k['score']:.2f}", 'Tác động': min(max(abs(k['score']), 0.05), 1.0)}
                            for k in kw_ham
                        ])
                        st.dataframe(
                            df_hm,
                            column_config={
                                "Tác động": st.column_config.ProgressColumn("Tác động", format="%.2f", min_value=0, max_value=1.0),
                            },
                            use_container_width=True,
                            hide_index=True
                        )


# ==================== CỘT PHẢI: BẢNG ĐIỀU KHIỂN & BIỂU ĐỒ PLOTLY ====================
with col_analytics:
    st.markdown('<div class="section-label">📊 Hiệu Năng Mô Hình Độc Lập (Test Metrics)</div>', unsafe_allow_html=True)

    # Hàng 4 thẻ chỉ số KPI tinh xảo
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-stat-box"><div class="kpi-stat-label">Accuracy</div><div class="kpi-stat-value">{test_metrics["accuracy"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-stat-box"><div class="kpi-stat-label">Precision</div><div class="kpi-stat-value">{test_metrics["precision"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-stat-box"><div class="kpi-stat-label">Recall</div><div class="kpi-stat-value">{test_metrics["recall"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-stat-box"><div class="kpi-stat-label">F1-Score</div><div class="kpi-stat-value kpi-accent-val">{test_metrics["f1"]*100:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    # Sub-tabs thông minh vừa vặn tầm mắt
    st.markdown('<div class="right-tabs">', unsafe_allow_html=True)
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "📐 Ma Trận & Hiệu Năng",
        "🔬 Từ Khóa & Phân Bổ",
        "📋 Chi Tiết Đánh Giá"
    ])

    with sub_tab1:
        fig_cm = create_plotly_confusion_matrix(test_metrics['confusion_matrix'])
        st.plotly_chart(fig_cm, use_container_width=True)

        fig_bar = create_plotly_metrics_chart(test_metrics)
        st.plotly_chart(fig_bar, use_container_width=True)

    with sub_tab2:
        fig_kw = create_plotly_top_keywords(model, top_n=7)
        st.plotly_chart(fig_kw, use_container_width=True)

        fig_split = create_plotly_split_chart(X_train, X_val, X_test, y_train, y_val, y_test)
        st.plotly_chart(fig_split, use_container_width=True)

    with sub_tab3:
        st.markdown("##### Báo Cáo Đánh Giá Chi Tiết (Classification Report)")
        st.code(test_metrics['classification_report'], language="text")

        st.markdown("##### Bảng Thống Kê Phân Bổ 3 Tập Dữ Liệu")
        split_df = pd.DataFrame([
            {'Tập dữ liệu': 'Huấn luyện (Train)', 'Tỷ lệ': '70%', 'Số mẫu': f"{len(X_train):,}", 'Hợp lệ': int((y_train == 0).sum()), 'Spam': int(y_train.sum())},
            {'Tập dữ liệu': 'Kiểm định (Val)', 'Tỷ lệ': '15%', 'Số mẫu': f"{len(X_val):,}", 'Hợp lệ': int((y_val == 0).sum()), 'Spam': int(y_val.sum())},
            {'Tập dữ liệu': 'Kiểm thử (Test)', 'Tỷ lệ': '15%', 'Số mẫu': f"{len(X_test):,}", 'Hợp lệ': int((y_test == 0).sum()), 'Spam': int(y_test.sum())},
        ])
        st.dataframe(split_df, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)
