"""
Module: app.py
Giao diện Web Streamlit Đồ án Phân loại Email Spam Tiếng Việt
(TF-IDF n-gram + Multinomial Naive Bayes).
Phiên bản Cao Cấp: Giao diện toàn màn hình, tích hợp Plotly Interactive Charts,
phân tích từ khóa với ProgressColumn thông minh, loại bỏ hoàn toàn lỗi hiển thị mã thô và đè chữ.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
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

# ==================== GIAO DIỆN HIỆN ĐẠI (CLEAN SLATE & INDIGO) ====================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }

    /* Ẩn hoàn toàn sidebar và chỉnh trong suốt header */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Căn lề thoáng đãng không bị che khuất tiêu đề */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1260px;
    }

    /* Header thương hiệu */
    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 60%, #4338CA 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 18px;
        color: #FFFFFF;
        margin-top: 0.5rem;
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
        line-height: 1.25;
    }
    .header-desc {
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
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

    /* Thẻ kết quả phân loại */
    .result-banner {
        border-radius: 16px;
        padding: 1.4rem 1.7rem;
        margin: 1.2rem 0;
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


# ==================== CÁC HÀM BIỂU ĐỒ PLOTLY TƯƠNG TÁC CAO CẤP ====================
def create_plotly_metrics_chart(test_metrics: dict):
    """Vẽ biểu đồ cột tương tác Plotly cho 4 chỉ số hiệu năng."""
    names = ['Accuracy (Độ chính xác)', 'Precision (Độ chuẩn xác)', 'Recall (Độ nhạy)', 'F1-Score']
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
        marker=dict(color=colors, corner_radius=6),
        hovertemplate="<b>%{x}</b><br>Tỷ lệ đạt được: <b>%{y:.2f}%</b><extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="<b>Chỉ Số Hiệu Năng Mô Hình (Tập Test Độc Lập)</b>", font=dict(size=14, color="#0F172A")),
        yaxis=dict(range=[0, 118], title="Tỷ lệ (%)", gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
        xaxis=dict(tickfont=dict(size=11, color="#334155")),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=50, b=40),
        height=350
    )
    return fig


def create_plotly_confusion_matrix(cm):
    """Vẽ Heatmap tương tác Plotly cho Ma trận nhầm lẫn."""
    labels_x = ['Dự đoán Hợp lệ (0)', 'Dự đoán Spam (1)']
    labels_y = ['Thực tế Spam (1)', 'Thực tế Hợp lệ (0)']

    # Sắp xếp ma trận theo thứ tự hiển thị trục y từ dưới lên
    z_values = [
        [int(cm[1, 0]), int(cm[1, 1])],
        [int(cm[0, 0]), int(cm[0, 1])]
    ]

    annotations = [
        [f"<b>FN (Sót Spam)</b><br>{cm[1, 0]:,} mẫu", f"<b>TP (Bắt đúng Spam)</b><br>{cm[1, 1]:,} mẫu"],
        [f"<b>TN (Đúng Hợp lệ)</b><br>{cm[0, 0]:,} mẫu", f"<b>FP (Báo nhầm Spam)</b><br>{cm[0, 1]:,} mẫu"]
    ]

    fig = ff.create_annotated_heatmap(
        z=z_values,
        x=labels_x,
        y=labels_y,
        annotation_text=annotations,
        colorscale=[[0, '#F5F3FF'], [0.5, '#C4B5FD'], [1, '#4F46E5']],
        showscale=False
    )
    fig.update_layout(
        title=dict(text="<b>Ma Trận Nhầm Lẫn (Confusion Matrix Heatmap)</b>", font=dict(size=14, color="#0F172A")),
        xaxis=dict(tickfont=dict(size=11, color="#334155")),
        yaxis=dict(tickfont=dict(size=11, color="#334155")),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=50, r=40, t=60, b=50),
        height=350
    )
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12, color="#0F172A")
    return fig


def create_plotly_split_chart(X_train, X_val, X_test, y_train, y_val, y_test):
    """Vẽ biểu đồ phân bổ số lượng mẫu Hợp lệ vs Spam trên 3 tập dữ liệu."""
    splits = ['Tập Train (70%)', 'Tập Val (15%)', 'Tập Test (15%)']
    ham_counts = [int((y_train == 0).sum()), int((y_val == 0).sum()), int((y_test == 0).sum())]
    spam_counts = [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Email Hợp lệ (Ham)',
        x=splits,
        y=ham_counts,
        marker=dict(color='#10B981', corner_radius=6),
        text=[f"{c:,}" for c in ham_counts],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Hợp lệ: <b>%{y:,}</b> mẫu<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        name='Spam / Thư rác',
        x=splits,
        y=spam_counts,
        marker=dict(color='#F43F5E', corner_radius=6),
        text=[f"{c:,}" for c in spam_counts],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Spam: <b>%{y:,}</b> mẫu<extra></extra>"
    ))
    fig.update_layout(
        barmode='group',
        title=dict(text="<b>Phân Bổ Dữ Liệu Train / Val / Test (Stratified Split)</b>", font=dict(size=14, color="#0F172A")),
        yaxis=dict(title="Số lượng mẫu", gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
        xaxis=dict(tickfont=dict(size=11, color="#334155")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=60, b=40),
        height=370
    )
    return fig


def create_plotly_top_keywords(model, top_n=8):
    """Vẽ biểu đồ Top từ khóa đặc trưng nhất cho Spam và Hợp lệ."""
    vectorizer = model.named_steps['tfidf']
    nb = model.named_steps['nb']
    fn = np.array(vectorizer.get_feature_names_out())
    diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]

    spam_idx = np.argsort(diff)[-top_n:][::-1]
    ham_idx = np.argsort(diff)[:top_n]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"<b>Top {top_n} Từ Khóa Kéo Về SPAM</b>", f"<b>Top {top_n} Từ Khóa Kéo Về HỢP LỆ</b>"),
        horizontal_spacing=0.15
    )

    # Top Spam
    fig.add_trace(
        go.Bar(
            y=fn[spam_idx][::-1],
            x=diff[spam_idx][::-1],
            orientation='h',
            marker=dict(color='#F43F5E', corner_radius=4),
            text=[f"+{v:.2f}" for v in diff[spam_idx][::-1]],
            textposition='outside',
            hovertemplate="Từ khóa: <b>%{y}</b><br>Trọng số: +%{x:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    # Top Ham
    fig.add_trace(
        go.Bar(
            y=fn[ham_idx][::-1],
            x=(-diff[ham_idx])[::-1],
            orientation='h',
            marker=dict(color='#10B981', corner_radius=4),
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
        margin=dict(l=60, r=40, t=50, b=40),
        height=370
    )
    fig.update_xaxes(title_text="Mức độ đặc trưng (Log-Odds)", gridcolor="#F1F5F9", tickfont=dict(color="#64748B"))
    fig.update_yaxes(tickfont=dict(size=11, color="#1E293B", family="sans-serif"))
    return fig


def render_keyword_table(keywords: list):
    """Hiển thị bảng trực quan hóa từ khóa bằng Native Dataframe có Progress Bar, chống 100% lỗi code thô."""
    if not keywords:
        st.info("Không tìm thấy từ khóa đặc trưng trùng khớp trong từ điển TF-IDF.")
        return

    spam_kw = [k for k in keywords if k['tendency'] == 'Spam']
    ham_kw = [k for k in keywords if k['tendency'] == 'Ham']

    col_spam, col_ham = st.columns(2)
    with col_spam:
        st.markdown("##### 🚨 Từ khóa nghi vấn (Kéo về SPAM)")
        if not spam_kw:
            st.caption("Không phát hiện từ khóa nghi vấn nổi bật.")
        else:
            df_spam = pd.DataFrame([
                {
                    'Từ khóa': k['word'],
                    'Điểm số': f"+{k['score']:.2f}",
                    'Mức độ tác động': min(max(abs(k['score']), 0.05), 1.0)
                }
                for k in spam_kw
            ])
            st.dataframe(
                df_spam,
                column_config={
                    "Mức độ tác động": st.column_config.ProgressColumn(
                        "Mức độ tác động",
                        help="Thanh đo tỷ lệ kéo email về nhãn Spam",
                        format="%.2f",
                        min_value=0,
                        max_value=1.0,
                    ),
                },
                use_container_width=True,
                hide_index=True
            )

    with col_ham:
        st.markdown("##### 🛡️ Từ khóa an toàn (Kéo về HỢP LỆ)")
        if not ham_kw:
            st.caption("Không phát hiện từ khóa an toàn nổi bật.")
        else:
            df_ham = pd.DataFrame([
                {
                    'Từ khóa': k['word'],
                    'Điểm số': f"{k['score']:.2f}",
                    'Mức độ tác động': min(max(abs(k['score']), 0.05), 1.0)
                }
                for k in ham_kw
            ])
            st.dataframe(
                df_ham,
                column_config={
                    "Mức độ tác động": st.column_config.ProgressColumn(
                        "Mức độ tác động",
                        help="Thanh đo tỷ lệ kéo email về nhãn Hợp lệ",
                        format="%.2f",
                        min_value=0,
                        max_value=1.0,
                    ),
                },
                use_container_width=True,
                hide_index=True
            )


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
    "📊 Báo cáo Thông số & Biểu đồ Trực quan (Plotly)",
])

# ==================== TAB 1: TRẢI NGHIỆM PHÂN LOẠI EMAIL ====================
with tab_demo:
    state_key = "email_input_vi"
    if state_key not in st.session_state:
        st.session_state[state_key] = ""

    # Callbacks cập nhật state an toàn không bao giờ văng lỗi Streamlit
    def set_sample_text(text: str):
        st.session_state[state_key] = text

    def clear_text_area():
        st.session_state[state_key] = ""

    st.markdown("##### ⚡ Chọn nhanh mẫu thử để trình chiếu demo:")
    sample_items = list(QUICK_SAMPLES.items())

    cols_row1 = st.columns(4)
    for col, (btn_label, sample_text) in zip(cols_row1, sample_items[:4]):
        with col:
            st.button(
                btn_label,
                on_click=set_sample_text,
                args=(sample_text,),
                key=f"btn_{btn_label}",
                use_container_width=True,
            )

    cols_row2 = st.columns(4)
    for col, (btn_label, sample_text) in zip(cols_row2, sample_items[4:]):
        with col:
            st.button(
                btn_label,
                on_click=set_sample_text,
                args=(sample_text,),
                key=f"btn_{btn_label}",
                use_container_width=True,
            )

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
        st.button("🧹 Xóa nội dung", on_click=clear_text_area, use_container_width=True)

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

            # Bảng trực quan hóa từ khóa bằng Native Dataframe có Progress Bar, không bao giờ bị in mã thô
            st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
            st.markdown("##### 🔎 Phân tích Từ khóa Ảnh hưởng đến Quyết định:")
            keywords = explain_prediction(model, email_text, top_k=8)
            render_keyword_table(keywords)


# ==================== TAB 2: BÁO CÁO THÔNG SỐ & BIỂU ĐỒ TRỰC QUAN (PLOTLY) ====================
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

    # Cụm biểu đồ 1: Cột 4 chỉ số Plotly + Heatmap Ma trận nhầm lẫn Plotly
    col_chart1, col_chart2 = st.columns([1.1, 1])
    with col_chart1:
        fig_bar = create_plotly_metrics_chart(test_metrics)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        fig_cm = create_plotly_confusion_matrix(test_metrics['confusion_matrix'])
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # Cụm biểu đồ 2: Phân bổ dữ liệu 3 tập Plotly + Top từ khóa đặc trưng Plotly
    col_split, col_kw_all = st.columns([1, 1.4])
    with col_split:
        fig_split = create_plotly_split_chart(X_train, X_val, X_test, y_train, y_val, y_test)
        st.plotly_chart(fig_split, use_container_width=True)

    with col_kw_all:
        fig_top_kw = create_plotly_top_keywords(model, top_n=8)
        st.plotly_chart(fig_top_kw, use_container_width=True)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown("##### 📋 Báo cáo Phân loại Chi tiết (Classification Report)")
    st.code(test_metrics['classification_report'], language="text")
