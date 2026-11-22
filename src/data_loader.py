"""
Module: src/data_loader.py
Chức năng:
  - Tự động tải và nạp bộ dữ liệu tiếng Việt (Hugging Face / Cache cục bộ).
  - Phân chia tập dữ liệu theo đúng tỷ lệ chuẩn 70% Train, 15% Validation, 15% Test
    với kỹ thuật Stratified Sampling để bảo toàn tỷ lệ nhãn.
Dành cho: Sinh viên học tập môn Nhập môn Học máy & Khai phá dữ liệu (Bản Demo Tiếng Việt).
"""

import io
import os
import ssl
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

# Nguồn dữ liệu spam tiếng Việt
VN_DATA_SOURCES = [
    (
        "https://raw.githubusercontent.com/tabbyneko2k4/vietnamese-spam-detection/main/data/vietnamese_spam_dataset.csv",
        "text",
        "label",
    ),
    (
        "https://raw.githubusercontent.com/HuynhPhong385/Vietnamese-SMS-Spam-Detection/main/data/dataset/dataset.csv",
        "texts_vi",
        "labels",
    ),
]
VN_LOCAL_PATH = os.path.join("data", "vietnamese_spam.csv")


def load_vietnamese_data(
    local_path: str = VN_LOCAL_PATH,
    sources: list[tuple[str, str, str]] = None
) -> pd.DataFrame:
    """
    Nạp dữ liệu email/tin nhắn spam tiếng Việt từ cache cục bộ; nếu chưa có thì tự động
    tải và hợp nhất từ các bộ dữ liệu mã nguồn mở tiếng Việt (chứa đầy đủ email/tin nhắn
    giao tiếp tự nhiên và thư rác tiếng Việt).
    
    Args:
        local_path (str): Đường dẫn lưu file cục bộ (mặc định 'data/vietnamese_spam.csv').
        sources (list): Danh sách (url, cột_văn_bản, cột_nhãn) để tải khi chưa có cache.
        
    Returns:
        pd.DataFrame: Bảng dữ liệu sạch chứa 2 cột ['text', 'label'] (0: Ham, 1: Spam).
    """
    if sources is None:
        sources = VN_DATA_SOURCES

    if not os.path.exists(local_path):
        parent_dir = os.path.dirname(local_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        ssl_context = ssl._create_unverified_context()
        frames = []

        for url, text_col, label_col in sources:
            print(f"[INFO] Đang tải dữ liệu tiếng Việt từ:\n       {url}")
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(request, context=ssl_context, timeout=30) as response:
                content = response.read().decode('utf-8')

            raw_df = pd.read_csv(io.StringIO(content))
            texts = raw_df[text_col].fillna('').astype(str).str.strip()
            labels_raw = raw_df[label_col].astype(str).str.strip().str.lower()
            labels = labels_raw.isin(['spam', '1']).astype(int)

            frames.append(pd.DataFrame({'text': texts, 'label': labels}))

        merged_df = pd.concat(frames, ignore_index=True)
        merged_df = merged_df[merged_df['text'].str.len() > 0]
        merged_df = merged_df.drop_duplicates(subset=['text']).reset_index(drop=True)
        merged_df.to_csv(local_path, index=False, encoding='utf-8-sig')
        print(f"[INFO] Đã lưu {len(merged_df):,} mẫu tiếng Việt sạch tại: {local_path}")

    df = pd.read_csv(local_path)
    raw_text = df['text'].fillna('').astype(str).str.strip()
    valid_mask = raw_text.str.len() > 0

    clean_df = pd.DataFrame({
        'text': raw_text[valid_mask].values,
        'label': df.loc[valid_mask, 'label'].astype(int).values
    }).reset_index(drop=True)

    return clean_df


def split_dataset(
    df: pd.DataFrame,
    text_col: str = 'text',
    label_col: str = 'label',
    random_state: int = 42
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Phân chia tập dữ liệu thành 3 tập độc lập theo tỷ lệ chuẩn:
      - 70% tập Huấn luyện (Train)
      - 15% tập Kiểm định (Validation)
      - 15% tập Kiểm thử (Test)
      
    Sử dụng Stratified Split 2 bước để bảo đảm tỷ lệ nhãn (Spam/Ham) đồng nhất trên cả 3 tập.
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        text_col (str): Tên cột chứa văn bản (mặc định 'text').
        label_col (str): Tên cột chứa nhãn số học (mặc định 'label').
        random_state (int): Hạt giống ngẫu nhiên giúp tái lập kết quả phân chia (mặc định 42).
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df[text_col]
    y = df[label_col]

    # Bước 1: Tách 70% Train và 30% phần còn lại (Temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y
    )

    # Bước 2: Tách 30% Temp thành 15% Validation và 15% Test (tỷ lệ 50-50 trên Temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(
    X_train: pd.Series,
    X_val: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    output_dir: str
) -> dict[str, str]:
    """
    Lưu 3 tập dữ liệu Train, Validation, Test thành 3 file CSV riêng biệt
    (train.csv, val.csv, test.csv) để dễ kiểm tra và tái sử dụng.
    """
    os.makedirs(output_dir, exist_ok=True)

    splits = {
        'train': pd.DataFrame({'text': X_train.values, 'label': y_train.values}),
        'val': pd.DataFrame({'text': X_val.values, 'label': y_val.values}),
        'test': pd.DataFrame({'text': X_test.values, 'label': y_test.values}),
    }

    paths = {}
    for name, split_df in splits.items():
        file_path = os.path.join(output_dir, f"{name}.csv")
        split_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        paths[name] = file_path

    return paths


def load_splits(
    split_dir: str
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Đọc lại 3 file train.csv, val.csv, test.csv đã lưu từ thư mục split_dir.
    """
    train_df = pd.read_csv(os.path.join(split_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(split_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(split_dir, 'test.csv'))

    return (
        train_df['text'].fillna('').astype(str),
        val_df['text'].fillna('').astype(str),
        test_df['text'].fillna('').astype(str),
        train_df['label'].astype(int),
        val_df['label'].astype(int),
        test_df['label'].astype(int),
    )
