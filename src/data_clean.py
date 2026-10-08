import pandas as pd
import numpy as np
import json
import re

# 1. Đọc file resultsRaw_completed.csv
print("Đang đọc file resultsRaw_completed.csv...")
df = pd.read_csv('resultsRaw_completed.csv')

# 2. Đổi tên cột về chuẩn 9 thuộc tính BTL
rename_dict = {
    'Release_Year': 'Year',
    'IMDb_Rating': 'Rating',
    'Budget_USD': 'Budget',
    'Revenue_USD': 'Revenue',
    'Duration_min': 'Duration',
    'Actors': 'Cast'
}
df = df.rename(columns=rename_dict)

# 3. Trích xuất tên thể loại từ chuỗi JSON cột Genre
def clean_genre(val):
    if pd.isna(val):
        return np.nan
    try:
        s = str(val).strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        s = s.replace('""', '"')
        data = json.loads(s)
        names = [item['name'] for item in data if 'name' in item]
        return ", ".join(names) if names else np.nan
    except Exception:
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', str(val))
        return ", ".join(names) if names else str(val)

if 'Genre' in df.columns:
    df['Genre'] = df['Genre'].apply(clean_genre)

# 4. Ép kiểu dữ liệu tiền tệ (Budget, Revenue) về dạng float
def clean_currency(val):
    if pd.isna(val) or str(val).strip() in ['', '0', 'N/a', 'nan', 'None']:
        return np.nan
    clean_val = str(val).replace('$', '').replace(',', '').strip()
    try:
        return float(clean_val)
    except ValueError:
        return np.nan

for col in ['Budget', 'Revenue']:
    if col in df.columns:
        df[col] = df[col].apply(clean_currency)

# 5. Lọc bỏ các bản ghi trùng lặp (trùng Title và Year)
if 'Title' in df.columns and 'Year' in df.columns:
    df = df.drop_duplicates(subset=['Title', 'Year'])

# 6. Sắp xếp danh sách phim theo Title (A-Z) và Year (giảm dần)
if 'Title' in df.columns and 'Year' in df.columns:
    df = df.sort_values(by=['Title', 'Year'], ascending=[True, False])
elif 'Title' in df.columns:
    df = df.sort_values(by='Title', ascending=True)

# 7. Giữ đúng 9 cột chuẩn
target_cols = ['Title', 'Year', 'Genre', 'Duration', 'Rating', 'Budget', 'Revenue', 'Director', 'Cast']
existing_cols = [c for c in target_cols if c in df.columns]
df = df[existing_cols]

# 8. Thay ô trống thành "N/a"
df = df.fillna('N/a')

# 9. Xuất file clean
df.to_csv('results.csv', index=False, encoding='utf-8-sig')

print("Đã tạo file results.csv sạch thành công!")
