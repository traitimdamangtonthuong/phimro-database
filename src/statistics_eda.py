"""Phần II: Thống kê mô tả và tìm phim có giá trị cực trị."""

import numpy as np
import pandas as pd

# Các cột số có ý nghĩa phân tích
STAT_COLUMNS = ["Duration", "Rating", "Budget", "Revenue"]

def calculate_basic_stats(df):
    """Tính trung bình, trung vị và độ lệch chuẩn của các cột số.

    Tham số:
        df (pandas.DataFrame): Dữ liệu phim đã làm sạch.

    Trả về khi hoàn thiện:
        pandas.DataFrame: Mỗi hàng ứng với một cột số; các cột kết quả
        là Mean, Median và Std; chỉ mục mang tên column.

    Công việc cần triển khai:
        - Chọn cột số có ý nghĩa phân tích, không dùng mã định danh phim.
        - Tính Mean, Median và Std mẫu (ddof=1), bỏ qua giá trị thiếu.
        - Giữ giá trị thiếu nếu không đủ quan sát để tính thống kê.
        - Xử lý đầu vào rỗng hoặc không có cột số một cách nhất quán.
        - Trả về bảng; tầng gọi lưu data/results2.csv với index=True
          để giữ tên các cột được thống kê.
    """
    cols = [c for c in STAT_COLUMNS if df is not None and c in df.columns]
    
    # Đầu vào rỗng hoặc không có cột số -> bảng rỗng cùng cấu trúc
    if df is None ỏ df.empty or not cols:
        empty = pd.DataFrame(columns = ["Mean", "Median", "Std"])
        empty.index.name = "column"
        return empty
    # Ép kiểu số trên bản sao: "N/a" -> NaN, inf -> NaN
    numeric = df[cols].apply(pd.to_numeric, erors = "coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    result = pd.DataFrame({
        "Mean": numeric.mean(),
        "Median": numeric.median(),
        "Std": numeric.syd(ddof = 1), # NaN nếu chỉ có 1 quan sát
    })
    result.index.name = "column"
    return result
    
                        
def get_top_bottom_movies(df, column, top_n=3):
    """Tìm tất cả phim có giá trị lớn nhất và nhỏ nhất trong một cột.

    Tham số:
        df (pandas.DataFrame): Dữ liệu phim đã làm sạch.
        column (str): Tên cột số cần tìm cực trị.
        top_n (int): Số phim lấy ở mỗi đầu bảng, mặc định là 3.

    Trả về khi hoàn thiện:
        tuple[pandas.DataFrame, pandas.DataFrame]: Hai bảng gồm top_n phim
        theo thứ tự (điểm cao nhất, điểm thấp nhất), có cột Title và column.

    Công việc cần triển khai:
        - Kiểm tra cột tồn tại và có kiểu số; bỏ qua giá trị thiếu.
        - Dùng .nlargest(top_n, column) và .nsmallest(top_n, column).
        - Nếu không có giá trị hợp lệ, trả về hai bảng rỗng cùng cấu trúc.
        - Không sửa đổi DataFrame đầu vào.
    """
    if "Title" not in df.columns or column not in df.columns:
        raise ValueError(f"Thiếu cột 'Title' hoặc '{column}' trong dữ liệu.")
    values = pd.to_numeric(df[column], errors = "coerce")
    values = values.replace([np.inf, -np.inf], np.nan)

    data = pd.DataFrame({"Title": df["Title"], column: values})
    data = data.dropna(subset = [column]) # bỏ giá trị thiếu
    if data.empty: # không có giá trị hợp lệ
        empty = pd.DataFrame(columns = ["Title", column])
        return empty, empty.copy()
    top_movies = data.nlargest(top_n, column)
    bottom_movies = data.nsmallest(Top_n, column)
    return top_movies, bottom_movies
