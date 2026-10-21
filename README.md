# 🎬 Movie Data Analysis & Prediction Web App

Ứng dụng Streamlit phân tích dữ liệu phim TMDB, trực quan hóa đặc trưng phim,
phân cụm K-means và dự đoán doanh thu phòng vé. Dự án sử dụng Python, Pandas,
Matplotlib và Scikit-learn.

## Trạng thái hiện tại

| Phần | Chức năng | Trạng thái |
| --- | --- | --- |
| I | Thu thập, chuẩn hóa và làm sạch dữ liệu TMDB | ✅ Hoàn thành |
| II | Thống kê mô tả, thống kê theo thể loại và histogram | ✅ Hoàn thành |
| III | K-means, PCA và so sánh hai phim bằng radar | ✅ Hoàn thành |
| IV | Huấn luyện và dự đoán doanh thu bằng Random Forest | ✅ Hoàn thành |
| UI | Giao diện Streamlit cho bốn phần | ✅ Hoàn thành |

## Chạy dự án

Yêu cầu Python 3.10+ được khuyến nghị.

```powershell
Set-Location "D:\PRJ\phimro-database"
python -m pip install -r requirements.txt
```

### 1. Tạo dữ liệu thô

```powershell
python src\collect_data.py
```

Lệnh này tạo các tệp trong `data/collected/`, gồm
`resultsRaw_completed.csv`. Nếu các tệp TMDB đã có sẵn và chỉ cần tạo lại bảng
thô, có thể chạy trong Python:

```python
from src.collect_data import main
main(download=False)
```

### 2. Làm sạch dữ liệu

`src/data_pipeline.py` là pipeline chính. Chạy lệnh sau từ thư mục gốc:

```powershell
python -c "from src.data_pipeline import load_and_clean_data; load_and_clean_data('data/collected/resultsRaw_completed.csv')"
```

Hoặc mở ứng dụng và chọn **Phần I → Chạy làm sạch dữ liệu**. Kết quả được ghi
vào `data/cleaned/results.csv`.

### 3. Chạy web

```powershell
python -m streamlit run app.py
```

> Thư mục `data/` được bỏ qua khỏi Git. Sau khi clone repository, cần tạo dữ
> liệu thô và chạy pipeline trước khi dùng toàn bộ chức năng của ứng dụng.

## Pipeline dữ liệu

```text
TMDB 5000 movies + credits
        │
        ▼
src/collect_data.py
        │  resultsRaw_completed.csv
        ▼
src/data_pipeline.py
        │  results.csv
        ▼
app.py / EDA / ML
```

`load_and_clean_data(raw_file_path)` thực hiện các bước sau:

1. Đọc dữ liệu thô và thống nhất tên cột.
2. Bóc tách `Genre` từ chuỗi JSON thành danh sách thể loại dễ đọc.
3. Chuyển các cột số sang kiểu số.
4. Coi `0`, chuỗi rỗng, `N/a`, `nan` và giá trị không hợp lệ là thiếu dữ liệu
   (`NaN`) đối với các chỉ số số học.
5. Loại bản ghi trùng theo cặp `Title` và `Year`, sau đó sắp xếp theo tên phim.
6. Xuất dữ liệu sạch, không thay `NaN` bằng chuỗi hiển thị như `N/a`.

### Vai trò của hai file làm sạch

| File | Vai trò |
| --- | --- |
| `src/data_pipeline.py` | Nguồn logic duy nhất của pipeline; chứa `load_and_clean_data()`. |
| `src/data_clean.py` | File cũ, nội dung hiện được vô hiệu hóa và không phải bước thực thi chính. Dùng `data_pipeline.py` cho code mới và tài liệu mới. |

## Schema dữ liệu sạch

Tệp `data/cleaned/results.csv` hiện có 10 cột. Dữ liệu mẫu đang có 4.803 bộ
phim; số dòng có thể thay đổi nếu nguồn dữ liệu được tạo lại.

| Cột | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `Title` | text | Tên phim |
| `Year` | number | Năm phát hành |
| `Genre` | text | Một hoặc nhiều thể loại, phân tách bằng dấu phẩy |
| `Votes` | number | Số lượt bình chọn trên TMDB (`vote_count`) |
| `Duration_min` | number | Thời lượng, phút |
| `IMDb_Rating` | number | Điểm TMDB (`vote_average`), thang 0–10; tên cột được giữ để tương thích mã nguồn |
| `Budget_USD` | number | Kinh phí, USD |
| `Revenue_USD` | number | Doanh thu, USD |
| `Director` | text | Đạo diễn |
| `Cast` | text | Danh sách diễn viên chính |

Giá trị `NaN` là dữ liệu chưa có hoặc không hợp lệ, không đồng nghĩa với giá
bằng 0. Các hàm EDA và ML bỏ qua hoặc điền khuyết các giá trị này theo ngữ
cảnh của từng phép tính.

## Chức năng ứng dụng

### Phần I — Tổng quan và làm sạch dữ liệu

- Chạy pipeline từ dữ liệu thô ngay trong giao diện.
- Hiển thị số bộ phim, tổng số cột và số cột số.
- Nạp lại dữ liệu sạch sau khi pipeline hoàn tất.

### Phần II — Thống kê và phân bố

- Tính trung bình, trung vị và độ lệch chuẩn cho `Duration_min`,
  `IMDb_Rating`, `Budget_USD`, `Revenue_USD`.
- Tạo bảng thống kê theo genre ở định dạng phẳng, ví dụ
  `Budget_USD_mean`, `Budget_USD_median`, `Budget_USD_std`.
- Xuất bảng tổng quan và bảng theo genre ra `data/results2.csv` và
  `data/results2_by_genre.csv`.
- Hiển thị Top/Bottom phim và histogram có đơn vị dễ đọc cho thời lượng,
  điểm, kinh phí và doanh thu.

### Phần III — Phân cụm và so sánh

- Phân cụm K-means theo `Budget_USD`, `Revenue_USD`, `IMDb_Rating` sau khi
  chuẩn hóa bằng `StandardScaler`.
- Hiển thị bản đồ PCA hai chiều và tỷ lệ phương sai được giải thích.
- So sánh hai phim qua biểu đồ radar với 5 thuộc tính: điểm TMDB, lượt đánh
  giá, kinh phí, doanh thu và thời lượng.
- Radar dùng chuẩn hóa Min-Max dựa trên toàn bộ tập dữ liệu. Thang 0–100 trên
  biểu đồ là vị trí tương đối; giá trị gốc có đơn vị được hiển thị bên cạnh.

### Phần IV — Dự đoán doanh thu

- Mô hình `RandomForestRegressor` được huấn luyện theo tỷ lệ train/test 80/20
  với `random_state=42`.
- Đặc trưng đầu vào: kinh phí, điểm TMDB, lượt đánh giá, thời lượng và genre.
- Có biến đổi `log_Budget_USD`, `log_Votes` và tương tác
  `Budget_x_Rating`.
- Tiền xử lý và mô hình được gói trong `sklearn.Pipeline` để tránh data
  leakage; giao diện hiển thị MAE, RMSE, R² và cho phép nhập phim mới để dự
  đoán.

## Cấu trúc dự án

```text
phimro-database/
├── app.py                         # Giao diện Streamlit
├── requirements.txt
├── data/
│   ├── collected/                 # Dữ liệu thô, không commit
│   └── cleaned/results.csv        # Dữ liệu sạch sinh bởi pipeline
└── src/
    ├── collect_data.py            # Tải và ghép dữ liệu TMDB
    ├── data_pipeline.py           # Pipeline làm sạch chính
    ├── data_clean.py              # File cũ, không dùng trong pipeline hiện tại
    ├── statistics_eda.py          # Thống kê mô tả và extreme values
    ├── visualizer.py              # Histogram và radar chart
    └── models.py                  # K-means, PCA và dự đoán doanh thu
```

## API chính

| Hàm | Mô tả |
| --- | --- |
| `load_and_clean_data(raw_file_path)` | Làm sạch dữ liệu và ghi `data/cleaned/results.csv`. |
| `calculate_basic_stats(df)` | Trả về `(overall, by_genre)` với thống kê tổng quan và theo thể loại. |
| `get_top_bottom_movies(df, column, top_n=3)` | Trả về hai bảng phim có giá trị cao nhất/thấp nhất. |
| `plot_histogram(df, column)` | Trả về đối tượng `matplotlib.figure.Figure`. |
| `plot_radar_chart(df, movie1_name, movie2_name)` | Trả về radar so sánh hai phim. |
| `run_kmeans_clustering(df, n_clusters)` | Trả về dữ liệu hợp lệ cùng nhãn `Cluster`. |
| `run_pca_reduction(df)` | Trả về hai tọa độ PCA và lưu explained variance trong `.attrs`. |
| `train_revenue_prediction_model(df)` | Trả về pipeline đã huấn luyện, kèm `.metrics_` và `.feature_columns_`. |
| `predict_revenue(model, X_raw)` | Dự đoán doanh thu bằng pipeline đã huấn luyện. |

## Quy ước kỹ thuật

- Không điền giá trị suy đoán cho `Revenue_USD` khi thiếu.
- Hàm biểu đồ trả về `fig`, không gọi `plt.show()`; giao diện tự hiển thị và
  đóng figure.
- K-means/PCA dùng `StandardScaler`; radar dùng Min-Max scaling trên toàn bộ
  dữ liệu tham chiếu.
- Mọi bước ngẫu nhiên quan trọng dùng `random_state=42` để dễ tái lập.
