# Bối cảnh và mục tiêu dự án

## 1. Bối cảnh

Dự án **phimro-database** là một ứng dụng web phân tích và dự đoán doanh thu phim được xây dựng bằng **Streamlit**. Ứng dụng sử dụng dữ liệu phim từ **TMDB (The Movie Database)** gồm khoảng 5.000 bộ phim với các thuộc tính: tên phim, năm phát hành, thể loại, thời lượng, điểm IMDb, kinh phí, doanh thu, đạo diễn và diễn viên.

Dự án được chia thành 4 phần theo nhóm thực hiện:

| Phasen | Nhóm | Nội dung chính |
|--------|------|----------------|
| **Phần I** | Tổ 1 | Thu thập, chuẩn hóa và làm sạch dữ liệu TMDB |
| **Phần II** | Tổ 2 | Thống kê mô tả, phân phối và biểu đồ trực quan |
| **Phần III** | Tổ 3 | Học máy không giám sát: phân cụm K-means, giảm chiều PCA, so sánh hai phim bằng radar |
| **Phần IV** | Tổ 4 | Học máy có giám sát: dự đoán doanh thu phòng vé bằng Random Forest |

Dữ liệu thô được thu thập từ TMDB và lưu tại `data/collected/resultsRaw_completed.csv` (không được commit lên Git do dung lượng lớn ~46 MB). Sau khi làm sạch, kết quả được lưu tại `data/cleaned/results.csv`.

## 2. Mục tiêu

### Mục tiêu chính

Xây dựng một ứng dụng phân tích dữ liệu phim hoàn chỉnh, cho phép người dùng:

1. **Làm sạch và chuẩn hóa dữ liệu** thô từ TMDB thành bảng sạch có cấu trúc, sẵn sàng phân tích.
2. **Thống kê mô tả dữ liệu**: trung bình, trung vị, độ lệch chuẩn cho các cột số; thống kê theo thể loại; histogram phân phối.
3. **Phân cụm phim** bằng K-means dựa trên kinh phí, doanh thu và điểm IMDB; trực quan hóa bản đồ cụm 2D bằng PCA; so sánh hai phim bằng biểu đồ radar.
4. **Dự đoán doanh thu phòng vé** cho một phim mới dựa trên kinh phí, điểm IMDb, thời lượng và thể loại.

### Mục tiêu cụ thể theo từng phần

| Phần | Mục tiêu chi tiết |
|------|-------------------|
| **Phần I** | Đọc dữ liệu thô, chuyển đổi tên cột thống nhất, xử lý giá trị 0 thành NaN (quy ước TMDB: 0 = chưa công bố), loại trùng lặp, xuất bảng sạch. |
| **Phần II** | Tính thống kê cơ bản (mean, median, std) cho toàn bộ dữ liệu và theo thể loại; tìm phim cao nhất/thấp nhất theo chỉ số chọn lọc; vẽ histogram phân phối. |
| **Phần III** | Phân cụm K-means trên các đặc trưng đã chuẩn hóa; giảm chiều bằng PCA để vẽ bản đồ 2D; so sánh hai phim trên radar (5 thuộc tính: Rating, Votes, Budget, Revenue, Duration). |
| **Phần IV** | Huấn luyện mô hình dự đoán doanh thu; đánh giá bằng MAE, RMSE, R²; so sánh với baseline đơn giản (trung bình / trung vị doanh thu); đưa ra dự đoán cho phim mới. |

## 3. Quyết định kỹ thuật chính

### 3.1 Xử lý giá trị thiếu và giá trị 0

Trong dữ liệu TMDB, giá trị **0** trên các cột kinh tế (`Budget_USD`, `Revenue_USD`) và điểm số (`IMDb_Rating`, `Duration_min`) không phải là giá trị thực mà là biểu thị **"chưa công bố / không có dữ liệu"**. Pipeline làm sạch (`src/data_pipeline.py`) quy ước:

- Giá trị 0 trong các cột số kinh tế và điểm số → chuyển thành `NaN`.
- Các hàng thiếu doanh thu mục tiêu (`Revenue_USD`) → bị loại bỏ trước khi huấn luyện mô hình (không được suy đoán).

### 3.2 Tên cột

Pipeline giữ nguyên tên cột dài từ dữ liệu thù (`Budget_USD`, `Revenue_USD`, `IMDb_Rating`, `Duration_min`, `Votes`) thay vì rút gọn ngắn, để các module sau (`statistics_eda`, `visualizer`, `models`) tra cứu nhất quán.

### 3.3 Mô hình dự đoán doanh thu (Phần IV)

- **Thuật toán**: `RandomForestRegressor` với `n_estimators=300`, `max_depth=15`, `min_samples_leaf=3`, `max_features="sqrt"`, `n_jobs=-1`.
- **Tiền xử lý số**: `SimpleImputer(strategy="median")` + `StandardScaler`.
- **Tiền xử lý phân loại**: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5)`.
- **Đặc trưng kỹ thuật**:
  - `log_Budget_USD`: `log1p(kinh phí)` — giảm sự lệch phải của phân phối kinh phí.
  - `Budget_x_Rating`: `kinh phí × điểm IMDb` — biểu diễn tương tác giữa năng lực tài chính và chất lượng phim.
- **Mục tiêu**: `Revenue_USD` trên thang USD gốc (không log-transform target, vì thử nghiệm cho thấy log1p(Revenue) làm R² giảm từ 0.53 xuống 0.23).
- **Chia tập**: 80% train / 20% test, `random_state=42`, chia trước khi fit tiền xử lý (tránh data leakage).

### 3.4 Baseline đơn giản (Phần IV)

Để đánh giá xem mô hình có học được gì hay chỉ chuẩn hóa sai lệch, dự án tính toán baseline:

- **Baseline trung bình**: dự đoán `mean(doanh thu huấn luyện)` cho mọi mẫu test.
- **Baseline trung vị**: dự đoán `median(doanh thu huấn luyện)` cho mọi mẫu test.

Kết quả so sánh được hiển thị trên giao diện phần IV.

### 3.5 Phân cụm K-means (Phần III)

- Đặc trưng: `Budget_USD`, `Revenue_USD`, `IMDb_Rating` (đã chuẩn hóa bằng `StandardScaler`).
- `KMeans(random_state=42, n_init=10)` để tái lập kết quả.
- Slider `n_clusters` động: tối đa `min(10, số mẫu hợp lệ)`, tránh lỗi `n_clusters > số mẫu`.

## 4. Kết quả đánh giá (trên dữ liệu TMDB ~4.800 bộ phim)

| Chỉ số | Giá trị |
|--------|---------|
| R² mô hình | **0.57** |
| MAE | **~$70 triệu USD** |
| RMSE | **~$118 triệu USD** |
| Baseline R² (trung bình) | ≈ 0.00 |
| Baseline R² (trung vị) | ≈ -0.14 |

Mô hình Random Forest tốt hơn đáng kể so với baseline đơn giản (dự đoán trung bình/trung vị), chứng tỏ các đặc trưng kinh phí, điểm IMDb, thời lượng và thể loại mang thông tin dự báo doanh thu.

**Lưu ý**: Dữ liệu thô hiện tại **không có cột `Votes`** (số lượt đánh giá), nên đặc trưng này chưa được kích hoạt trong mô hình. Nếu dữ liệu sau này bổ sung cột Votes, code đã sẵn sàng hỗ trợ (khai báo `log_Votes` và điều kiện hiển thị ô nhập trên UI).


## 5. Cấu trúc repository

```
phimro-database/
├── app.py                      # Giao diện Streamlit chính (4 phần)
├── README.md                   # Hướng dẫn chạy dự án
├── PROJECT_CONTEXT.md          # Bối cảnh và mục tiêu dự án (file này)
├── requirements.txt            # Thư viện Python cần thiết
├── .gitignore                  # Bỏ qua data/*.csv, __pycache__, .idea, v.v.
├── data/
│   ├── collected/              # Dữ liệu thô từ TMDB (không commit)
│   └── cleaned/                # Dữ liệu đã làm sạch (không commit)
├── src/
│   ├── collect_data.py         # Thu thập dữ liệu TMDB
│   ├── data_pipeline.py        # Pipeline làm sạch dữ liệu (single source of truth)
│   ├── data_clean.py           # Lớp tương thích ngược (import lại từ data_pipeline)
│   ├── models.py               # K-means, PCA, RandomForest, predict_revenue
│   ├── statistics_eda.py       # Thống kê mô tả, top/bottom phim
│   └── visualizer.py           # Biểu đồ histogram, radar chart
└── tests/
    └── test_data_pipeline.py   # 2 bài test kiểm tra pipeline làm sạch
```

## 6. Cách chạy dự án

```powershell
# 1. Cài đặt thư viện
python -m pip install -r requirements.txt

# 2. Thu thập dữ liệu thô (nếu chưa có)
python src\collect_data.py

# 3. Làm sạch dữ liệu
python -c "from src.data_pipeline import load_and_clean_data; load_and_clean_data('data/collected/resultsRaw_completed.csv')"

# 4. Chạy ứng dụng Streamlit
python -m streamlit run app.py
```

> Thư mục `data/` được bỏ qua khỏi Git. Sau khi clone repository, cần tạo dữ liệu thô và chạy pipeline trước khi dùng toàn bộ chức năng của ứng dụng.

## 7. Chạy kiểm tra tự động

```powershell
python -m pytest tests/ -v
```

Hiện có 2 bài test cho `src/data_pipeline.py`, đều pass.

---

*Tài liệu này được cập nhật lần cuối sau khi hoàn thành Part II, Part III, Part IV và sửa lỗi phân cụm động (n_clusters giới hạn theo số mẫu hợp lệ).*
