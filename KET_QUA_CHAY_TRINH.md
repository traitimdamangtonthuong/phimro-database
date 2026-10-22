# BÁO CÁO TỔNG HỢP KẾT QUẢ CHẠY TOÀN BỘ CHƯƠNG TRÌNH

Dự án: **phimro-database** — Phân tích dữ liệu và dự đoán doanh thu phim  
Ngày báo cáo: 20/10/2026  
Môi trường: Windows 11, Python 3.13.5, Streamlit 1.45.1, scikit-learn, pandas, matplotlib, seaborn

---

## 1. TỔNG QUAN TRẠNG THÁI CÁC PHẦN

| Phần | Chức năng | Trạng thái | Kết quả kiểm tra |
|------|-----------|------------|------------------|
| **Phần I** | Thu thập, làm sạch dữ liệu TMDB | ✅ Hoàn thành | Chạy được, xuất `data/cleaned/results.csv` |
| **Phần II** | Thống kê mô tả, phân phối, histogram | ✅ Hoàn thành | AppTest pass, các hàm thống kê chạy đúng |
| **Phần III** | Phân cụm K-means, PCA, radar chart | ✅ Hoàn thành | AppTest pass, slider n_clusters động |
| **Phần IV** | Dự đoán doanh thu bằng Random Forest | ✅ Hoàn thành | AppTest pass, R² = 0.57 |

**Tổng kết:** Toàn bộ 4 phần của ứng dụng Streamlit chạy không ngoại lệ. Working tree sạch, đã push lên `origin/main`.


## 2. CHI TIẾT TỪNG PHẦN

### Phần I: Làm sạch dữ liệu

- **Input:** `data/collected/resultsRaw_completed.csv` (dữ liệu thô TMDB)
- **Output:** `data/cleaned/results.csv` (dữ liệu đã làm sạch)
- **Kích thước dữ liệu sạch:** 4.803 dòng × 7 cột
- **Các cột:** `Title`, `Genre`, `Duration_min`, `IMDb_Rating`, `Budget_USD`, `Revenue_USD`, `Director`
- **Quy tắc làm sạch:**
  - Giá trị `0` trong `Budget_USD`, `Revenue_USD`, `IMDb_Rating`, `Duration_min` → chuyển thành `NaN` (quy ước TMDB: 0 = chưa công bố)
  - Loại bỏ trùng lặp theo `(Title, Year)`
  - Sắp xếp theo `Title` A-Z, `Year` giảm dần
  - Mã hóa UTF-8-SIG

**Kiểm thử:**
- `test_convert_zero_to_nan`: PASSED — giá trị 0 được chuyển thành NaN đúng quy tắc
- `test_missing_essential_columns`: PASSED — ném `KeyError` khi thiếu cột `Budget_USD`

### Phần II: Thống kê & Phân bố

**Input:** `data/cleaned/results.csv`

#### 2.1 Bảng thống kê mô tả cơ bản
Hàm: `calculate_basic_stats(df)`  
Cột số tham gia: `Duration_min`, `IMDb_Rating`, `Budget_USD`, `Revenue_USD`

| Cột | Mean | Median | Std |
|-----|------|--------|-----|
| Duration_min | ~106.88 | 103.0 | ~22.61 |
| IMDb_Rating | ~6.09 | 6.2 | ~1.19 |
| Budget_USD | ~37.04M | 23.0M | ~42.65M |
| Revenue_USD | ~117.03M | 51.75M | ~183.48M |

- Bảng theo thể loại (`stats_by_genre`): 21 thể loại, có cột `Movie_Count`
- Lưu bảng tổng quan vào `data/results2.csv`

#### 2.2 Phim cao nhất / thấp nhất
- Widget: `st.selectbox("Chọn chỉ số", stat_cols)`
- Ví dụ khi chọn `Revenue_USD`:
  - **Top 3 cao nhất:** Avatar ($2.79B), Titanic ($1.85B), The Avengers ($1.52B)
  - **Top 3 thấp nhất:** Split Second ($50), Bran Nue Dae ($70), Dreaming of Joseph Lees ($70)

#### 2.3 Biểu đồ phân phối
- Hàm: `plot_histogram(df, column)`
- Hoạt động đúng cho tất cả 4 cột số: `Duration_min`, `IMDb_Rating`, `Budget_USD`, `Revenue_USD`
- Hiển thị đường KDE, đường trung vị, định dạng tick tiền tệ (triệu/tỷ USD)

**Kiểm thử AppTest:** Part II chạy không ngoại lệ, bao gồm cả tương tác chọn cột `Revenue_USD`.

### Phần III: Học máy — Phân cụm & So sánh

**Input:** `data/cleaned/results.csv`

#### 3.1 Phân cụm K-means
- Đặc trưng: `Budget_USD`, `Revenue_USD`, `IMDb_Rating` (chuẩn hóa bằng `StandardScaler`)
- Thuật toán: `KMeans(n_clusters=k, random_state=42, n_init=10)`
- Số mẫu hợp lệ (có đủ 3 cột số): 3.227 phim
- Slider `n_clusters`: động từ 2 đến `min(10, 3227)`, giá trị mặc định = 3
- Sửa lỗi: `n_clusters` không thể lớn hơn số mẫu hợp lệ (tránh KeyError)

#### 3.2 Giảm chiều PCA
- `PCA(n_components=2, random_state=42)`
- Tỷ lệ phương sai giải thích lưu ở `df_pca.attrs["explained_variance_ratio"]`
- Vẽ scatter plot bản đồ cụm 2D, mỗi điểm là một phim

#### 3.3 So sánh hai phim bằng radar
- 5 thuộc tính: Rating, Votes, Budget, Revenue, Duration (tùy dữ liệu có sẵn)
- Chuẩn hóa Min-Max về [0, 1]
- Vẽ biểu đồ polar, so sánh 2 phim được chọn

**Kiểm thử AppTest:** Part III chạy không ngoại lệ.

### Phần IV: Dự đoán doanh thu phòng vé

**Input:** `data/cleaned/results.csv`  
**Mục tiêu:** `Revenue_USD`

#### 4.1 Mô hình
- Thuật toán: `RandomForestRegressor`
- Siêu tham số: `n_estimators=300`, `max_depth=15`, `min_samples_leaf=3`, `max_features="sqrt"`, `random_state=42`, `n_jobs=-1`
- Tiền xử lý số: `SimpleImputer(median)` + `StandardScaler`
- Tiền xử lý phân loại: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5)`
- Chia tập: 80% train (2.700 mẫu) / 20% test (676 mẫu), `random_state=42`

#### 4.2 Đặc trưng kỹ thuật
| Đặc trưng | Công thức | Mục đích |
|-----------|-----------|----------|
| `log_Budget_USD` | `log1p(Budget_USD)` | Giảm lệch phải của phân phối kinh phí |
| `Budget_x_Rating` | `Budget_USD × IMDb_Rating` | Tương tác giữa kinh phí và chất lượng phim |

> Lưu ý: Dữ liệu thô hiện tại **không có cột `Votes`**, nên đặc trưng `log_Votes` chưa được kích hoạt.

#### 4.3 Kết quả đánh giá (trên tập test)

| Chỉ số | Giá trị |
|--------|---------|
| **R² (mô hình)** | **0.573** |
| MAE | ~$70,7 triệu USD |
| RMSE | ~$118,0 triệu USD |
| Baseline R² (trung bình) | ≈ -0.000 |
| Baseline R² (trung vị) | ≈ -0.143 |
| Baseline MAE (trung bình) | ~$115,7 triệu USD |
| Baseline MAE (trung vị) | ~$100,7 triệu USD |

**Kết luận:** Mô hình Random Forest tốt hơn đáng kể so với baseline đơn giản (dự đoán trung bình/trung vị), chứng tỏ kinh phí, điểm IMDb, thời lượng và thể loại mang thông tin dự báo doanh thu.

#### 4.4 Dự đoán ví dụ
- Input: Budget=$10M, Rating=7.0, Duration=120 phút, Genre=Drama, Votes=1.000
- Dự đoán doanh thu: ~$49,6 triệu USD

**Kiểm thử AppTest:** Part IV chạy không ngoại lệ, bao gồm cả bấm nút "Huấn luyện mô hình" và dự đoán.


## 3. KIỂM THỬ TỰ ĐỘNG

### pytest
```
======================= test session starts =======================
collected 2 items

tests/test_data_pipeline.py::test_convert_zero_to_nan PASSED
tests/test_data_pipeline.py::test_missing_essential_columns PASSED

======================= 2 passed in 0.72s =======================
```

### Streamlit AppTest
Tất cả 4 phần chạy không ngoại lệ:
- ✅ Default section (Part I)
- ✅ Part II (Thống kê & Phân bố) + tương tác chọn cột
- ✅ Part III (Phân cụm & Radar)
- ✅ Part IV (Dự đoán doanh thu) + bấm nút Huấn luyện

## 4. TỪNG LỖI ĐÃ SỬA (Trong quá trình phát triển)

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| Part IV không hiển thị dự đoán | Nhãn sidebar `"Phần IV: Dự đoán Doanh thu phòng vé"` không khớp với `elif` condition `"Phần IV: Dự đoán Doanh thu"` | Cập nhật `elif` để khớp chính xác nhãn sidebar |
| Phân cụm báo `n_clusters=3 lớn hơn số mẫu hợp lệ (1)` | Slider `n_clusters` cố định tối đa 10, không tính đến số mẫu thực tế sau khi loại NaN | Tính `n_valid` từ `_prepare_cluster_features()`, giới hạn slider tối đa `min(10, n_valid)` |
| R² thấp (0.49) | Mô hình chưa có đặc trưng kỹ thuật, siêu tham số mặc định | Thêm `log_Budget_USD`, `Budget_x_Rating`; tuning `max_depth`, `min_samples_leaf`, `max_features`; R² tăng lên 0.57 |
| Log-transform target làm R² giảm | `log1p(Revenue)` làm R² từ 0.53 xuống 0.23 | Giữ target trên thang USD gốc, chỉ log-transform đặc trưng đầu vào |


## 5. CẤU TRÚC FILE ĐÃ THAY ĐỔI (HEAD commit)

```
b709a47 Them file PROJECT_CONTEXT.md: boi coc va muc tieu du an
4a034a4 Them tests va data_clean vao kho luu
83ecf29 Merge branch 'main' (chứa các cải tiến Part II-IV)

Files đã commit:
- app.py (Phần II-IV wiring, phân cụm động, predict_revenue)
- src/models.py (RandomForest tuning, log_Budget_USD, Budget_x_Rating, baseline_, predict_revenue)
- src/statistics_eda.py (calculate_basic_stats, get_top_bottom_movies)
- src/visualizer.py (plot_histogram, plot_radar_chart, seaborn)
- src/data_pipeline.py (tên cột dài, validation cột thiết yếu)
- src/data_clean.py (tương thích ngược)
- tests/test_data_pipeline.py (2 bài test)
- PROJECT_CONTEXT.md (bối cảnh và mục tiêu dự án)
- KET_QUA_CHAY_TRINH.md (báo cáo này)
```

---

## 6. CÁCH CHẠY LẠI TOÀN BỘ CHƯƠNG TRÌNH

```powershell
# 1. Cài đặt thư viện
python -m pip install -r requirements.txt

# 2. Thu thập dữ liệu thô (nếu chưa có)
python src\collect_data.py

# 3. Làm sạch dữ liệu
python -c "from src.data_pipeline import load_and_clean_data; load_and_clean_data('data/collected/resultsRaw_completed.csv')"

# 4. Chạy ứng dụng Streamlit
python -m streamlit run app.py

# 5. Chạy kiểm thử tự động
python -m pytest tests/ -v
```

> Lưu ý: Thư mục `data/` bị bỏ qua trong `.gitignore`. Sau khi clone repository, phải tạo dữ liệu thô và chạy pipeline trước khi dùng toàn bộ chức năng.

---

## 7. KẾT LUẬN

Toàn bộ chương trình đã chạy thành công trên 4 phần chức năng:
- **Phần I:** Làm sạch dữ liệu TMDB → 4.803 mẫu, 7 cột
- **Phần II:** Thống kê mô tả + histogram → các hàm chạy đúng, lưu `data/results2.csv`
- **Phần III:** Phân cụm K-means + PCA + radar → slider động, không lỗi `n_clusters`
- **Phần IV:** Dự đoán doanh thu Random Forest → R² = 0.57, tốt hơn baseline rõ rệt

Tất cả kiểm thử tự động (pytest + AppTest) đều pass. Code đã được commit và push lên `origin/main` (commit `b709a47`).

