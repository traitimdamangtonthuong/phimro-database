# TÀI LIỆU KỸ THUẬT — PHIMRO DATABASE

Mã dự án: `phimro-database`  
Ngày cập nhật: 20/10/2026  
Ngôn ngữ: Python 3.10+  
Framework chính: Streamlit, Pandas, Scikit-learn, Matplotlib, Seaborn

---

## 1. Giới thiệu dự án

Dự án **phimro-database** là một ứng dụng web phân tích và dự đoán doanh thu phim được xây dựng bằng **Streamlit**. Ứng dụng sử dụng dữ liệu phim từ **TMDB (The Movie Database)** gồm khoảng 5.000 bộ phim với các thuộc tính: tên phim, năm phát hành, thể loại, thời lượng, điểm IMDb, kinh phí, doanh thu, đạo diễn và diễn viên.

Dự án được chia thành 4 phần theo nhóm thực hiện:

| Phần | Nhóm | Nội dung chính | Trạng thái |
|------|------|----------------|------------|
| **Phần I** | Tổ 1 | Thu thập, chuẩn hóa và làm sạch dữ liệu TMDB | ✅ Hoàn thành |
| **Phần II** | Tổ 2 | Thống kê mô tả, phân phối và biểu đồ trực quan | ✅ Hoàn thành |
| **Phần III** | Tổ 3 | Học máy không giám sát: phân cụm K-means, giảm chiều PCA, radar chart | ✅ Hoàn thành |
| **Phần IV** | Tổ 4 | Học máy có giám sát: dự đoán doanh thu phòng vé bằng Random Forest | ✅ Hoàn thành |

---

## 2. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI (app.py)                   │
│  Phần I: Làm sạch  │  Phần II: Thống kê  │  Phần III: ML-KM │
│  Phần IV: Dự đoán                                            │
└─────────────────────────────────────────────────────────────┘
                              │
    ┌───────────┬───────────┬───────────┬───────────┐
    │           │           │           │           │
    ▼           ▼           ▼           ▼           ▼
data_pipeline.py statistics_eda.py visualizer.py models.py collect_data.py
(Phần I)        (Phần II)     (Phần II/III) (Phần III/IV) (Phần I)
                              │           │           │
                              └───────────┴───────────┘
                                      │
                              ┌───────▼───────┐
                              │ data/cleaned/ │
                              │ results.csv   │
                              └───────┬───────┘
                                      │
                              ┌───────▼───────┐
                              │ data/collected/│
                              │ resultsRaw...  │
                              └───────────────┘
```

**Luồng dữ liệu chính:**

1. `src/collect_data.py` — Tải dữ liệu thô từ TMDB → `data/collected/resultsRaw_completed.csv`
2. `src/data_pipeline.py` — Làm sạch dữ liệu thô → `data/cleaned/results.csv`
3. `app.py` — Giao diện Streamlit, gọi các module phân tích/dự đoán trên dữ liệu đã làm sạch
4. `src/statistics_eda.py`, `src/visualizer.py`, `src/models.py` — Logic xử lý cho Phần II–IV

---

## 3. Cấu trúc thư mục

```
phimro-database/
├── app.py                         # Giao diện Streamlit chính (4 phần)
├── requirements.txt               # Thư viện Python cần thiết
├── README.md                      # Hướng dẫn chạy nhanh
├── PROJECT_CONTEXT.md             # Bối cảnh và mục tiêu dự án
├── DOCUMENTATION.md               # Tài liệu kỹ thuật chi tiết (file này)
├── KET_QUA_CHAY_TRINH.md          # Báo cáo kết quả chạy thử
├── .gitignore                     # Bỏ qua data/*.csv, __pycache__, .idea, v.v.
├── data/
│   ├── collected/                 # Dữ liệu thô từ TMDB (không commit)
│   └── cleaned/                   # Dữ liệu đã làm sạch (sinh bởi pipeline)
├── tests/
│   └── test_data_pipeline.py      # 2 bài test kiểm tra pipeline làm sạch
└── src/
    ├── collect_data.py            # Thu thập dữ liệu TMDB
    ├── data_pipeline.py           # Pipeline làm sạch chính (single source of truth)
    ├── data_clean.py              # Lớp tương thích ngược (không dùng chính)
    ├── statistics_eda.py          # Thống kê mô tả, top/bottom phim (Phần II)
    ├── visualizer.py              # Biểu đồ histogram, radar chart (Phần II/III)
    └── models.py                  # K-means, PCA, RandomForest (Phần III/IV)
```

## 4. Chi tiết các module

### 4.1 src/data_pipeline.py — Pipeline làm sạch dữ liệu (Phần I)

**Vai trò:** Single source of truth cho việc đọc, chuẩn hóa và làm sạch dữ liệu thô từ TMDB. Được gọi bởi `app.py`, `src/data_clean.py` (tương thích ngược) và các bài test.

#### Hàm chính: `load_and_clean_data(raw_file_path)`

| Thuộc tính | Giá trị |
|------------|---------|
| Tham số | `raw_file_path` (str hoặc pathlib.Path) — đường dẫn file CSV thô |
| Trả về | `pandas.DataFrame` — dữ liệu 9 cột chuẩn đã làm sạch |
| Ném | `FileNotFoundError` nếu đường dẫn không tồn tại; `KeyError` nếu thiếu cột thiết yếu |

**Các bước xử lý:**

| Bước | Nội dung |
|------|----------|
| 1 | Đọc CSV bằng `pandas.read_csv()` |
| 2 | Kiểm tra cột thiết yếu: `Budget_USD` (nếu thiếu → ném `KeyError`) |
| 3 | Đổi tên cột thô về chuẩn qua `_RENAME_MAP` (ví dụ: `Release_Year` → `Year`, `Actors` → `Cast`) |
| 4 | Parse cột `Genre` từ chuỗi JSON thành danh sách tên thể loại (ví dụ: `"Action, Adventure"`) |
| 5 | Ép `Budget`, `Revenue` về float qua `_clean_currency()` (0, "", "N/a" → `NaN`) |
| 6 | Ép các cột số (`Year`, `Votes`, `Duration_min`, `IMDb_Rating`, `Budget_USD`, `Revenue_USD`) về float; giá trị 0 được coi là thiếu dữ liệu theo quy ước TMDB |
| 7 | Loại bản ghi trùng theo cặp `(Title, Year)`; sắp xếp theo `Title` A-Z, `Year` giảm dần |
| 8 | Giữ đúng 9 cột chuẩn: `Title`, `Year`, `Genre`, `Votes`, `Duration_min`, `IMDb_Rating`, `Budget_USD`, `Revenue_USD`, `Director`, `Cast` |
| 9 | Xuất `data/cleaned/results.csv` (mã hóa `utf-8-sig`) |

#### Hàm hỗ trợ

| Hàm | Mô tả |
|-----|-------|
| `_clean_genre(val)` | Trích tên thể loại từ chuỗi JSON, có fallback regex nếu JSON lỗi |
| `_clean_currency(val)` | Ép chuỗi tiền tệ (`"$1,234"`, `"N/a"`, rỗng) về `float`, lỗi thành `NaN` |

#### Hằng số cấu hình

| Hằng số | Giá trị |
|---------|---------|
| `_TARGET_COLUMNS` | 9 cột đầu ra chuẩn |
| `_RENAME_MAP` | Ánh xạ tên cột thô → tên chuẩn |
| `_NUMERIC_COLUMNS` | Cột số cần ép kiểu |
| `_ESSENTIAL_COLUMNS` | Cột bắt buộc (`Budget_USD`); thiếu cột này sẽ ném `KeyError` |

#### Quy ước giá trị 0

Trong dữ liệu TMDB, giá trị `0` trên các cột kinh tế và điểm số biểu thị **"chưa công bố / không có dữ liệu"**, không phải giá trị thực. Pipeline quy ước chuyển `0` thành `NaN` cho các cột: `Budget_USD`, `Revenue_USD`, `IMDb_Rating`, `Duration_min`, `Votes`, `Year`.


### 4.2 src/statistics_eda.py — Thống kê mô tả (Phần II)

**Vai trò:** Tính toán thống kê mô tả và tìm phim có giá trị cực trị (cao nhất/thấp nhất).

#### Hàm: `calculate_basic_stats(df)`

| Thuộc tính | Giá trị |
|------------|---------|
| Tham số | `df` (pandas.DataFrame) — dữ liệu đã làm sạch |
| Trả về | `tuple[DataFrame, DataFrame]` — `(overall, by_genre)` |
| `overall` | Mỗi hàng là một cột số; các cột: `Mean`, `Median`, `Std` (mẫu, ddof=1); chỉ mục tên cột |
| `by_genre` | Thống kê theo thể loại, cột phẳng: `Budget_mean`, `Budget_median`, `Budget_std`, ...; cột `Movie_Count` là số phim thuộc thể loại |

**Chi tiết xử lý:**
- Chọn các cột trong `STAT_COLUMNS = ["Duration_min", "IMDb_Rating", "Budget_USD", "Revenue_USD"]` nếu có trong df
- Ép kiểu số, thay `inf`/`−inf` bằng `NaN`
- Nếu df rỗng hoặc không có cột số → trả về 2 bảng rỗng
- Phim đa thể loại (cột `Genre` chứa nhiều giá trị phân tách bằng dấu phẩy) được tính vào từng thể loại của nó
- Gộp MultiIndex cột thành tên đơn tầng (ví dụ: `Budget_mean`, `Budget_median`, `Budget_std`)

#### Hàm: `get_top_bottom_movies(df, column, top_n=3)`

| Thuộc tính | Giá trị |
|------------|---------|
| Tham số | `df` (DataFrame), `column` (tên cột số), `top_n` (số phim mỗi đầu, mặc định 3) |
| Trả về | `tuple[DataFrame, DataFrame]` — `(top_movies, bottom_movies)` với cột `Title` và `column` |
| Ném | `ValueError` nếu thiếu cột `Title` hoặc cột chỉ số |

**Chi tiết:**
- Ép kiểu cột số, loại `NaN` và `inf`
- Dùng `nlargest()` và `nsmallest()`
- Nếu không có giá trị hợp lệ → trả về 2 bảng rỗng cùng cấu trúc
- Không sửa đổi DataFrame đầu vào

#### Hằng số

| Hằng số | Giá trị |
|---------|---------|
| `STAT_COLUMNS` | `["Duration_min", "IMDb_Rating", "Budget_USD", "Revenue_USD"]` |


### 4.3 src/visualizer.py — Biểu đồ trực quan (Phần II & III)

**Vai trò:** Tạo biểu đồ Matplotlib/Seaborn cho histogram phân phối và radar chart so sánh hai phim.

#### Hàm: `plot_histogram(df, column)`

| Thuộc tính | Giá trị |
|------------|---------|
| Tham số | `df` (DataFrame), `column` (tên cột số) |
| Trả về | `matplotlib.figure.Figure` |
| Ném | `ValueError` nếu cột không tồn tại, không phải số, hoặc không có dữ liệu hợp lệ |

**Chi tiết:**
- Vẽ `sns.histplot` với `bins=20`, `kde=True`, màu `steelblue`, viền trắng
- Vẽ đường thẳng trung vị màu `darkorange`, kèm nhãn giá trị đã định dạng (triệu/tỷ USD, phút, hoặc /10 cho điểm)
- Định dạng tick trục X cho tiền tệ lớn (≥ 1 tỷ USD)
- Gọi `plt.close(fig)` sau khi hiển thị để giải phóng bộ nhớ

#### Hàm: `plot_radar_chart(df, movie1_name, movie2_name)`

| Thuộc tính | Giá trị |
|------------|---------|
| Tham số | `df` (DataFrame), `movie1_name`, `movie2_name` (tên phim) |
| Trả về | `matplotlib.figure.Figure` — biểu đồ radar polar |
| Ném | `TypeError` nếu df không phải DataFrame; `ValueError` nếu thiếu cột `Title`, không tìm thấy phim, tên trùng, hoặc ít hơn 3 thuộc tính đủ dữ liệu |

**Chi tiết:**
- 5 thuộc tính so sánh: `IMDb_Rating`, `Votes`, `Budget_USD`, `Revenue_USD`, `Duration_min`
- Chuẩn hóa Min-Max về [0, 1] dựa trên toàn bộ dữ liệu tham chiếu
- Vẽ trên projection polar, mỗi phim là một đa giác kèm tô màu trong suốt
- Thang 0–100 trên biểu đồ là vị trí tương đối; giá trị gốc kèm đơn vị hiển thị bên cạnh

#### Hằng số

| Hằng số | Mô tả |
|---------|-------|
| `_RADAR_ATTRIBUTES` | Ánh xạ tên chuẩn → các tên cột chấp nhận được |
| `_RADAR_LABELS` | Nhãn hiển thị cho người xem (tiếng Việt) |
| `_HISTOGRAM_LABELS` | Nhãn và đơn vị cho histogram |
| `_CURRENCY_COLUMNS` | Tập hợp các cột tiền tệ |


### 4.4 src/models.py — Học máy (Phần III & IV)

**Vai trò:** Chứa các hàm phân cụm K-means, giảm chiều PCA, và dự đoán doanh thu bằng Random Forest.

#### Hàm: `run_kmeans_clustering(df, n_clusters)`

| Thuộc tính | Giá trị |
|------------|---------|
| Tham số | `df` (DataFrame), `n_clusters` (int) |
| Trả về | `DataFrame` — dữ liệu hợp lệ cộng cột `Cluster` |
| Ném | `ValueError` nếu `n_clusters` không phải số nguyên, < 1, > số mẫu hợp lệ, hoặc > số mẫu phân biệt |

**Chi tiết:**
- Đặc trưng: `Budget_USD`, `Revenue_USD`, `IMDb_Rating`
- Chuẩn hóa bằng `StandardScaler`
- `KMeans(random_state=42, n_init=10)`
- Các hàng thiếu/vô hạn bị loại khỏi kết quả

#### Hàm: `run_pca_reduction(df)`

| Thuộc tính | Giá trị |
|------------|---------|
| Tham số | `df` (DataFrame) |
| Trả về | `DataFrame` — hai cột `Trục X (PCA 1)`, `Trục X (PCA 2)` |
| Thuộc tính attrs | `df_pca.attrs["explained_variance_ratio"]` — mảng 2 phần tử tỷ lệ phương sai giải thích |

**Chi tiết:**
- Dùng cùng đặc trưng với K-means
- Chuẩn hóa `StandardScaler` rồi `PCA(n_components=2, random_state=42)`
- Giữ chỉ mục gốc để ghép với tên phim/nhãn cụm

#### Hàm: `train_revenue_prediction_model(df)`

| Thuộc tính | Giá trị |
|------------|---------|
| Tham số | `df` (DataFrame) |
| Trả về | `sklearn.pipeline.Pipeline` — mô hình đã huấn luyện |
| Thuộc tính gán | `model.metrics_`, `model.baseline_`, `model.feature_columns_`, `model.target_log_transformed_` |

**Chi tiết:**
- Mục tiêu: `Revenue_USD` (thang USD gốc, không log-transform)
- Đặc trưng số: `Budget_USD`, `IMDb_Rating`, `Duration_min`, `Votes` (nếu có)
- Đặc trưng kỹ thuật: `log_Budget_USD`, `log_Votes`, `Budget_x_Rating` (tương tác)
- Đặc trưng phân loại: `Genre`
- Tiền xử lý số: `SimpleImputer(median)` + `StandardScaler`
- Tiền xử lý phân loại: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5)`
- Mô hình: `RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3, max_features="sqrt", random_state=42, n_jobs=-1)`
- Chia train/test: 80/20, `random_state=42`, chia trước khi fit tiền xử lý

**`model.metrics_`:**
| Key | Mô tả |
|-----|-------|
| `MAE` | Mean Absolute Error (USD) |
| `RMSE` | Root Mean Squared Error (USD) |
| `R2` | Hệ sốdetermination trên tập test |
| `n_train`, `n_test` | Số mẫu train/test |

**`model.baseline_`:**
| Key | Mô tả |
|-----|-------|
| `mean_R2`, `median_R2` | R² của baseline dự đoán trung bình / trung vị |
| `mean_MAE`, `median_MAE` | MAE của baseline |

#### Hàm: `predict_revenue(model, X_raw)`

| Thuộc tính | Giá trị |
|------------|---------|
| Tham số | `model` (pipeline đã train), `X_raw` (DataFrame đặc trưng thô) |
| Trả về | `numpy.ndarray` — doanh thu dự đoán (USD) |
| Ném | `TypeError` nếu X_raw không phải DataFrame; `ValueError` nếu thiếu cột đặc trưng |

**Chi tiết:** Áp dụng cùng biến đổi log và tương tác như lúc huấn luyện, rồi gọi `model.predict()`.

#### Hàm: `_prepare_cluster_features(df)` (hàm nội bộ)

Trích xuất 3 đặc trưng số cho K-means/PCA, ép kiểu số, loại `NaN`/`inf`, giữ chỉ mục gốc.

#### Hằng số

| Hằng số | Mô tả |
|---------|-------|
| `_CLUSTER_FEATURES` | Ánh xạ tên chuẩn → các tên cột cho clustering |


### 4.5 src/collect_data.py — Thu thập dữ liệu (Phần I)

**Vai trò:** Tải và ghép dữ liệu từ TMDB (5000 movies + credits), tạo file `resultsRaw_completed.csv` trong `data/collected/`.

**Hàm chính:** `main(download=False)` — nếu `download=False`, dùng dữ liệu TMDB đã có sẵn local để tạo bảng thô.

#### Hằng số

| Hằng số | Mô tả |
|---------|-------|
| `_RENAME_MAP` | Ánh xạ cột thô → tên chuẩn (`Release_Year` → `Year`, `Actors` → `Cast`) |

---

### 4.6 app.py — Giao diện Streamlit (Tất cả phần)

**Vai trò:** Ứng dụng web chính, điều hướng 4 phần qua sidebar radio.

**Hàm chính:**

| Hàm | Mô tả |
|-----|-------|
| `load_cleaned_data(file_modified_at)` | Đọc dữ liệu sạch, có cache key theo thời gian sửa file |
| `get_data()` | Ưu tiên `st.session_state["df"]`, ngược lại đọc file sạch |
| `format_usd(value)` | Định dạng tiền tệ USD (tỷ/triệu) |
| `format_year(value)` | Định dạng năm phát hành |

**Các phần giao diện:**

| Phần | Chức năng chính |
|------|-----------------|
| Phần I | Nút "Chạy làm sạch dữ liệu", hiển thị số bộ phim/số cột |
| Phần II | Chọn chỉ số số, bảng thống kê, Top/Bottom 3, histogram |
| Phần III | Slider `n_clusters` động, bản đồ PCA, radar chart so sánh 2 phim |
| Phần IV | Nút Huấn luyện, hiển thị MAE/RMSE/R² + baseline, form dự đoán doanh thu |

**Trạng thái phiên:** Mô hình dự đoán được lưu trong `st.session_state["revenue_model"]` để không train lại sau mỗi tương tác.

---

## 5. API tham chiếu (tổng hợp tất cả hàm công khai)

| Hàm | Module | Mô tả |
|-----|--------|-------|
| `load_and_clean_data(raw_file_path)` | `data_pipeline.py` | Làm sạch dữ liệu và ghi `data/cleaned/results.csv` |
| `calculate_basic_stats(df)` | `statistics_eda.py` | Trả về `(overall, by_genre)` với thống kê tổng quan và theo thể loại |
| `get_top_bottom_movies(df, column, top_n=3)` | `statistics_eda.py` | Trả về hai bảng phim có giá trị cao nhất/thấp nhất |
| `plot_histogram(df, column)` | `visualizer.py` | Trả về đối tượng `matplotlib.figure.Figure` |
| `plot_radar_chart(df, movie1_name, movie2_name)` | `visualizer.py` | Trả về radar so sánh hai phim |
| `run_kmeans_clustering(df, n_clusters)` | `models.py` | Trả về dữ liệu hợp lệ cùng nhãn `Cluster` |
| `run_pca_reduction(df)` | `models.py` | Trả về hai tọa độ PCA và lưu explained variance trong `.attrs` |
| `train_revenue_prediction_model(df)` | `models.py` | Trả về pipeline đã huấn luyện, kèm `.metrics_` và `.feature_columns_` |
| `predict_revenue(model, X_raw)` | `models.py` | Dự đoán doanh thu bằng pipeline đã huấn luyện |


---

## 6. Schema dữ liệu

Tệp `data/cleaned/results.csv` hiện có 10 cột. Dữ liệu mẫu đang có 4.803 bộ phim; số dòng có thể thay đổi nếu nguồn dữ liệu được tạo lại.

| Cột | Kiểu | Ý nghĩa |
|-----|------|---------|
| `Title` | text | Tên phim |
| `Year` | number | Năm phát hành |
| `Genre` | text | Một hoặc nhiều thể loại, phân tách bằng dấu phẩy |
| `Votes` | number | Số lượt bình chọn trên TMDB (`vote_count`) |
| `Duration_min` | number | Thời lượng, phút |
| `IMDb_Rating` | number | Điểm TMDB (`vote_average`), thang 0–10 |
| `Budget_USD` | number | Kinh phí, USD |
| `Revenue_USD` | number | Doanh thu, USD |
| `Director` | text | Tên đạo diễn |
| `Cast` | text | Danh sách diễn viên chính |

## 7. Cách cài đặt và chạy

### 7.1 Cài đặt thư viện

```powershell
python -m pip install -r requirements.txt
```

### 7.2 Tạo dữ liệu thô

```powershell
python src\collect_data.py
```

Lệnh này tạo các tệp trong `data/collected/`, gồm `resultsRaw_completed.csv`.

### 7.3 Làm sạch dữ liệu

```powershell
python -c "from src.data_pipeline import load_and_clean_data; load_and_clean_data('data/collected/resultsRaw_completed.csv')"
```

Hoặc mở ứng dụng và chọn **Phần I → Chạy làm sạch dữ liệu**. Kết quả được ghi vào `data/cleaned/results.csv`.

### 7.4 Chạy web

```powershell
python -m streamlit run app.py
```

> Thư mục `data/` được bỏ qua khỏi Git. Sau khi clone repository, cần tạo dữ liệu thô và chạy pipeline trước khi dùng toàn bộ chức năng của ứng dụng.

### 7.5 Chạy kiểm thử tự động

```powershell
python -m pytest tests/ -v
```

Hiện có 2 bài test cho `src/data_pipeline.py`, đều pass.

---

## 8. Quy ước kỹ thuật

| Quy ước | Nội dung |
|---------|----------|
| Không điền suy đoán | Không điền giá trị suy đoán cho `Revenue_USD` khi thiếu |
| Hàm biểu đồ | Trả về `fig`, không gọi `plt.show()`; giao diện tự hiển thị và đóng figure |
| K-means/PCA | Dùng `StandardScaler`; radar dùng Min-Max scaling trên toàn bộ dữ liệu tham chiếu |
| Tái lập | Mọi bước ngẫu nhiên quan trọng dùng `random_state=42` |
| Mã hóa CSV | File đầu ra dùng `utf-8-sig` để mở chính xác trên Excel |
| Giá trị 0 | Coi là thiếu dữ liệu (`NaN`) cho các cột số kinh tế và điểm số |

---

## 9. Ví dụ sử dụng

### 9.1 Làm sạch dữ liệu từ Python

```python
from src.data_pipeline import load_and_clean_data

df = load_and_clean_data("data/collected/resultsRaw_completed.csv")
print(df.shape)          # ví dụ: (4803, 10)
print(df.columns.tolist())
```

### 9.2 Thống kê cơ bản

```python
from src.statistics_eda import calculate_basic_stats

overall, by_genre = calculate_basic_stats(df)
print(overall)
print(by_genre.head())
```

### 9.3 Tìm phim cao nhất / thấp nhất

```python
from src.statistics_eda import get_top_bottom_movies

top, bottom = get_top_bottom_movies(df, "Revenue_USD", top_n=3)
print(top)
print(bottom)
```

### 9.4 Huấn luyện và dự đoán doanh thu

```python
from src.models import train_revenue_prediction_model, predict_revenue
import pandas as pd

model = train_revenue_prediction_model(df)
print(model.metrics_)

# Dự đoán cho phim mới
new_movie = pd.DataFrame([{
    "Budget_USD": 10_000_000.0,
    "IMDb_Rating": 7.0,
    "Duration_min": 120,
    "Genre": "Drama",
    "Votes": 1_000,
}])
prediction = predict_revenue(model, new_movie)
print(f"Doanh thu dự đoán: ${prediction[0]:,.0f}")
```

### 9.5 Phân cụm K-means

```python
from src.models import run_kmeans_clustering

clustered = run_kmeans_clustering(df, n_clusters=3)
print(clustered["Cluster"].value_counts())
```

---

