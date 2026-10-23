"""Giao diện chính cho ứng dụng phân tích và dự đoán dữ liệu phim."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_pipeline import load_and_clean_data
from src.models import (
    run_kmeans_clustering,
    run_pca_reduction,
    train_revenue_prediction_model,
    predict_revenue,
    _prepare_cluster_features,
)
from src.statistics_eda import calculate_basic_stats, get_top_bottom_movies
from src.visualizer import plot_histogram, plot_radar_chart

# Đường dẫn dữ liệu sạch (Tổ 1 xuất ra), tính từ vị trí file app.py
BASE_DIR = Path(__file__).resolve().parent
CLEANED_CSV = BASE_DIR / "data" / "cleaned" / "results.csv"


@st.cache_data
def load_cleaned_data(file_modified_at):
    """Đọc dữ liệu sạch cho các trang; trả về None nếu chưa có tệp.

    Thời điểm sửa file là một phần của cache key, nên cache tự hết hiệu lực
    khi Phần I ghi một phiên bản dữ liệu sạch mới.
    """
    del file_modified_at  # Chỉ dùng để phân biệt phiên bản cache.
    if not CLEANED_CSV.exists():
        return None
    return pd.read_csv(CLEANED_CSV)


def get_data():
    """Lấy dữ liệu để phân tích: ưu tiên bảng vừa làm sạch trong session.

    Trả về None nếu người dùng chưa tải/làm sạch và cũng chưa có
    data/cleaned/results.csv.
    """
    if "df" in st.session_state:
        return st.session_state["df"]
    if not CLEANED_CSV.exists():
        return None
    return load_cleaned_data(CLEANED_CSV.stat().st_mtime_ns)


def format_usd(value):
    """Định dạng tiền USD ngắn gọn, phù hợp phần dữ liệu gốc của radar."""
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return "Không có dữ liệu"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}".replace(".", ",") + " tỷ USD"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.0f}".replace(",", ".") + " triệu USD"
    return f"{value:,.0f}".replace(",", ".") + " USD"


def format_year(value):
    """Định dạng năm phát hành nếu dữ liệu có sẵn."""
    value = pd.to_numeric(value, errors="coerce")
    return str(int(value)) if pd.notna(value) else "Không rõ năm"


def apply_streamlit_theme():
    """Áp dụng theme trình bày; không làm thay đổi bố cục nội dung từng phần."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;500;600;700&display=swap');

        :root {
            --streamlit-background: #0E1117;
            --streamlit-surface: #151A23;
            --streamlit-text: #FAFAFA;
            --streamlit-muted: #A6AFBD;
            --streamlit-border: #303846;
            --streamlit-primary: #FF4B4B;
        }

        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
            display: none;
        }

        header[data-testid="stHeader"] {
            display: none;
        }

        .stApp {
            background: var(--streamlit-background);
            color: var(--streamlit-text);
            font-family: "Source Sans Pro", sans-serif;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 6.1rem;
            padding-bottom: 3.5rem;
        }

        .st-key-top_nav {
            position: fixed;
            z-index: 999999;
            top: 0;
            left: 50%;
            width: 100vw;
            transform: translateX(-50%);
            min-height: 64px;
            padding: 0 5vw;
            background: rgba(14, 17, 23, 0.96);
            border-bottom: 1px solid var(--streamlit-border);
            box-shadow: 0 1px 6px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(10px);
        }

        .top-navigation__brand {
            display: flex;
            align-items: center;
            min-height: 64px;
            color: var(--streamlit-text) !important;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            white-space: nowrap;
        }

        .top-navigation__brand::before {
            content: "●";
            margin-right: 0.45rem;
            color: var(--streamlit-primary);
            font-size: 0.75rem;
        }

        .st-key-top_nav [data-testid="stHorizontalBlock"] {
            align-items: stretch;
            gap: 0;
        }

        .st-key-top_nav .stButton {
            height: 64px;
        }

        .st-key-top_nav .stButton > button {
            position: relative;
            width: 100%;
            height: 64px;
            min-height: 64px;
            padding: 0 0.75rem;
            border: 0 !important;
            border-radius: 0;
            color: var(--streamlit-muted) !important;
            background: transparent !important;
            font-size: 1rem;
            font-weight: 600;
            box-shadow: none !important;
        }

        .st-key-top_nav .stButton > button:hover {
            color: var(--streamlit-text) !important;
            background: rgba(255, 255, 255, 0.04) !important;
        }

        .st-key-top_nav .stButton > button[kind="primary"] {
            color: var(--streamlit-text) !important;
            background: transparent !important;
        }

        .st-key-top_nav .stButton > button[kind="primary"]::after {
            position: absolute;
            right: 0.75rem;
            bottom: 0;
            left: 0.75rem;
            height: 3px;
            content: "";
            background: var(--streamlit-primary);
            border-radius: 3px 3px 0 0;
        }

        .hero-banner {
            position: relative;
            min-height: 300px;
            display: grid;
            place-items: center;
            overflow: hidden;
            padding: 2rem;
            border-radius: 0.5rem;
            background-image:
                linear-gradient(90deg, rgba(18, 22, 31, 0.86), rgba(18, 22, 31, 0.54)),
                url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=2070");
            background-position: center;
            background-size: cover;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.16);
            text-align: center;
        }

        .hero-banner h1 {
            max-width: 900px;
            margin: 0;
            color: #FFFFFF;
            font-size: clamp(1.8rem, 4vw, 3rem);
            font-weight: 700;
            letter-spacing: -0.025em;
            line-height: 1.18;
            text-shadow: 0 1px 10px rgba(0, 0, 0, 0.4);
        }

        .hero-banner .hero-subtitle {
            color: rgba(255, 255, 255, 0.85);
            font-size: clamp(0.95rem, 2vw, 1.15rem);
            margin-top: 0.75rem;
            text-shadow: 0 1px 8px rgba(0, 0, 0, 0.4);
        }

        div[data-testid="stMetric"] {
            min-height: 100px;
            padding: 1.25rem 1.35rem;
            background: var(--streamlit-surface);
            border: 1px solid var(--streamlit-border);
            border-radius: 0.5rem;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--streamlit-muted);
            font-size: 0.95rem;
            font-weight: 500;
        }

        div[data-testid="stMetricValue"] {
            color: var(--streamlit-text);
            font-family: "Source Sans Pro", sans-serif;
            font-weight: 700;
        }

        .stButton > button {
            border: 1px solid var(--streamlit-primary);
            border-radius: 0.5rem;
            color: var(--streamlit-primary);
            font-family: "Source Sans Pro", sans-serif;
            font-weight: 600;
        }

        .stButton > button:hover {
            color: #FFFFFF;
            background: var(--streamlit-primary);
            border-color: var(--streamlit-primary);
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stExpander"] {
            background: var(--streamlit-surface);
            border: 1px solid var(--streamlit-border);
            border-radius: 0.5rem;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
            overflow: hidden;
        }

        h1, h2, h3 { color: var(--streamlit-text); }

        
        </style>
        """,
        unsafe_allow_html=True,
    )


def _select_page(page_key):
    """Đổi nội dung trong phiên Streamlit, không điều hướng trình duyệt."""
    st.session_state["page"] = page_key


def render_top_navigation(current_page):
    """Hiển thị top navigation bằng nút Streamlit trong cùng một trang."""
    pages = (
        ("overview", "Tổng quan"),
        ("part-i", "Phần I"),
        ("part-ii", "Phần II"),
        ("part-iii", "Phần III"),
        ("part-iv", "Phần IV"),
    )
    with st.container(key="top_nav"):
        columns = st.columns((2.4, 1, 1, 1, 1, 1), gap="small")
        columns[0].markdown(
            '<div class="top-navigation__brand">Phimro-Database</div>',
            unsafe_allow_html=True,
        )
        for column, (page_key, label) in zip(columns[1:], pages):
            column.button(
                label,
                key=f"nav_{page_key}",
                type="primary" if page_key == current_page else "secondary",
                use_container_width=True,
                on_click=_select_page,
                args=(page_key,),
            )


def render_overview():
    """Hiển thị trang Tổng quan theo thiết kế hero và metrics Streamlit."""
    st.markdown(
        """
        <section class="hero-banner">
            <h1>PHIMRO-DATABASE: HỆ THỐNG PHÂN TÍCH VÀ DỰ ĐOÁN
            DOANH THU PHIM ĐIỆN ẢNH</h1>
            <p class="hero-subtitle">Phân tích dữ liệu TMDB, thống kê mô tả,
            phân cụm K-means và dự đoán doanh thu phòng vé bằng Random Forest</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Tập dữ liệu", "4.803 phim", "Dữ liệu đã làm sạch")
    m2.metric("Mô hình Random Forest (R²)", "0.699", "Trên tập kiểm tra")
    m3.metric(
        "Sai số tuyệt đối (MAE)",
        "$57,201,999",
        "Sai số trên tập kiểm tra",
        delta_color="inverse",
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", "$99,195,202", "Trên tập kiểm tra")
    c2.metric("Số thể loại", "21", "Action, Comedy, Drama…")
    c3.metric("Số cột dữ liệu", "10", "Title, Genre, Budget…")
    c4.metric("Số mẫu test", "676", "20% dữ liệu hợp lệ")

    st.markdown("---")

    st.subheader("Truy cập nhanh")
    q1, q2, q3, q4 = st.columns(4)
    if q1.button("🧹 Làm sạch dữ liệu", use_container_width=True, key="q_clean"):
        st.session_state["page"] = "part-i"
        st.rerun()
    if q2.button("📊 Thống kê & Phân bố", use_container_width=True, key="q_stats"):
        st.session_state["page"] = "part-ii"
        st.rerun()
    if q3.button("🎯 Phân cụm & Radar", use_container_width=True, key="q_cluster"):
        st.session_state["page"] = "part-iii"
        st.rerun()
    if q4.button("💰 Dự đoán doanh thu", use_container_width=True, key="q_predict"):
        st.session_state["page"] = "part-iv"
        st.rerun()



# Các hàm khác được nhập sẵn để từng nhóm tích hợp khi hoàn thành.
st.set_page_config(
    page_title="Phimro-Database",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_streamlit_theme()
if "page" not in st.session_state:
    st.session_state["page"] = "overview"
page = st.session_state["page"]
page_to_section = {
    "overview": "Tổng quan",
    "part-i": "Phần I: Tổng quan & Xử lý Dữ liệu",
    "part-ii": "Phần II: Thống kê & Phân bố",
    "part-iii": "Phần III: Học máy - Phân cụm & So sánh",
    "part-iv": "Phần IV: Dự đoán Doanh thu phòng vé",
}
if page not in page_to_section:
    page = "overview"
    st.session_state["page"] = page
section = page_to_section[page]
render_top_navigation(page)

if section == "Tổng quan":
    render_overview()

elif section == "Phần I: Tổng quan & Xử lý Dữ liệu":
    st.header("Phần I: Tổng quan và làm sạch dữ liệu")

    # Nút làm sạch file thô mặc định có sẵn trong repo (không cần upload)
    default_raw = BASE_DIR / "data" / "collected" / "resultsRaw_completed.csv"
    if not default_raw.exists():
        default_raw = BASE_DIR / "data" / "resultsRaw.csv"

    st.subheader("1. Làm sạch từ file thô trong dự án")
    if default_raw.exists():
        st.caption(
            f"File thô: data/{default_raw.parent.name}/{default_raw.name}"
        )
        if st.button("Chạy làm sạch dữ liệu"):
            with st.spinner("Đang làm sạch..."):
                try:
                    st.session_state["df"] = load_and_clean_data(default_raw)
                    # File đã được ghi mới; bỏ kết quả đọc cũ để các phần
                    # III/IV không sử dụng DataFrame đã cache từ lần trước.
                    load_cleaned_data.clear()
                    st.success(
                        "Đã làm sạch và lưu ra data/cleaned/results.csv!"
                    )
                except Exception as exc:
                    st.error(f"Làm sạch thất bại: {exc}")
    else:
        st.warning("Chưa có file thô. Hãy chạy `python src/collect_data.py`.")


    # Hiển thị kết quả nếu đã có dữ liệu sạch
    df_raw = st.session_state.get("df")
    if df_raw is None:
        df_raw = get_data()
    if df_raw is None:
        st.info("Chưa có dữ liệu. Hãy chạy làm sạch hoặc tải file thô lên.")
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Số bộ phim", f"{len(df_raw):,}")
        col_b.metric("Số cột", len(df_raw.columns))
        
        st.subheader("Xem trước dữ liệu đã làm sạch")
        st.dataframe(df_raw, use_container_width=True)

elif section == "Phần II: Thống kê & Phân bố":
    st.header("Phần II: Thống kê và phân phối")
    
    # 1. Lấy dữ liệu từ session (nếu đã làm sạch ở Phần I)
    df = get_data()
    
    if df is None:
        st.error("Chưa có dữ liệu. Hãy quay lại 'Phần I' để chạy làm sạch hoặc tải file lên trước.")
    else:
            # Cột số khả dụng cho thống kê
            stat_cols = [c for c in ("Duration_min", "IMDb_Rating", "Budget_USD", "Revenue_USD") if c in df.columns]
            if not stat_cols:
                st.warning("Không tìm thấy cột số nào để thống kê trong dữ liệu.")
            else:
                # --- 1. Bảng thống kê cơ bản ---
                st.subheader("1. Thống kê mô tả cơ bản")
                try:
                    st.subheader("Thống kê chung")
                    stats_overall, stats_by_genre = calculate_basic_stats(df)
                    st.dataframe(stats_overall, use_container_width=True)
                    if not stats_by_genre.empty:
                        st.subheader("Thống kê theo thể loại")
                        st.dataframe(stats_by_genre, use_container_width=True)
                    # Lưu hai artifact: thống kê chung và bảng genre phẳng,
                    # dùng trực tiếp được trong Excel/báo cáo.
                    stats_by_genre.to_csv(
                        "data/results2_by_genre.csv",
                        index=True,
                        encoding="utf-8-sig",
                    )
                    try:
                        stats_overall.to_csv(
                            "data/results2.csv", index=True, encoding="utf-8-sig"
                        )
                        st.caption(
                            "Đã lưu data/results2.csv và "
                            "data/results2_by_genre.csv"
                        )
                    except Exception as exc:
                        st.warning(f"Không lưu được data/results2.csv: {exc}")
                except Exception as exc:
                    st.error(f"Lỗi khi tính thống kê cơ bản: {exc}")

                st.divider()

                # --- 2. Phim cao nhất / thấp nhất theo chỉ số ---
                st.subheader("2. Phim cao nhất / thấp nhất theo chỉ số")
                column = st.selectbox("Chọn chỉ số", stat_cols)
                try:
                    top_movies, bottom_movies = get_top_bottom_movies(
                        df, column, top_n=3
                    )
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.subheader(f"Top 3 cao nhất ({column})")
                        st.dataframe(top_movies, use_container_width=True)
                    with col_b:
                        st.subheader(f"Top 3 thấp nhất ({column})")
                        st.dataframe(bottom_movies, use_container_width=True)
                except Exception as exc:
                    st.error(f"Lỗi khi tìm phim cực trị: {exc}")

                st.divider()

                # --- 3. Biểu đồ phân phối ---
                st.subheader(f"3. Biểu đồ phân phối: {column}")
                try:
                    fig = plot_histogram(df, column)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Lỗi khi vẽ biểu đồ: {exc}")


elif section == "Phần III: Học máy - Phân cụm & So sánh":
    st.header("Phần III: Phân cụm và biểu đồ radar")
    # Ưu tiên DataFrame vừa làm sạch trong phiên hiện tại. Điều này tránh
    # dùng lại file cache cũ khi người dùng vừa chạy lại Phần I.
    df = get_data()
    if df is None:
        st.error(
            "Chưa tìm thấy data/cleaned/results.csv. "
            "Hãy chạy phần làm sạch dữ liệu trước."
        )
    else:
        # Kiểm tra số mẫu hợp lệ để giới hạn n_clusters. Không che nguyên
        # nhân gốc (ví dụ thiếu cột) bằng thông báo "0 mẫu hợp lệ".
        try:
            X_valid = _prepare_cluster_features(df)
        except (TypeError, ValueError) as exc:
            st.error(f"Không thể chuẩn bị dữ liệu phân cụm: {exc}")
            st.stop()
        n_valid = len(X_valid)
        max_clusters = min(10, n_valid)
        if max_clusters < 2:
            st.error(
                f"Không đủ dữ liệu để phân cụm (cần ít nhất 2 mẫu hợp lệ, "
                f"hiện chỉ có {n_valid}). Hãy kiểm tra dữ liệu đầu vào."
            )
            st.stop()

        st.caption(
            f"Phân cụm dùng {n_valid:,}/{len(df):,} phim có đủ Kinh phí, "
            "Doanh thu và Điểm TMDB."
        )
        n_clusters = st.slider(
            "Số cụm (n_clusters)",
            2,
            max_clusters,
            min(3, max_clusters),
            key="n_clusters",
        )
        try:
            clustered = run_kmeans_clustering(df, n_clusters)
            pca_coords = run_pca_reduction(df)
        except ValueError as exc:
            st.error(f"Không chạy được phân cụm: {exc}")
        else:
            # Gộp tên phim + nhãn cụm + tọa độ PCA (khóa chung: index)
            table = (
                df.loc[clustered.index, ["Title"]]
                .join(clustered)
                .join(pca_coords)
            )
            counts = (
                clustered["Cluster"]
                .value_counts()
                .sort_index()
                .rename_axis("Cluster")
                .reset_index(name="Số phim")
            )
            st.subheader("1. Kết quả phân cụm")

            def find_cluster_column(*candidates):
                return next((col for col in candidates if col in clustered.columns), None)

            rating_col = find_cluster_column("Rating", "IMDb_Rating")
            budget_col = find_cluster_column("Budget", "Budget_USD")
            revenue_col = find_cluster_column("Revenue", "Revenue_USD")
            vote_col = find_cluster_column("Votes", "Votes")
            cluster_summary = clustered.groupby("Cluster").agg(
                Số_phim=("Cluster", "size"),
                Rating_trung_vị=(rating_col, "median"),
                Kinh_phí_trung_vị=(budget_col, "median"),
                Doanh_thu_trung_vị=(revenue_col, "median"),
            ).reset_index()
            cluster_summary["Rating_trung_vị"] = cluster_summary[
                "Rating_trung_vị"
            ].map(lambda value: f"{value:.1f}".replace(".", ",") + " / 10")
            for column in ("Kinh_phí_trung_vị", "Doanh_thu_trung_vị"):
                cluster_summary[column] = cluster_summary[column].map(format_usd)

            # Bản đồ cụm 2D sau khi giảm chiều PCA
            ev = pca_coords.attrs["explained_variance_ratio"]
            fig, ax = plt.subplots(figsize=(8, 5.5))
            palette = plt.get_cmap("tab10", n_clusters)
            for cluster_id in sorted(clustered["Cluster"].unique()):
                mask = clustered.loc[pca_coords.index, "Cluster"] == cluster_id
                ax.scatter(
                    pca_coords.loc[mask, pca_coords.columns[0]],
                    pca_coords.loc[mask, pca_coords.columns[1]],
                    color=palette(cluster_id),
                    s=16,
                    alpha=0.72,
                    label=f"Cụm {cluster_id} · {int(mask.sum()):,} phim",
                )
            ax.set_xlabel(f"PCA 1 · giải thích {ev[0]:.1%} phương sai")
            ax.set_ylabel(f"PCA 2 · giải thích {ev[1]:.1%} phương sai")
            ax.set_title(f"Bản đồ phân cụm phim (K = {n_clusters})")
            ax.grid(alpha=0.25)
            ax.legend(title="Nhóm phim", frameon=False, loc="best")
            fig.tight_layout()

            map_column, summary_column = st.columns([1.2, 0.8])
            with map_column:
                st.caption(
                    "Mỗi điểm là một phim; vị trí gần nhau biểu thị hồ sơ "
                    "Budget, Revenue và Rating tương đồng."
                )
                st.pyplot(fig, use_container_width=True)
            with summary_column:
                st.caption("Tóm tắt theo cụm bằng số liệu gốc (trung vị).")
                st.dataframe(cluster_summary, hide_index=True, use_container_width=True)
            plt.close(fig)

            with st.expander("Xem bảng dữ liệu chi tiết (toàn bộ)"):
                st.dataframe(table, use_container_width=True)


            st.subheader("2. So sánh hai phim bằng biểu đồ radar")

            radar_df = df.dropna(
                subset=["IMDb_Rating", "Votes", "Budget_USD", "Revenue_USD", "Duration_min"]
            ).copy()

            # Hàm radar chủ động báo lỗi với title trùng; loại khỏi danh sách chọn UI.
            duplicate_titles = radar_df["Title"].duplicated(keep=False)
            movie_options = sorted(
                radar_df.loc[~duplicate_titles, "Title"]
                .astype(str)
                .str.strip()
                .unique()
            )

            col1, col2 = st.columns(2)
            with col1:
                movie1 = st.selectbox("Phim thứ nhất", movie_options, key="radar_movie_1")
            with col2:
                movie2 = st.selectbox(
                    "Phim thứ hai",
                    movie_options,
                    index=1,
                    key="radar_movie_2",
                )

            if movie1 == movie2:
                st.warning("Hãy chọn hai phim khác nhau.")
            else:
                try:
                    fig = plot_radar_chart(df, movie1, movie2)
                    chart_column, detail_column = st.columns([1.2, 0.8])
                    with chart_column:
                        st.caption(
                            "Radar thể hiện mức tương đối 0–100. "
                            "Xem số liệu gốc có đơn vị ở bên phải."
                        )
                        st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    with detail_column:
                        # Chỉ thu gọn các metric thuộc khối dữ liệu radar;
                        # metric trên trang Tổng quan vẫn giữ kích thước lớn.
                        with st.container(key="radar_details"):
                            st.markdown(
                                """
                                <style>
                                .st-key-radar_details div[data-testid="stMetric"] {
                                    min-height: 0;
                                    padding: 0.7rem 0.9rem 0.75rem;
                                }
                                .st-key-radar_details div[data-testid="stMetricLabel"] {
                                    font-size: 0.82rem;
                                }
                                .st-key-radar_details div[data-testid="stMetricValue"] {
                                    font-size: clamp(1.1rem, 2.2vw, 1.42rem);
                                    line-height: 1.2;
                                    white-space: normal;
                                    overflow-wrap: anywhere;
                                }
                                .st-key-radar_details div[data-testid="stMetricDelta"] {
                                    display: none;
                                }
                                .st-key-radar_details hr {
                                    margin: 1rem 0;
                                }
                                </style>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.markdown("#### Dữ liệu gốc")
                            selected_rows = []
                            titles = radar_df["Title"].astype(str).str.strip()
                            for name in (movie1, movie2):
                                selected_rows.append(
                                    radar_df.loc[titles == name].iloc[0]
                                )

                            for icon, movie, row in zip(
                                ("🔵", "🔴"),
                                (movie1, movie2),
                                selected_rows,
                            ):
                                st.markdown(
                                    f"**{icon} {movie} ({format_year(row.get('Year'))})**"
                                )
                                metric_left, metric_right = st.columns(2)
                                metric_left.metric(
                                    "Điểm TMDB",
                                    f"{row['IMDb_Rating']:.1f}".replace(".", ",") + " / 10",
                                )
                                metric_right.metric(
                                    "Thời lượng",
                                    f"{row['Duration_min']:,.0f}".replace(",", ".") + " phút",
                                )
                                metric_left, metric_right = st.columns(2)
                                metric_left.metric("Kinh phí", format_usd(row["Budget_USD"]))
                                metric_right.metric("Doanh thu", format_usd(row["Revenue_USD"]))
                                st.metric(
                                    "Lượt đánh giá",
                                    f"{row['Votes']:,.0f}".replace(",", "."),
                                )
                                st.divider()
                except ValueError as exc:
                    st.error(str(exc))


elif section == "Phần IV: Dự đoán Doanh thu phòng vé":
    st.header("Phần IV: Dự đoán doanh thu")
    df = get_data()
    if df is None:
        st.error(
            "Chưa tìm thấy data/cleaned/results.csv. "
            "Hãy chạy phần làm sạch dữ liệu trước."
        )
    else:
        # Chỉ huấn luyện khi bấm nút; kết quả giữ trong session
        # để không train lại sau mỗi lần tương tác giao diện.
        if st.button("Huấn luyện mô hình (RandomForest)"):
            with st.spinner("Đang huấn luyện mô hình..."):
                try:
                    model = train_revenue_prediction_model(df)
                except ValueError as exc:
                    st.error(f"Không huấn luyện được: {exc}")
                else:
                    st.session_state["revenue_model"] = model
                    st.success("Huấn luyện hoàn tất!")

        model = st.session_state.get("revenue_model")
        if model is None:
            st.info(
                "Bấm nút bên trên để huấn luyện mô hình. Mô hình "
                "chỉ train một lần cho mỗi phiên làm việc."
            )
        else:
            metrics = model.metrics_
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"${metrics['MAE']:,.0f}")
            col2.metric("RMSE", f"${metrics['RMSE']:,.0f}")
            col3.metric("R²", f"{metrics['R2']:.3f}")
            col4.metric("Số mẫu test", f"{metrics['n_test']:,}")
            st.caption(
                "Đặc trưng đầu vào: " + ", ".join(model.feature_columns_)
            )
            if hasattr(model, "baseline_") and model.baseline_:
                b = model.baseline_
                st.caption(
                    f"Baseline: R² trung bình = {b['mean_R2']:.3f}, "
                    f"R² trung vị = {b['median_R2']:.3f} "
                    f"(mô hình R² = {metrics['R2']:.3f})"
                )

            st.subheader("Dự đoán doanh thu cho phim mới")
            with st.form("revenue_form"):
                budget = st.number_input(
                    "Kinh phí (USD)",
                    min_value=0.0,
                    value=10_000_000.0,
                    step=500_000.0,
                )
                rating = st.number_input(
                    "Điểm IMDb", 0.0, 10.0, 7.0, 0.1
                )
                duration = st.number_input(
                    "Thời lượng (phút)", 0, 300, 120
                )
                genre = st.text_input("Thể loại (Genre)", "Drama")
                # Ô Votes chỉ xuất hiện nếu mô hình dùng cột này
                votes = 0
                if "Votes" in model.feature_columns_ or "log_Votes" in model.feature_columns_:
                    votes = st.number_input(
                        "Số lượt đánh giá (Votes)", 0, 10_000_000, 1_000
                    )
                submitted = st.form_submit_button("Dự đoán doanh thu")

            if submitted:
                row = {
                    "Budget_USD": budget,
                    "IMDb_Rating": rating,
                    "Duration_min": duration,
                    "Genre": genre,
                    "Votes": votes,
                }
                # Dự đoán qua predict_revenue (xử lý log-transform nội bộ)
                prediction = float(predict_revenue(model, pd.DataFrame([row]))[0])
                st.success(
                    f"Doanh thu dự đoán: **${prediction:,.0f}** USD"
                )
