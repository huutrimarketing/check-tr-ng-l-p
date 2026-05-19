import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import time
import io

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="Wiki Duplicate Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tối thiểu — chỉ style header banner và nút bấm
st.markdown("""
<style>
    .banner {
        background: linear-gradient(90deg, #1e40af, #2563eb);
        color: white; padding: 1.2rem 1.8rem; border-radius: 10px;
        margin-bottom: 1.2rem;
    }
    .banner h2 { margin: 0; font-size: 1.5rem; font-weight: 700; }
    .banner p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# KHỞI TẠO SESSION STATE
# ============================================================
for key, default in {
    "running": False,
    "paused": False,
    "stop_requested": False,
    "current_index": 0,
    "toc_data": [],
    "valid_urls": [],
    "all_urls": [],
    "results": None,
    "delay": 1.0,
    "threshold": 0.75,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Safety reset khi hot-reload làm mất session state
_req = ["toc_data", "valid_urls", "all_urls"]
if st.session_state.running and any(k not in st.session_state for k in _req):
    for k in ["running", "paused", "stop_requested"]:
        st.session_state[k] = False
    for k in _req:
        st.session_state[k] = []
    st.session_state.current_index = 0
    st.session_state.results = None

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="banner">
    <h2>🔍 Wiki Duplicate Checker</h2>
    <p>Phát hiện nội dung trùng lặp bằng AI hai lớp: TF-IDF lọc nhanh + SBERT phân tích ngữ nghĩa sâu</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HÀM XỬ LÝ
# ============================================================
def get_urls_from_sitemap(sitemap_url):
    try:
        resp = requests.get(sitemap_url, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml-xml")
        return [loc.text.strip() for loc in soup.find_all("loc")]
    except Exception as e:
        st.error(f"❌ Lỗi đọc sitemap: {e}")
        return []

def get_urls_from_sheet(sheet_url, col_index):
    try:
        match_id  = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", sheet_url)
        match_gid = re.search(r"gid=(\d+)", sheet_url)
        if not match_id:
            st.error("❌ Link Google Sheet không hợp lệ!")
            return []
        sheet_id = match_id.group(1)
        gid = match_gid.group(1) if match_gid else "0"
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv_url, header=0)
        urls = df.iloc[:, col_index].dropna().astype(str).tolist()
        return [u.strip() for u in urls if u.strip().startswith("http")]
    except Exception as e:
        st.error(f"❌ Lỗi đọc Google Sheet: {e}")
        st.warning("⚠️ Đảm bảo Google Sheet chia sẻ 'Anyone with the link - Viewer'")
        return []

def extract_toc(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=15, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        h1 = soup.find("h1")
        h1_text = (h1.get_text(strip=True) + " ") * 3 if h1 else ""
        headings = soup.find_all(["h2", "h3"])
        toc_text = (" ".join([h.get_text(strip=True) for h in headings]) + " ") * 2
        content_area = (
            soup.find("div", class_="entry-content") or
            soup.find("div", class_="post-content") or
            soup.find("article") or soup.find("main")
        )
        intro_text = ""
        if content_area:
            for tag in content_area.find_all(["h1", "h2", "h3", "h4"]):
                tag.decompose()
            words = content_area.get_text(separator=" ", strip=True).split()
            intro_text = " ".join(words[:200])
        combined = h1_text + toc_text + intro_text
        return re.sub(r"[^\w\s]", " ", combined).lower().strip()
    except:
        return ""

# ============================================================
# DANH SÁCH TỈNH/THÀNH PHỐ VIỆT NAM
# ============================================================
VIETNAM_LOCATIONS = [
    "ha-noi", "ho-chi-minh", "tp-hcm", "tphcm", "sai-gon",
    "da-nang", "hai-phong", "can-tho",
    "an-giang", "ba-ria-vung-tau", "vung-tau",
    "bac-giang", "bac-kan", "bac-lieu", "bac-ninh",
    "ben-tre", "binh-dinh", "binh-duong", "binh-phuoc", "binh-thuan",
    "ca-mau", "cao-bang", "dak-lak", "dak-nong", "dien-bien",
    "dong-nai", "dong-thap", "gia-lai",
    "ha-giang", "ha-nam", "ha-tinh", "hai-duong", "hau-giang",
    "hoa-binh", "hung-yen", "khanh-hoa", "kien-giang", "kon-tum",
    "lai-chau", "lam-dong", "lang-son", "lao-cai", "long-an",
    "nam-dinh", "nghe-an", "ninh-binh", "ninh-thuan", "phu-tho", "phu-yen",
    "quang-binh", "quang-nam", "quang-ngai", "quang-ninh", "quang-tri",
    "soc-trang", "son-la", "tay-ninh", "thai-binh", "thai-nguyen",
    "thanh-hoa", "thua-thien-hue", "hue", "tien-giang", "tra-vinh",
    "tuyen-quang", "vinh-long", "vinh-phuc", "yen-bai"
]

def is_local_duplicate(url_a, url_b):
    slug_a = url_a.rstrip("/").split("/")[-1].lower()
    slug_b = url_b.rstrip("/").split("/")[-1].lower()
    clean_a, clean_b = slug_a, slug_b
    loc_a, loc_b = None, None
    for loc in VIETNAM_LOCATIONS:
        if loc in slug_a:
            clean_a = slug_a.replace(loc, "").replace("--", "-").strip("-")
            loc_a = loc
        if loc in slug_b:
            clean_b = slug_b.replace(loc, "").replace("--", "-").strip("-")
            loc_b = loc
    if loc_a and loc_b and loc_a != loc_b:
        words_a = set(clean_a.split("-")) - {""}
        words_b = set(clean_b.split("-")) - {""}
        if not words_a or not words_b:
            return False
        return len(words_a & words_b) / len(words_a | words_b) >= 0.65
    return False

@st.cache_resource
def load_sbert_model():
    # paraphrase-multilingual-MiniLM-L12-v2: nhẹ ~120MB, vẫn hỗ trợ tiếng Việt tốt
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def run_similarity_analysis(toc_data, valid_urls, threshold):
    # Lớp 1: TF-IDF lọc nhanh
    vectorizer    = TfidfVectorizer()
    tfidf_matrix  = vectorizer.fit_transform(toc_data)
    tfidf_sim_mat = cosine_similarity(tfidf_matrix)
    candidates = []
    for i in range(len(valid_urls)):
        for j in range(i + 1, len(valid_urls)):
            if tfidf_sim_mat[i][j] >= 0.15:
                candidates.append((i, j, float(tfidf_sim_mat[i][j])))
    if not candidates:
        return []

    # Lớp 2: SBERT phân tích ngữ nghĩa
    model = load_sbert_model()
    needed_idx = sorted(set(i for i, j, _ in candidates) | set(j for i, j, _ in candidates))
    embeddings = model.encode(
        [toc_data[i] for i in needed_idx],
        show_progress_bar=False,
        batch_size=32
    )
    embed_map = {idx: emb for idx, emb in zip(needed_idx, embeddings)}

    results = []
    for i, j, tfidf_score in candidates:
        sbert_score = float(cosine_similarity(
            embed_map[i].reshape(1, -1),
            embed_map[j].reshape(1, -1)
        )[0][0])
        combined = round(tfidf_score * 0.3 + sbert_score * 0.7, 4)
        if combined >= threshold:
            level = (
                "🔴 Rất cao" if combined >= 0.85 else
                "🟠 Cao"     if combined >= 0.70 else
                "🟡 Trung bình"
            )
            loai = "🏙️ Local Duplicate" if is_local_duplicate(valid_urls[i], valid_urls[j]) else "📄 Content Duplicate"
            results.append({
                "Bài viết A":        valid_urls[i],
                "Bài viết B":        valid_urls[j],
                "Loại":              loai,
                "TF-IDF (%)":        round(tfidf_score * 100, 1),
                "SBERT (%)":         round(sbert_score * 100, 1),
                "Điểm tổng hợp (%)": round(combined * 100, 1),
                "Cảnh báo":          level,
                "Đề xuất":           "Xóa + Redirect 301" if "Local" in loai else "Gộp bài + Redirect 301"
            })
    return results

# ============================================================
# HIỂN THỊ KẾT QUẢ
# ============================================================
def show_results(results, valid_urls):
    st.divider()
    st.subheader("📊 Kết quả phân tích")

    total      = len(results)
    high_risk  = sum(1 for r in results if "🔴" in r.get("Cảnh báo", ""))
    local_dup  = sum(1 for r in results if "🏙️" in r.get("Loại", ""))
    dup_rate   = round(total / max(len(valid_urls), 1) * 100, 1)

    # Metric cards — native Streamlit
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Bài đã phân tích", len(valid_urls))
    c2.metric("⚠️ Cặp trùng lặp",   total)
    c3.metric("🔴 Nguy cơ cao",      high_risk)
    c4.metric("📊 Tỷ lệ trùng lặp", f"{dup_rate}%")

    if total > 0:
        st.caption(f"🏙️ Local Duplicate: **{local_dup}** cặp &nbsp;|&nbsp; 📄 Content Duplicate: **{total - local_dup}** cặp")

    st.write("")
    if results:
        df = pd.DataFrame(results).sort_values("Điểm tổng hợp (%)", ascending=False)
        st.dataframe(
            df,
            use_container_width=True,
            height=min(500, 60 + len(df) * 38),
            column_config={
                "Bài viết A": st.column_config.LinkColumn("Bài viết A", width="large"),
                "Bài viết B": st.column_config.LinkColumn("Bài viết B", width="large"),
                "Điểm tổng hợp (%)": st.column_config.ProgressColumn(
                    "Điểm tổng hợp", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
            hide_index=True
        )
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)
        st.download_button(
            "⬇️ Tải báo cáo Excel",
            data=buf,
            file_name="duplicate_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.success("🎉 Không tìm thấy bài viết nào bị trùng lặp!")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙️ Cài đặt")

    # --- Nguồn dữ liệu ---
    st.subheader("📂 Nguồn dữ liệu")
    source = st.radio("Chọn nguồn:", ["🗺️ Sitemap XML", "📊 Google Sheet"], label_visibility="collapsed")

    sitemap_url = ""
    sheet_url   = ""
    col_index   = 1

    if source == "🗺️ Sitemap XML":
        sitemap_url = st.text_input(
            "URL Sitemap",
            value="https://homenest.com.vn/post-sitemap.xml"
        )
    else:
        sheet_url = st.text_input(
            "Link Google Sheet",
            placeholder="https://docs.google.com/spreadsheets/d/..."
        )
        col_index = st.number_input("Cột URL (0=A, 1=B...)", min_value=0, value=1)

    st.divider()

    # --- Cấu hình phân tích ---
    st.subheader("🔧 Cấu hình")
    threshold = st.slider("Ngưỡng cảnh báo (%)", 50, 95, 75, 5)
    max_urls  = st.number_input("Giới hạn URL", min_value=10, max_value=5000, value=300, step=50)
    delay     = st.slider("Delay giữa request (s)", 0.5, 3.0, 1.0, 0.5)

    st.divider()

    # --- Tùy chọn ---
    st.subheader("🔀 Tùy chọn")
    hide_local = st.checkbox(
        "Ẩn Local Duplicate",
        value=False,
        help="Ẩn các cặp trang địa phương theo mẫu, chỉ hiển thị Content Duplicate"
    )

    enable_replace = st.checkbox("Chuyển đổi domain", value=False)
    if enable_replace:
        domain_from = st.text_input("Từ domain", value="crm.homenest.com.vn")
        domain_to   = st.text_input("Sang domain", value="homenest.com.vn")
    else:
        domain_from = ""
        domain_to   = ""

# ============================================================
# NÚT ĐIỀU KHIỂN
# ============================================================
st.write("")
col1, col2, col3, _ = st.columns([1.5, 1.5, 1.5, 3])

with col1:
    start_btn = st.button(
        "🚀 Bắt đầu phân tích",
        disabled=st.session_state.running or st.session_state.paused,
        type="primary",
        use_container_width=True
    )
with col2:
    stop_btn = st.button(
        "⏹ Dừng",
        disabled=not st.session_state.running,
        use_container_width=True
    )
with col3:
    resume_btn = st.button(
        "▶ Tiếp tục",
        disabled=not st.session_state.paused,
        use_container_width=True
    )

# ============================================================
# XỬ LÝ NÚT
# ============================================================
if stop_btn:
    st.session_state.stop_requested = True

if resume_btn and st.session_state.paused:
    st.session_state.paused        = False
    st.session_state.running       = True
    st.session_state.stop_requested = False
    st.rerun()

if start_btn:
    if source == "🗺️ Sitemap XML" and not sitemap_url.strip():
        st.error("❌ Vui lòng nhập URL Sitemap!")
        st.stop()
    if source == "📊 Google Sheet" and not sheet_url.strip():
        st.error("❌ Vui lòng nhập link Google Sheet!")
        st.stop()

    with st.spinner("📥 Đang lấy danh sách URL..."):
        if source == "🗺️ Sitemap XML":
            urls = get_urls_from_sitemap(sitemap_url)
        else:
            urls = get_urls_from_sheet(sheet_url, int(col_index))

    if not urls:
        st.error("❌ Không lấy được URL nào. Kiểm tra lại link và quyền truy cập!")
        st.stop()

    if enable_replace and domain_from.strip() and domain_to.strip():
        urls = [u.replace(domain_from.strip(), domain_to.strip()) for u in urls]
        st.info(f"🔄 Đã chuyển đổi domain: `{domain_from}` → `{domain_to}`")

    urls = urls[:int(max_urls)]
    st.session_state.all_urls       = urls
    st.session_state.toc_data       = []
    st.session_state.valid_urls     = []
    st.session_state.current_index  = 0
    st.session_state.results        = None
    st.session_state.running        = True
    st.session_state.stop_requested = False
    st.session_state.delay          = delay
    st.session_state.threshold      = threshold / 100
    st.success(f"✅ Tìm thấy **{len(urls)}** URL. Bắt đầu phân tích!")
    st.rerun()

# ============================================================
# VÒNG LẶP CÀO DỮ LIỆU
# ============================================================
if st.session_state.running:
    urls  = st.session_state.all_urls
    idx   = st.session_state.current_index
    total = len(urls)

    progress_bar = st.progress(idx / total if total > 0 else 0,
                               text=f"Đang xử lý: {idx}/{total} bài...")
    status_text  = st.empty()

    if st.session_state.stop_requested:
        st.session_state.running        = False
        st.session_state.paused         = True
        st.session_state.stop_requested = False
        st.warning(f"⏸ Tạm dừng tại **{idx}/{total}**. Đã cào **{len(st.session_state.valid_urls)}** bài.")

        if len(st.session_state.valid_urls) >= 2:
            with st.spinner("🧠 Đang tính toán kết quả tạm thời..."):
                st.session_state.results = run_similarity_analysis(
                    st.session_state.toc_data,
                    st.session_state.valid_urls,
                    st.session_state.threshold
                )
        st.rerun()

    elif idx < total:
        url = urls[idx]
        toc = extract_toc(url)
        if toc:
            st.session_state.toc_data.append(toc)
            st.session_state.valid_urls.append(url)
        st.session_state.current_index += 1
        slug = url.rstrip("/").split("/")[-1] or url
        status_text.caption(f"[{idx+1}/{total}] {'✅' if toc else '⚠️ Bỏ qua:'} {slug}")
        progress_bar.progress((idx + 1) / total, text=f"Đang xử lý: {idx+1}/{total} bài...")
        time.sleep(st.session_state.delay)
        st.rerun()

    else:
        st.session_state.running = False
        progress_bar.progress(1.0, text=f"✅ Hoàn thành {total}/{total} bài!")

        if len(st.session_state.valid_urls) >= 2:
            with st.spinner("🧠 Đang phân tích ngữ nghĩa AI (SBERT)..."):
                st.session_state.results = run_similarity_analysis(
                    st.session_state.toc_data,
                    st.session_state.valid_urls,
                    st.session_state.threshold
                )
        st.session_state.paused = False
        st.rerun()

# ============================================================
# HIỂN THỊ KẾT QUẢ
# ============================================================
if st.session_state.results is not None:
    results_to_show = st.session_state.results

    local_count   = sum(1 for r in results_to_show if "🏙️" in r.get("Loại", ""))
    content_count = len(results_to_show) - local_count

    if hide_local:
        results_to_show = [r for r in results_to_show if "🏙️" not in r.get("Loại", "")]
        if local_count > 0:
            st.info(f"🏙️ Đã ẩn **{local_count}** cặp Local Duplicate. Hiển thị **{content_count}** cặp Content Duplicate.")
    elif local_count > 0:
        st.warning(f"🏙️ Phát hiện **{local_count}** cặp Local Duplicate (trang địa phương theo mẫu). Bật 'Ẩn Local Duplicate' trong sidebar nếu muốn lọc.")

    show_results(results_to_show, st.session_state.valid_urls)
