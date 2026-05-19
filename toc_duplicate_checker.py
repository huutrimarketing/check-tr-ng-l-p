import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
import re

# ============================================================
# CẤU HÌNH - Chỉnh sửa các thông số ở đây
# ============================================================
OUTPUT_FILE = "toc_duplicate_report.xlsx"   # Tên file báo cáo xuất ra
SIMILARITY_THRESHOLD = 0.75                 # Ngưỡng cảnh báo trùng lặp (75%)
DELAY_BETWEEN_REQUESTS = 1                  # Thời gian chờ giữa mỗi lần cào (giây)
MAX_URLS = 500                              # Giới hạn số URL tối đa (đặt 0 = không giới hạn)

# Thông số mặc định (dùng khi chọn nguồn tự động)
DEFAULT_SITEMAP_URL = "https://homenest.com.vn/post-sitemap.xml"
# URL trang tính Google Sheet của bạn (dạng link chia sẻ thông thường)
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1-1h1yQZxaZeKyzf57hNYGMxfomHZcPIOb4lB8xZFIiI/edit?gid=821837017#gid=821837017"
# Cột chứa URL trong Google Sheet (0 = cột A, 1 = cột B, ...)
URL_COLUMN_INDEX = 1  # Cột B
# ============================================================


def get_urls_from_sitemap(sitemap_url):
    """Lấy toàn bộ danh sách URL từ Sitemap XML"""
    print(f"\n📥 Đang đọc sitemap: {sitemap_url}")
    try:
        resp = requests.get(sitemap_url, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml-xml")
        urls = [loc.text.strip() for loc in soup.find_all("loc")]
        print(f"✅ Tìm thấy {len(urls)} URL trong sitemap.")
        return urls
    except Exception as e:
        print(f"❌ Lỗi đọc sitemap: {e}")
        return []


def get_urls_from_google_sheet(sheet_url, url_column_index):
    """Lấy danh sách URL từ Google Sheet (Sheet phải được chia sẻ công khai)"""
    print(f"\n📥 Đang đọc Google Sheet...")

    try:
        # Chuyển đổi link Google Sheet sang link xuất CSV
        # Ví dụ: .../spreadsheets/d/{ID}/edit?gid={GID} -> .../spreadsheets/d/{ID}/export?format=csv&gid={GID}
        match_id = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", sheet_url)
        match_gid = re.search(r"gid=(\d+)", sheet_url)

        if not match_id:
            print("❌ Link Google Sheet không hợp lệ!")
            return []

        sheet_id = match_id.group(1)
        gid = match_gid.group(1) if match_gid else "0"

        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        print(f"   Đang tải dữ liệu từ: {csv_url}")

        df = pd.read_csv(csv_url, header=0)

        # Lấy cột URL và bỏ các dòng trống
        urls = df.iloc[:, url_column_index].dropna().astype(str).tolist()
        urls = [u.strip() for u in urls if u.strip().startswith("http")]

        print(f"✅ Tìm thấy {len(urls)} URL trong Google Sheet.")
        return urls

    except Exception as e:
        print(f"❌ Lỗi đọc Google Sheet: {e}")
        print("   ⚠️  Hãy đảm bảo trang tính đã được chia sẻ 'Bất kỳ ai có đường liên kết'!")
        return []


def extract_toc(url):
    """Cào và trích xuất nội dung TOC (H2, H3) từ bài viết"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=15, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Lấy toàn bộ thẻ H2 và H3 trong bài viết
        headings = soup.find_all(["h2", "h3"])
        toc_text = " ".join([h.get_text(strip=True) for h in headings])

        # Làm sạch văn bản
        toc_text = re.sub(r"[^\w\s]", " ", toc_text).lower().strip()
        return toc_text

    except Exception as e:
        return ""


def choose_input_source():
    """Hiển thị menu lựa chọn nguồn dữ liệu đầu vào"""
    print("\n" + "=" * 60)
    print("  CHỌN NGUỒN DỮ LIỆU ĐẦU VÀO")
    print("=" * 60)
    print("  [1] Sitemap XML  (tự động quét toàn bộ website)")
    print("  [2] Google Sheet (lấy từ danh sách URL có sẵn)")
    print("=" * 60)

    while True:
        choice = input("\nBạn chọn (1 hoặc 2): ").strip()

        if choice == "1":
            sitemap = input(f"Nhập URL sitemap (Enter để dùng mặc định): ").strip()
            if not sitemap:
                sitemap = DEFAULT_SITEMAP_URL
            return get_urls_from_sitemap(sitemap)

        elif choice == "2":
            sheet = input(f"Nhập link Google Sheet (Enter để dùng mặc định): ").strip()
            if not sheet:
                sheet = DEFAULT_SHEET_URL
            return get_urls_from_google_sheet(sheet, URL_COLUMN_INDEX)

        else:
            print("❌ Vui lòng chỉ nhập 1 hoặc 2!")


def main():
    print("=" * 60)
    print("🔍 CÔNG CỤ KIỂM TRA TRÙNG LẶP NỘI DUNG (DỰA TRÊN TOC)")
    print("=" * 60)

    # Bước 1: Chọn nguồn dữ liệu
    urls = choose_input_source()

    if not urls:
        print("Không có URL để xử lý. Thoát.")
        return

    # Giới hạn số URL nếu cần
    if MAX_URLS > 0 and len(urls) > MAX_URLS:
        urls = urls[:MAX_URLS]
        print(f"⚠️  Giới hạn xử lý tối đa {MAX_URLS} URL.")

    # Bước 2: Cào TOC từng bài viết
    print(f"\n🕷️  Đang cào TOC từ {len(urls)} bài viết...")
    toc_data = []
    valid_urls = []

    for i, url in enumerate(urls):
        toc = extract_toc(url)
        if toc:
            toc_data.append(toc)
            valid_urls.append(url)

        pct = round((i + 1) / len(urls) * 100)
        label = url.split("/")[-2] or url
        print(f"  [{i+1}/{len(urls)}] {pct}% - {'✅' if toc else '⚠️ Không có TOC'} {label}")
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\n✅ Đã cào xong {len(valid_urls)} bài viết có TOC hợp lệ.")

    if len(valid_urls) < 2:
        print("❌ Cần ít nhất 2 bài viết để so sánh. Thoát.")
        return

    # Bước 3: Phân tích ngữ nghĩa
    print("\n🧠 Đang phân tích ngữ nghĩa và tính toán độ tương đồng...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(toc_data)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Bước 4: Lọc các cặp bài viết trùng lặp
    print(f"🔎 Đang lọc các cặp có độ giống nhau > {int(SIMILARITY_THRESHOLD * 100)}%...")
    results = []

    for i in range(len(valid_urls)):
        for j in range(i + 1, len(valid_urls)):
            score = similarity_matrix[i][j]
            if score >= SIMILARITY_THRESHOLD:
                if score >= 0.90:
                    level = "🔴 Rất cao - Cần gộp ngay"
                elif score >= 0.80:
                    level = "🟠 Cao - Xem xét gộp bài"
                else:
                    level = "🟡 Trung bình - Theo dõi"

                results.append({
                    "Bài viết A": valid_urls[i],
                    "Bài viết B": valid_urls[j],
                    "Độ giống nhau (%)": round(score * 100, 2),
                    "Mức độ cảnh báo": level,
                    "Đề xuất hành động": "Gộp bài (Merge) + Redirect 301"
                })

    # Bước 5: Xuất báo cáo
    print(f"\n📊 Tìm thấy {len(results)} cặp bài viết có nội dung trùng lặp.")

    if results:
        df = pd.DataFrame(results)
        df.sort_values(by="Độ giống nhau (%)", ascending=False, inplace=True)
        df.to_excel(OUTPUT_FILE, index=False)
        print(f"✅ Đã lưu báo cáo tại: {OUTPUT_FILE}")
    else:
        print("🎉 Không tìm thấy bài viết nào bị trùng lặp!")

    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)


if __name__ == "__main__":
    main()
