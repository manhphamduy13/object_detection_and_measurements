# HỆ THỐNG PHÂN LOẠI & ĐO ĐẠC KÍCH THƯỚC QUANG HỌC QC_CAM (100% THUẬT TOÁN HÌNH HỌC - 0% AI)

Hệ thống thị giác máy tính công nghiệp chuyên dụng cho dây chuyền kiểm định chất lượng (QC) linh kiện cơ khí chính xác, tự động hóa từ khâu đọc bản vẽ kỹ thuật 2D CAD/PDF đến đo đạc subpixel và đánh giá dung sai xuất xưởng.

---

## 1. Cấu Trúc Dự Án Sau Khi Chuẩn Hóa (Clean Architecture)

Toàn bộ dự án đã được tinh gọn và tổ chức lại theo cấu trúc công nghiệp tiêu chuẩn:

```
object-classification/
├── assets/
│   ├── references/               # Mẫu chuẩn đối sánh (Golden Samples, spec.json, drawing.png)
│   │   ├── 98003P/
│   │   ├── 98475/
│   │   ├── 98661BBS-1/
│   │   ├── UH-004/
│   │   └── UH-8715P/
│   └── tests/                    # Tập ảnh phôi thực tế trên dây chuyền kiểm thử (64 ảnh)
│       ├── 98003P/
│       ├── 98475/
│       ├── 98661BBS-1/
│       ├── UH-004/
│       └── UH-8715P/
├── configs/
│   └── products.json             # Cơ sở dữ liệu cấu hình hệ số quang học & dung sai toàn hệ thống
├── items/                        # Bản vẽ kỹ thuật 2D gốc dạng PDF (98003P.pdf, 98475.pdf,...)
├── outputs/                      # Thư mục xuất kết quả kiểm tra
│   ├── visualizations/           # Ảnh đo Caliper (*_measured.png) & ảnh tổng hợp 6 panel (*.png)
│   ├── steps/                    # 5 bước bóc tách trung gian chi tiết theo từng sản phẩm
│   ├── predictions/              # File JSON báo cáo chi tiết theo từng sản phẩm (*.json)
│   ├── evaluation.csv            # Bảng CSV thống kê kiểm định toàn hệ thống
│   ├── predictions.json          # File JSON tổng hợp kết quả toàn bộ dây chuyền
│   └── prediction_measured.png   # Ảnh xem nhanh kết quả kiểm tra gần nhất
├── src/                          # THƯ VIỆN LÕI CÔNG NGHỆ (CORE INSPECTION ENGINE)
│   ├── __init__.py
│   ├── config.py                 # Cấu hình tham số lọc nhiễu, ngưỡng và đường dẫn tập trung
│   ├── classifier.py             # Thuật toán bóc tách biên dạng Stage 2 & phân loại bất biến
│   └── measurer.py               # Thước đo Subpixel Caliper quy đổi mm từ Optical Scale
├── tools/                        # CÔNG CỤ TỰ ĐỘNG HÓA
│   └── auto_setup_from_pdf.py    # Tự động nạp PDF, bóc tách dung sai & tính tỷ lệ quang học
├── tests/                        # BỘ KIỂM THỬ TỰ ĐỘNG (REGRESSION & UNIT TESTS)
│   ├── test_accuracy.py          # Kiểm tra độ chính xác phân loại 100.00% trên 64 ảnh
│   └── test_metrology.py         # Kiểm tra tính toán kích thước Caliper subpixel
├── main.py                       # CLI: Đánh giá toàn bộ dây chuyền kiểm thử
├── predict.py                    # CLI: Kiểm định 1 sản phẩm đơn lẻ (Kèm cơ chế ghi đè thông minh)
├── requirements.txt              # Danh sách thư viện phụ thuộc (OpenCV, NumPy, PyPDF, Matplotlib)
└── README.md                     # Tài liệu hướng dẫn kỹ thuật
```

## 3. Quy Trình Sử Dụng Chuẩn (Workflow Commands)

### 1. Nạp bản vẽ PDF & Tự động cấu hình linh kiện mới:
Đặt file PDF vào thư mục `items/` (ví dụ `items/98003P.pdf`) và chạy:
```bash
python tools/auto_setup_from_pdf.py
```
*Tự động trích xuất ảnh drawing, đọc dung sai kỹ thuật, đo mẫu chuẩn đầu tiên (Golden Sample) và tính ra hệ số quang học ($\mu m/px$).*

### 2. Kiểm định đơn lẻ 1 sản phẩm (Có cơ chế ghi đè):
```bash
python predict.py assets/tests/98003P/images/98003P_1_rot23.png
```
* **Cơ chế ghi đè:** Nếu sản phẩm `98003P_1_rot23` đã từng được đo trước đó, hệ thống sẽ **ghi đè** kết quả mới lên chính sản phẩm này trong `outputs/visualizations/`, `outputs/steps/`, `outputs/predictions/`, `outputs/predictions.json` và `outputs/evaluation.csv`.
* **Bảo toàn nguyên vẹn:** Tất cả các sản phẩm khác cùng loại (như `98003P_1_rot71`, `98003P_2`) hoặc khác loại đều được giữ nguyên.

### 3. Đánh giá toàn bộ dây chuyền hàng loạt:
```bash
python main.py
```
*Đánh giá toàn bộ 64 ảnh trên 5 SKU, xuất bảng tổng hợp `evaluation.csv`, `predictions.json` và 128 ảnh trực quan độ nét cao.*

### 4. Chạy kiểm thử tự động (CI/CD Regression Tests):
```bash
python tests/test_accuracy.py    # Kiểm tra độ chính xác đạt 100.00%
python tests/test_metrology.py   # Kiểm tra tính toán thước đo Caliper
```

---

## 4. Cam Kết Tiêu Chuẩn Kỹ Thuật

- **0% AI / Deep Learning:** Không sử dụng bất kỳ mạng neural hay trọng số đen nào; 100% giải thuật tất định, minh bạch và có thể truy vết tới từng pixel.
- **Độ chính xác:** Đạt 100.00% (64/64 ảnh kiểm thử nhận diện chính xác, 0 sai lệch, 0 rejected).
- **Zero-Bleed Edge Detection:** Thuật toán Stage 2 Sobel gradient bám sát mép phôi thực tế, loại bỏ hoàn toàn hiện tượng ăn loang nền giấy hay nhiễu bóng mờ.