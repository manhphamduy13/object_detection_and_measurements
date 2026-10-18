# Đo theo golden contour

## Chạy Studio

Môi trường dự án là Conda `base`. Từ `D:\Projects\CV`:

```powershell
conda activate base
python .\object_detection_and_measurements\tools\scanline_studio.py
```

Nếu đang đứng trong thư mục `object_detection_and_measurements`:

```powershell
python .\tools\scanline_studio.py
```

## Hai đường dung sai

Trong panel phải, cuộn xuống `04 TEMPLATE CONTOUR`:

Bật `CHECK GOLDEN CONTOUR` để kiểm tra toàn contour (mặc định bật trong Studio).
Chức năng này độc lập với loại phép đo: recipe chỉ có SCAN vẫn hiển thị golden,
hai đường giới hạn và kết luận profile; không cần tạo một hàng TPL để bật template.

| Ô nhập | Ý nghĩa | Đường hiển thị |
|---|---|---|
| Inward / Lower (mm) | Biên được phép lệch vào trong golden bao nhiêu mm | Xanh lam |
| Outward / Upper (mm) | Biên được phép lệch ra ngoài golden bao nhiêu mm | Cam |

Golden màu xám; trong ảnh kiểm tra, contour vật thật màu xanh lá và các vị trí
vượt band màu đỏ. Thay đổi offset sẽ cập nhật hai đường xem trước ngay.

Đây là offset đều theo khoảng cách, không phải phóng to/thu nhỏ từ tâm.
Dung sai áp dụng cho toàn bộ **contour ngoài** của SKU. Offset vùng lõm có thể
tách thành nhiều đường; vùng quá hẹp có thể biến mất khi offset vào trong.
Hình hiển thị offset được dựng trên lưới pixel phụ; phép kiểm tra dùng khoảng
cách có dấu tới polygon, đồng thời kiểm tra phần lõi bắt buộc phải còn vật liệu.
Tất cả đỉnh contour được kiểm tra; đoạn dài được lấy mẫu tối đa cách nhau 1 px.

Ví dụ Lower=0.10, Upper=0.20: mỗi mép được lệch vào 0.10 mm hoặc lệch ra 0.20 mm.
Với hai mép thẳng đối diện, chiều rộng có thể thay đổi tổng cộng -0.20/+0.40 mm.
Không nhầm các giá trị này với dung sai chiều rộng tổng.

`03 DIMENSION TOLERANCE` là dung sai kích thước riêng của từng scanline quanh
Nominal. Kích thước và profile phải cùng PASS để cổng metrology PASS. Các gate
phân loại/template matching khác của pipeline vẫn được tính trong kết luận cuối.

Recipe cũ có `contour_tolerance_mm` được đọc thành hai offset bằng nhau. Khi lưu
mới, recipe dùng `contour_lower_mm` và `contour_upper_mm`.

## Dạy và đo thử

1. Chọn SKU và ảnh golden. Kiểm tra mm/pixel đang hiển thị đúng.
2. Nhập ID, Nominal và dung sai kích thước ở các mục 02/03.
3. `Golden actual mm (blank = image)` để trống sẽ lấy chiều dài golden từ ảnh:
   `golden_px × mm_per_px`. Nếu có số đo thực, nhập tại đây để ghi đè cho scanline đó.
   Nominal vẫn là kích thước trên bản vẽ, không dùng để thay số đo golden.
4. Nhập Lower/Upper ở mục 04 để tạo hai đường giới hạn toàn contour.
5. Bấm `ADD TEMPLATE-CONTOUR DIMENSION`, click P1 rồi P2 trên hai mép cần đo.
6. Chọn hàng có MODE=`TPL`. Nếu sửa giá trị thực/dung sai/nominal, bấm
   `APPLY DIMENSION` trước khi lưu hoặc chạy thử. Xóa ô Golden actual và bấm Apply
   để quay về đo golden bằng ảnh.
7. Bấm `LƯU SPEC & CÔNG THỨC` để lưu. `CHẠY THỬ MẪU` cũng dùng các offset và
   scanline đang có trong Studio, kể cả trước khi lưu.
8. Hộp kết quả có tab `Contour / Scanline`, tab `Báo cáo 6 khung` và bảng Golden /
   Đo được / Chênh lệch / ΔP1 / ΔP2. Báo cáo tự lưu vào
   `outputs/visualizations/<tên ảnh>.png`; dữ liệu và recipe dùng khi kiểm tra lưu
   trong `outputs/predictions/<tên ảnh>.json`. Chạy lại cùng tên sẽ cập nhật báo cáo đó.

Các công cụ L/W/caliper thường vẫn tạo phép đo SCAN. Muốn thử phương pháp mới,
dùng nút template để tạo hàng TPL; có thể xóa hàng SCAN trùng ID trước khi tạo.
Với SCAN, bảng chạy thử lấy số đo trên ảnh golden làm baseline và hiển thị tổng
chênh lệch; ΔP1/ΔP2 chỉ có ý nghĩa với phép đo TPL giữ mốc golden.

## Ảnh thử bị lõm

Ảnh thử thông thường: `assets/tests/98475/images/98475_1_dented.png`.
Ảnh đối chứng: `assets/tests/98475/images/98475_1_golden_control.png`.
Không có chế độ hay công cụ chạy riêng cho ảnh lõm.

Báo cáo đã chạy tại `outputs/visualizations/98475_1_dented.png` và
`outputs/visualizations/98475_1_golden_control.png`. Với recipe thử ghi trong JSON
báo cáo (offset vào/ra 0.40 mm, scanline ngang qua chỗ lõm), đối chứng PASS;
ảnh lõm FAIL, số đo từ 4.124 mm còn 3.223 mm, chênh -0.901 mm.
Recipe thử chỉ dùng trong bộ nhớ, không thay recipe sản xuất đã lưu.

Có thể kiểm tra ảnh bằng recipe SKU hiện tại mà không qua nhận dạng tự động:

```powershell
python predict.py --image assets/tests/98475/images/98475_1_dented.png --sku 98475 --profile-lower 0.4 --profile-upper 0.4
```

Lệnh này dùng các scanline đang lưu của SKU nên số đo có thể khác recipe thử trên.
Ảnh lõm được tạo tổng hợp để kiểm tra chức năng, không phải mẫu đo vật lý đã hiệu chuẩn.

## Scanline và chênh lệch

Golden chỉ được quay và tịnh tiến để căn với mẫu; không scale theo kích thước mẫu.
Scanline giữ phương/vị trí đã teach trong hệ tọa độ golden và được kéo dài như
một đường thẳng khi tìm giao điểm, nên vẫn cắt mẫu lớn hơn đoạn P1–P2 ban đầu.
Giao điểm gần mỗi mốc golden ở hai phía của tâm scanline được ưu tiên.
Nếu không có đủ hai mép phù hợp, báo N/A và FAIL; không dịch đường đo sang chỗ khác.

```text
measured_mm = golden_length_mm + delta_p1_mm + delta_p2_mm
total_delta_mm = delta_p1_mm + delta_p2_mm
```

Mỗi delta dương khi mép đi xa tâm scanline, âm khi đi vào gần tâm scanline.
Ví dụ golden=10.00, ΔP1=+0.05, ΔP2=+0.07 thì mẫu đo=10.12, chênh=+0.12 mm.

Trong JSON `measurements_px`:

- `recipe_calipers.<ID>` chứa baseline, nguồn `golden_source` (`image`/`actual`),
  hai delta, tổng delta và kích thước cuối.
- `spec_report` có các giá trị tương tự cùng PASS/FAIL kích thước.
- `contour_profile` có Lower/Upper, sai lệch vào/ra lớn nhất, điểm vượt band,
  hình học golden đã căn và PASS/FAIL profile.
- `dimension_pass`, `profile_pass`, `caliper_pass` phân biệt các nguyên nhân lỗi.

## Zoom và di chuyển ảnh

- Ctrl + lăn lên: phóng to tại con trỏ, tối đa 32 lần so với ảnh vừa khung.
- Ctrl + lăn xuống: thu nhỏ về mức vừa khung (100%), không thu nhỏ hơn.
- Giữ chuột trái và kéo: di chuyển ảnh đang phóng to.
- Click trái không kéo: chọn điểm hoặc chọn phép đo như bình thường.
- Con lăn trên panel bên phải: cuộn form.
- Hộp kết quả đo thử cũng hỗ trợ Ctrl + lăn và kéo ảnh.

Zoom chỉ đổi cách hiển thị. Contour, tọa độ teach và mm/pixel không thay đổi.

## Giới hạn khi thử bằng ảnh trên mạng

Kết quả mm dựa hoàn toàn vào mm/pixel đã đặt. Ảnh trên mạng chưa có vật chuẩn
chỉ cho số đo quy đổi theo hệ số đó, chưa xác nhận kích thước vật lý thực.
Golden và ảnh test cần cùng tỉ lệ ảnh; resize/zoom camera giữa hai lần chụp sẽ
bị hiểu thành thay đổi kích thước. Nếu có số đo thực của golden, ô ghi đè chỉ
điều chỉnh baseline scanline đó; hệ số quy đổi delta và offset vẫn là mm/pixel.

Căn chỉnh hiện tại dùng khung bao xoay và dấu lệch tâm của contour để chọn chiều.
Chi tiết gần vuông/đối xứng hoặc mất vật liệu lớn có thể làm căn chỉnh không ổn định;
cần đối chiếu overlay trước khi dùng kết quả để đánh giá hàng thực.
