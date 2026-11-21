# Hướng 1 — kết quả chạy trên trace container thật

## Dữ liệu và giới hạn phép thử

Nguồn: `Lab/data/container_usage.csv` (Alibaba 2018, tệp người dùng cung cấp). Tiền xử lý đọc toàn bộ container ID trong cửa sổ thử, gộp CPU/RAM bằng **giá trị lớn nhất trong mỗi ô thời gian**. Với cửa sổ 8 giờ [86400,115200) và ô 300 giây, độ phủ train tốt nhất là 63,2%; với cửa sổ 24 giờ [86400,172800) và ô 900 giây, độ phủ tốt nhất là 89,5%, đều dưới ngưỡng 90%. Với cửa sổ 24 giờ và ô 1200 giây, 8 container được chọn đạt ngưỡng; độ thiếu dữ liệu toàn mẫu 8 container là 28,1%, chủ yếu có khoảng trống chung về sau. Vì vậy, phép thử khả thi này lấy 4 container có độ phủ train cao nhất và đoạn 43 ô thời gian [86400, 138000) trước khoảng trống chung. Đây là **lựa chọn hậu nghiệm để kiểm tra khả năng chạy**, không phải mẫu đại diện hay tái lập bài báo gốc.

ID: `c_21192`, `c_10005`, `c_12910`, `c_17113`. Độ phủ train tương ứng 100%, 92%, 92%, 96%. Chia thời gian theo 25 ô train, 9 ô validation (4 ô chọn epoch, 5 ô calibration), 9 ô test. Khi yêu cầu lịch sử 6 ô và nhãn hiện tại đủ cho cả 4 container, chỉ có 5 bước test liên tiếp. Các con số dưới đây dùng 4 quyết định test sau bước khởi tạo. File đã dùng: `Lab/host_selection/prepared_shared.json`; báo cáo máy: `Lab/host_selection/results_real/report.json`.

Lưu ý: 1 horizon ở đây là **20 phút**, lịch sử 6 ô là **120 phút**, khác cấu hình 30 phút → 5 phút trong plan. CPU/RAM container là phần trăm sử dụng theo trace; schema chính thức không xác định đầy đủ mẫu số cho hai trường này. Bộ lập lịch cộng chúng trong **đơn vị mô phỏng** với máy dung lượng (100,100); không coi đây là tải hoặc dung lượng vật lý của máy Alibaba.

## Cách chạy

RNN Elman dùng chung giữa container, 16 nút ẩn, tối đa 30 epoch, early stopping trên nửa đầu validation. Ba seed 11, 22, 33. B0 dùng tải thực ở bước ngay trước, B1 dùng dự báo RNN; cùng thuật toán chọn máy, trạng thái đầu và giới hạn di chuyển. Các sai số dưới đây ở thang phần trăm gốc của dữ liệu, riêng CPU/RAM, horizon 1.

| Seed | MAE CPU | RMSE CPU | MAE RAM | RMSE RAM |
|---|---:|---:|---:|---:|
| 11 | 2,488 | 4,103 | 25,337 | 26,271 |
| 22 | 2,322 | 3,723 | 18,377 | 19,390 |
| 33 | 2,408 | 4,077 | 25,027 | 25,297 |

| Chỉ số lập lịch | B0 (tải gần nhất) | B1 (RNN) |
|---|---:|---:|
| Máy-bước quá dung lượng / quan sát | 4 / 12 | 4 / 12 |
| Tỷ lệ quá tải | 33,3% | 33,3% |
| Tổng thiếu CPU | 0 | 0 |
| Tổng thiếu RAM | 50 | 50 |
| Giờ máy hoạt động | 4,667 | 4,667 |
| Di chuyển / 100 container-giờ | 37,5 | 37,5 |

Kết quả lập lịch giống nhau ở cả ba seed. Trên mẫu này, **chưa có bằng chứng B1 cải thiện B0**. RAM là tài nguyên gây quá tải trong mô phỏng; độ dài test rất ngắn và cách chọn mẫu hậu nghiệm khiến kết quả chỉ có giá trị kiểm tra khả năng thực thi, chưa đủ để kết luận tổng quát.

## So với bài báo gốc

Bài báo gốc báo cáo RNN MAE 0,190 và RMSE 0,250 trong bảng so sánh mô hình dự báo, sau chuẩn hóa và chia 80:20 theo thời gian. Bài báo không công bố phép đo quá tải, giờ máy hay số lần di chuyển từ thuật toán chọn máy cụ thể. MAE/RMSE ở bảng trên dùng thang phần trăm gốc, mẫu dữ liệu, bước thời gian và cấu hình khác nên **không so sánh trực tiếp trị số** với 0,190/0,250. Đóng góp có thể kiểm tra của hướng 1 là bổ sung tầng quyết định chọn máy và chỉ số hệ thống, chưa chứng minh hiệu năng cao hơn bài báo gốc.

## Tái lập

```powershell
python Lab/host_selection/experiment.py Lab/host_selection/prepared_shared.json Lab/host_selection/results_real --seeds 11 22 33 --history 6 --horizon 1 --hidden 16 --epochs 30
```

Muốn đánh giá theo đúng plan ban đầu cần chuỗi container dày hơn hoặc cách xử lý thiếu được định nghĩa trước, horizon 5 phút, tập test dài hơn, và kiểm tra độ nhạy dung lượng máy. Cần tránh chọn mẫu sau khi xem kết quả test.

Kiểm thử mã hướng 1: `python -m unittest discover -s Lab/host_selection -p 'test_*.py' -v`, 9/9 bài kiểm thử đạt trong phiên chạy này.
