# Kết quả thực nghiệm hướng 3: GNN

## Dữ liệu và phép thử

- Nguồn: `Lab/data/container_usage.csv` (Alibaba 2018), tập con chung tại `Lab/host_selection/prepared_shared.json`.
- 4 container: `c_21192`, `c_10005`, `c_12910`, `c_17113`. Cửa sổ 43 bước liên tiếp, mỗi bước 1.200 giây (20 phút). Chia 25 bước train, 9 validation và 9 test theo thời gian. Chỉ các cửa sổ đồng thời đầy đủ cho cả 4 container được dùng.
- Tập con này được chọn sau khi kiểm tra khả năng tạo cửa sổ đồng thời, vì cửa sổ 24 giờ và 8 giờ ban đầu không đáp ứng ngưỡng độ phủ của plan. Đây là **mẫu thăm dò có thiên lệch chọn mẫu**, không đại diện cho toàn bộ trace.
- Lịch sử 6 bước = 2 giờ, dự báo 1 bước = 20 phút. Có **5 cửa sổ train**, 3 cửa sổ chọn epoch, 1 cửa sổ calibration (4 sai số container) và 5 bước test liên tiếp; mô phỏng tính quyết định từ bước test thứ hai nên chỉ có 4 chu kỳ quyết định. Tỷ lệ thiếu trên toàn tập con là 28,125%.
- Cạnh đồ thị được tạo bằng tương quan CPU dương chỉ từ train, tối đa 2 láng giềng mỗi container. Trên tập này có 2 cạnh có hướng (một cặp liên kết hai chiều). Ablation dùng cùng kiến trúc và seed nhưng đặt ma trận cạnh bằng 0. Chạy seed 11, 22, 33; q=0,95 của B4 được ấn định trước.

Tệp máy đọc được: [`Lab/gnn/results_real.json`](../gnn/results_real.json). Lệnh chạy và mã nằm trong [`Lab/gnn/README.md`](../gnn/README.md).

## Sai số dự báo trên test

Trung bình 3 seed, đơn vị là phần trăm sử dụng của container trong trace, **không phải thang chuẩn hóa [0,1] của bài báo gốc**:

| Mô hình | MAE CPU | RMSE CPU | MAE RAM | RMSE RAM |
|---|---:|---:|---:|---:|
| GNN một lớp | 2,469 | 3,985 | 17,525 | 17,867 |
| Cùng mạng, bỏ cạnh | 2,445 | 3,987 | 17,612 | 17,964 |

GNN cải thiện RMSE CPU khoảng 0,003 và MAE/RMSE RAM khoảng 0,087/0,097, nhưng MAE CPU kém hơn khoảng 0,024. Các khác biệt này nhỏ và không nhất quán theo tài nguyên; **không đủ bằng chứng GNN tốt hơn đối chứng bỏ đồ thị**. Độ lệch chuẩn giữa seed của MAE RAM GNN là 3,501, lớn hơn nhiều so với chênh lệch 0,087.

## Kết quả lập lịch

| Phương án | Tỷ lệ máy-bước quá dung lượng | Thiếu CPU | Thiếu RAM | Giờ máy hoạt động | Di chuyển / 100 container-giờ |
|---|---:|---:|---:|---:|---:|
| B3: GNN | 0,333 | 0 | 50 | 4,667 | 37,5 |
| B4: GNN + biên sai số | 0,333 | 0 | 50 | 4,667 | 37,5 |
| Bỏ cạnh, cùng bộ chọn máy | 0,333 | 0 | 50 | 4,667 | 37,5 |

Ba seed cho cùng quyết định lập lịch trong mẫu ngắn này. Biên q=0,95 không làm đổi quyết định, nên B4 chưa thể hiện lợi ích so với B3. Số thiếu RAM là đơn vị mô phỏng của phép cộng phần trăm tải container, không phải GB thiếu trên máy Alibaba thực.

Thí nghiệm hiện tại tập trung vào dự báo GNN so với RNN tái lập. Xem [report riêng hướng GNN](../gnn/paper_comparison/report.html) cho kết quả trên 2.067 cohort; phép thử 4 container trong tài liệu này chỉ có tính thăm dò.

## Thời gian tính toán

Trung bình 3 seed: huấn luyện GNN 0,0163 giây, đối chứng bỏ cạnh 0,0164 giây; suy luận GNN 0,0417 ms/bước, đối chứng 0,0320 ms/bước. Đây là thời gian trên tập 4 container và máy thử nghiệm hiện tại, không ngoại suy ra cụm lớn.

## Suy luận và ý nghĩa

Kiểm thử này chứng minh luồng tạo đồ thị từ train, huấn luyện mô hình truyền thông điệp, dự báo và lập lịch chạy được trên **dữ liệu container thật**. Nó chưa chứng minh quan hệ đồ thị giúp hệ thống tốt hơn. Việc chỉ có một cặp cạnh và 5 bước test làm giá trị bổ sung của GNN khó quan sát. Hướng nghiên cứu có ý nghĩa nếu mở rộng sang nhiều container có đồng thời đủ dữ liệu, xây cạnh theo quan hệ ứng dụng hoặc vị trí máy và đo cả hiệu quả lập lịch lẫn chi phí tính toán.

## Giới hạn so sánh với bài báo gốc

Bài báo gốc báo cáo RNN MAE 0,190 và RMSE 0,250 cho thang dữ liệu chuẩn hóa, không tách CPU/RAM hoặc horizon, và không công bố chỉ số lập lịch tương đương. Vì khác đơn vị, mẫu dữ liệu và cấu hình, **không được so trực tiếp các số MAE/RMSE trong bảng trên với 0,190/0,250**. B3/B4 là mô phỏng triển khai ý tưởng, không phải tái lập hệ thống gốc.
