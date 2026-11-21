# Nguồn số của báo cáo chạy thử cũ

`report.html` là báo cáo **chạy thử 10 cohort**, không phải kết quả bài báo
gốc và không phải report 2.067 cohort hiện tại.

## Dòng dữ liệu

- `Lab/data/container_usage.csv` được quét trước đó bởi
  `gnn/run_full_auto.py`, tạo catalog và SQLite tại
  `gnn/full_run/gnn_full_run/`.
- `cohorts.jsonl` trong thư mục này chứa đúng **10 cohort hợp lệ đầu tiên**
  của catalog, mỗi cohort có 4 container và kết quả cho seed 11, 22, 33.
- Số `31.670.001` dòng CSV và `8.979` cohort loại lấy từ **catalog quét toàn
  bộ**, không phải số mẫu được đánh giá trong report chạy thử này.
- 10 cohort × 3 seed tạo 30 lượt đánh giá; tổng số bước test là 111, tương
  ứng 444 container-bước dự báo. Chỉ 12 lượt cohort-seed đủ bước test liên
  tiếp để tính chỉ số lập lịch.

## Cách đọc số

`report.json` là bảng tổng hợp từ các bản ghi trong `cohorts.jsonl`; HTML
hiển thị bảng tổng hợp đó. MAE tổng gộp CPU/RAM, có trọng số theo số bước
test × 4 container; ví dụ RNN là 0,0671804673, NoGraph 0,0426906712 và
GNN 0,0427521426. CPU/RAM được chia cho 100. Tỷ lệ quá tải là tổng số
máy-bước quá tải chia tổng máy-bước quan sát chỉ trên các lượt có lập lịch;
RNN 0,225 và GNN 0,2142857. Các phép tính lại từ JSONL khớp `report.json`.

Thời gian huấn luyện, suy luận và quyết định là số đo đồng hồ trong lần chạy
thử đó, phụ thuộc máy và tải nền. Chỉ số máy là **mô phỏng** với dung lượng
giả định, không phải quan sát vận hành Alibaba.

## Công thức các cột trong bảng HTML

Gọi `y` là tải thật và `ŷ` là dự báo trên thang [0,1]. Mỗi mẫu test có CPU,
RAM của từng container. Gộp qua cohort/seed có trọng số theo số bước test
và 4 container:

| Cột | Cách tính từ `cohorts.jsonl` |
|---|---|
| MAE tổng | Trung bình `abs(ŷ-y)` trên mọi bước test, container và hai tài nguyên. |
| RMSE tổng | Căn của trung bình `(ŷ-y)²` trên cùng các giá trị. |
| Quá tải | Tổng `scheduler.overloaded_host_steps` / tổng `scheduler.observed_host_steps`, chỉ các lượt có scheduler. Máy quá tải nếu CPU **hoặc** RAM thực vượt dung lượng. |
| Tổng thiếu | Tổng `scheduler.cpu_shortfall + scheduler.ram_shortfall`; đơn vị điểm sử dụng mô phỏng, **không phải GB hay core**. |
| Giờ máy | Tổng `scheduler.active_host_hours`; trong simulator, mỗi máy có trong trạng thái được cộng một bước 20 phút, kể cả máy vừa mở chưa sẵn sàng. |
| Di chuyển | Tổng `scheduler.moves` qua các lượt chạy hợp lệ. |
| Huấn luyện TB (s) | Trung bình `training_seconds` của 30 lượt cohort–seed; chỉ là thời gian được bản ghi cũ lưu, không khẳng định đã tính chi phí thử mọi đồ thị ứng viên. |
| Suy luận TB (ms/bước) | Trung bình `inference_ms_per_step` của 30 lượt. |
| Quyết định TB (ms) | Trung bình `scheduler.decision_ms_median` của 12 lượt có lập lịch; đây là trung bình các trung vị, không phải trung vị gộp mọi quyết định. |

Các công thức trên đã được tính lại từ JSONL và khớp các giá trị lưu trong
`report.json` đến sai số làm tròn.

## Input của từng hướng

Đầu vào chung là CSV Alibaba với `container_id`, `time_stamp`, phần trăm CPU
và RAM. `run_full_auto.py` gom mỗi **1.200 giây**, dùng giá trị **lớn nhất**
trong ô nếu có nhiều dòng cùng container, xét cửa sổ **24 giờ (72 ô)** và
chọn cohort **4 container** có ít nhất 90% độ phủ train. Bài toán dự báo dùng
**6 ô lịch sử → 1 ô tương lai**, chia theo thời gian: 60% train, 20%
validation, 20% test. Giá trị CPU/RAM được chia cho 100.
Trường `machine_id` trong CSV không được dùng để đặt máy thật hoặc dựng cạnh
đồ thị ở phép thử này; cạnh dựa vào tương quan tải trong train.

- **Hướng 1 — chọn máy:** input là dự báo RNN, tải thật chỉ dùng sau quyết
  định để chấm điểm, vị trí ban đầu luân phiên giữa `h0`/`h1`, mỗi máy có
  dung lượng giả định CPU/RAM `(100,100)`. Đây là cùng bộ chọn máy dùng trong
  các hàng RNN/NoGraph/GNN của smoke report.
- **Hướng 2 — biên dự phòng:** input là dự báo RNN và sai số validation để
  tính phân vị biên. **Smoke report này không chạy hướng 2**; xem report
  riêng ở `safety_margin/paper_comparison/`.
- **Hướng 3 — GNN:** input là cùng lịch sử CPU/RAM của 4 container, cộng ma
  trận cạnh tính từ phần train. Đầu ra là dự báo CPU/RAM; NoGraph là cùng
  mạng bỏ cạnh để kiểm tra tác dụng thông tin quan hệ.

Máy chủ, dung lượng và lịch di chuyển ở trên thuộc **mô phỏng Lab**, không
phải thông tin máy chủ thật của trace hoặc cấu hình trong bài báo gốc.

## Giá trị có sẵn và tham số tự đặt

Header của `container_usage.csv` chứa `container_id`, `machine_id`,
`time_stamp`, `cpu_util_percent`, `mem_util_percent` cùng một số cột khác.
Ví dụ bản ghi `c_1,m_2556,86610,12.0,88,...` cung cấp CPU **12,0%** và
RAM **88%** của container tại thời điểm 86610; đầu vào dự báo trên thang
[0,1] là `[0.12, 0.88]`. Đây là **mức sử dụng được ghi**, không phải dung
lượng máy chủ hay lượng tài nguyên được cấp cho container.

Các tham số **không lấy từ CSV**: ô 1.200 giây, cửa sổ 24 giờ, nhóm 4
container, độ phủ train 90%, chia 60/20/20, lịch sử 6 ô, dự báo 1 ô,
16 nút ẩn RNN, tối đa 100 epoch, seed 11/22/33, hai máy ban đầu dung lượng
`(100,100)`, ngưỡng chọn máy 90%, giới hạn/cooldown di chuyển, và `q=0,95`
trong phép thử biên riêng. `boot_minutes=1` trong mã mô phỏng thực tế được
so với **chỉ số bước**, nên tương ứng một ô thời gian chứ không phải một
phút vật lý.

Trace có các tệp `machine_meta.csv` và `container_meta.csv`, nhưng pipeline
smoke hiện **không nối** chúng để lấy dung lượng máy, mức cấp phát hay vị trí
thực. Bộ lập lịch đang cộng phần trăm sử dụng container và so với dung lượng
giả định 100; vì vậy “tổng thiếu”, “giờ máy” và tỷ lệ quá tải chỉ là chỉ số
trong mô phỏng, **không được diễn giải thành tài nguyên thiếu thật của cụm**.

Script đã tạo **phiên bản chính xác** của report smoke này không còn được giữ
sau khi dọn luồng B0–B4 cũ; vì vậy có thể kiểm tra số đã lưu và phép tổng
hợp, nhưng không bảo đảm tái chạy cho ra cùng thời gian đo. Muốn kết quả
được duy trì và có thể chạy lại, dùng `gnn/compare_to_paper.py` và report
`gnn/paper_comparison/report.html` trên 2.067 cohort.
