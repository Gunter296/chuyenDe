# So sánh RNN + chọn máy, GNN dự báo và RNN bài báo gốc

Thí nghiệm hiện có **hai phương án chạy trên cùng Alibaba Cluster Trace 2018**:

1. **RNN + thuật toán chọn máy:** huấn luyện RNN dự báo CPU/RAM 5 bước, cộng biên sai số validation, rồi dùng Cost-aware Best-fit để chọn máy. Báo MAE/RMSE và chỉ số mô phỏng hệ thống.
2. **GNN chỉ dự báo:** dùng GRU và truyền tin trên đồ thị container–container động để dự báo cùng nhãn/test. Báo MAE/RMSE và thời gian mô hình; **không chạy thuật toán chọn máy cho GNN**.

**RNN của bài báo gốc** là mốc tham khảo đã công bố: MAE 0,190, RMSE 0,250 ở Bảng 1. Không lấy hiệu số với kết quả ở đây vì khác mẫu, horizon và chuẩn hóa. Bài báo gốc không công bố chỉ số chọn máy. RNN huấn luyện trong dự án là bản cài đặt đối chứng, không phải trọng số/mã nguồn của tác giả.

## Cài đặt và chạy

Các tệp dữ liệu CSV và chỉ mục SQLite là đầu vào/kết quả cục bộ, không được lưu trong GitHub. Hãy đặt các CSV theo đường dẫn dưới đây; lần chạy đầu sẽ tạo lại chỉ mục SQLite. Có thể dùng `--database` để trỏ tới chỉ mục có sẵn trên máy.

```bash
python3 -m pip install -r requirements.txt
python3 run.py --input ../Lab/data/container_usage.csv --output results_simple
python3 -m unittest discover -s tests -v
```

Lệnh một cohort tạo `results_simple/comparison.json`. Để quét toàn bộ CSV và đánh giá mọi cohort đủ điều kiện, chạy trên ổ cục bộ của VM:

```bash
python3 run.py --all-cohorts \
  --input ../Lab/data/container_usage.csv \
  --output ~/gnn_compare_full \
  --database gnn_host_full/usage.sqlite
```

Chế độ này dùng SQLite chỉ định (hoặc tạo `usage.sqlite` trong output), rồi ghi `comparison_cohorts.jsonl` (checkpoint từng cohort) và `comparison_full.json` (tổng hợp). Chạy lại **cùng lệnh** để tiếp tục sau khi dừng. Xem trước 10 cohort bằng `--max-cohorts 10` và **một thư mục output riêng**. Các file cũ `report.json`, `full_report.json`, `cohorts.jsonl` có nhãn B0–B4 là kết quả lịch sử của thiết kế trước, không phải đầu ra mới.

`--database gnn_host_full/usage.sqlite` tái sử dụng chỉ mục CSV từ lần chạy trước nếu đường dẫn, kích thước, thời gian sửa CSV và kích thước ô trùng nhau. Nếu không còn database đó, bỏ tùy chọn `--database` để chương trình tạo chỉ mục mới trong thư mục output.

## Giao thức và giới hạn

- CSV gốc: `container_usage.csv`; CPU/RAM phần trăm được chia 100 thành đơn vị tải mô phỏng. Chia thời gian 60/20/20 trong mỗi khoảng 24 giờ; ô 20 phút; lịch sử 12 ô; dự báo 5 ô (100 phút). Chỉ dùng cohort 4 container có đủ dữ liệu đồng thời, độ phủ train ≥90%, ít nhất 5 cửa sổ test liên tiếp. Report ghi số dòng/ID đã quét, đủ điều kiện, được đánh giá và bị loại.
- RNN và GNN dùng cùng cohort, mốc thời gian, nhãn, seed và ngân sách epoch. GNN tạo cạnh từ lịch sử quan sát được tại thời điểm dự báo, không nhìn nhãn tương lai. Chỉ RNN có bước chọn máy.
- Mô phỏng dùng `machine_meta.csv` làm dung lượng tương đối và `machine_usage_bigger.csv` làm hồ sơ tải nền. Tỷ lệ tải container cộng với tải nền là **giả định kịch bản**, không phải phép cộng vật lý chính xác core/GB. Chế độ toàn cohort dùng một pool máy chung, có thể đếm trùng tải nếu container vốn chạy trên máy trong pool. Kết quả quá tải/giờ máy là chỉ số mô phỏng, không phải đo trực tiếp cụm Alibaba.
- Chỉ có một seed mặc định. Muốn kết luận tổng quát cần nhiều seed và kiểm tra độ nhạy với dung lượng, tải nền, kích thước cohort và tham số chọn máy.
