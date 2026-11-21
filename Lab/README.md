# Kiểm chứng riêng ba hướng phát triển so với bài báo RNN

Chạy từ thư mục `Lab` với Python 3.10+ và NumPy, Pillow. Mỗi hướng có mã,
đối chứng và report **trong thư mục riêng**; không dùng thí nghiệm ghép B0–B4
làm kết luận. Bài báo công bố RNN MAE **0,190**, RMSE **0,250** trên dữ liệu
chuẩn hóa [0,1], nhưng không công bố thuật toán/chỉ số lập lịch hay chi tiết
đủ để tái lập chính xác mẫu và horizon. Các report luôn đặt hai số đó ở dòng
**tham khảo**, không tính chênh lệch trực tiếp với kết quả Lab.
Nguồn hai số: [PDF do người dùng cung cấp, trang 4, Bảng 1, hàng RNN(Ours)](../Time-SeriesLoadPredictionforCloudResourceAllocationUsingRecurrentNeuralNetworks-1.pdf#page=4).

## Hướng 1 — dự báo thành thuật toán chọn máy

```bash
python3 host_selection/compare_to_paper.py
```

Kết quả: `host_selection/paper_comparison/report.html` và `report.json`.
Script tái huấn luyện RNN, so chọn máy theo dự báo với đối chứng **giữ nguyên
vị trí** trên cùng test. Bài gốc không có số lập lịch để tính phần trăm cải thiện.
Mẫu mặc định `host_selection/prepared_shared.json` chỉ có **4 container**;
report này là thử nghiệm nhỏ, không đại diện toàn bộ trace.

Chạy hướng 1 trên toàn bộ cohort đã quét (lưu checkpoint sau từng cohort):

```bash
python3 host_selection/compare_full.py
```

Đầu ra: `host_selection/paper_comparison/full.html` và `full.json`.
Đã chạy mẫu 100 cohort đầu tại `host_selection/paper_comparison/preview.html`.
Checkpoint của 100 cohort đầu đã đặt sẵn cho lệnh full; lệnh trên tiếp tục từ
cohort 101.

## Hướng 2 — biên dự phòng theo sai số

```bash
python3 safety_margin/compare_to_paper.py --q 0.95
```

Kết quả: `safety_margin/paper_comparison/report.html` và `report.json`.
Script huấn luyện RNN riêng, tính biên trên phần calibration của validation,
rồi so RNN đối chứng của Lab với chính mô hình đó sau khi cộng biên trên cùng test. Mẫu mặc định cũng
chỉ có 4 container. Không dùng MAE của dự báo **đã cộng biên** để tuyên bố
tăng độ chính xác dự báo điểm.

Chạy hướng 2 trên toàn bộ cohort với biên q chốt trước test:

```bash
python3 safety_margin/compare_full.py --q 0.95
```

Đầu ra: `safety_margin/paper_comparison/full.html` và `full.json`.
Đã chạy mẫu 100 cohort đầu tại `safety_margin/paper_comparison/preview.html`.
Checkpoint của 100 cohort đầu đã đặt sẵn cho lệnh full; lệnh trên tiếp tục từ
cohort 101.

## Hướng 3 — GNN thay RNN trong dự báo

Đã có checkpoint hoàn chỉnh trên 2.067 cohort. Tạo report tách riêng, không
huấn luyện lại:

```bash
python3 gnn/compare_to_paper.py \
  --reuse-checkpoint gnn/future_work_report.checkpoint.json
```

Kết quả: `gnn/paper_comparison/report.html` và `report.json`. Để tự chạy lại
chỉ RNN và GNN trên SQLite hiện có, bỏ `--reuse-checkpoint`:

```bash
python3 gnn/compare_to_paper.py \
  --catalog gnn/full_run/gnn_full_run/report.json \
  --output gnn/paper_comparison/fresh.html
```

Lần chạy mới có thể kéo dài hàng giờ; in tiến độ mỗi 10 cohort và lưu checkpoint
sau từng cohort. Chạy lại cùng lệnh để tiếp tục. `--max-cohorts 10` chỉ dùng
kiểm tra nhanh, dùng `--output` riêng khi thay số cohort.

Hai lệnh full của hướng 1/2 cũng đọc SQLite sẵn có, **không quét lại CSV**, nhưng
vẫn phải huấn luyện RNN ở từng cohort/seed nên có thể mất hàng giờ. Chúng lưu
checkpoint cùng tên output, tiếp tục được khi chạy lại. Dùng
`--max-cohorts 10 --output .../preview.html` để kiểm tra nhanh.

## Kiểm thử mã

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
```

Ba report kiểm tra **ba câu hỏi độc lập**. Chỉ GNN có MAE/RMSE cùng loại với
Bảng 1 của bài gốc, nhưng vẫn chỉ so trực tiếp được với **RNN đối chứng của Lab trên
cùng cohort/test**. Muốn khẳng định vượt số 0,190/0,250 của tác giả cần mã
nguồn hoặc thông số tạo mẫu, mục tiêu, chuẩn hóa và tập test gốc.
