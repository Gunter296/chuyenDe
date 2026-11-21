# Hướng 1: chuyển dự báo RNN thành quyết định chọn máy

Chạy từ `Lab`:

```bash
python3 host_selection/compare_to_paper.py
```

`compare_to_paper.py` tái huấn luyện RNN bằng `train_rnn.py`, đưa dự báo vào
`scheduler.py` qua `run.py`, và so trên cùng tải thực với đối chứng giữ nguyên
vị trí. Report HTML và JSON nằm trong `paper_comparison/`. `prepare_trace.py`
tạo dữ liệu đầu vào từ trace khi cần; `experiment.py` chạy nhiều seed.

Bài báo gốc chỉ công bố MAE/RMSE dự báo, không công bố số lập lịch hoặc thuật
toán chọn máy đủ chi tiết để tái lập. Vì vậy report đánh giá **đóng góp mở rộng**
của Lab so với vị trí cố định; không gán kết quả vị trí cố định cho tác giả.

Đầu vào mặc định có bốn container và chỉ là phép thử nhỏ. Có thể dùng
`--prepared PATH` để chạy một tập đã chuẩn bị khác cùng cấu trúc JSON.
Để chạy trên mọi cohort đã quét, dùng `python3 host_selection/compare_full.py`
từ `Lab`; script tạo `paper_comparison/full.html` và checkpoint tiếp tục được.
Xem [hướng dẫn tổng](../README.md) để đọc các giới hạn khi so với bài báo.
