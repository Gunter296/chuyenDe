# GNN trên trace Alibaba: chia 80/20, thang [0,1]

Nguồn là `Lab/host_selection/prepared_shared.json`, trích từ `Lab/data/container_usage.csv`. Có 4 container và 43 ô thời gian, mỗi ô 20 phút. Chia theo thời gian: 34 ô đầu (80%) để phát triển mô hình, 9 ô cuối (20%) để test. Trong 34 ô đầu, 25 ô dùng huấn luyện và 9 ô validation để chọn epoch. Đầu vào là 6 ô lịch sử (2 giờ), dự báo 1 ô (20 phút). Dữ liệu CPU/RAM vốn là phần trăm trong [0,100], được chia cho 100 để về [0,1]; không dùng nhãn test để chọn mô hình hoặc tạo cạnh.

Vì thiếu quan sát đồng thời giữa các container, chỉ có **5 cửa sổ train, 4 cửa sổ validation và 5 cửa sổ test**. Chạy seed 11, 22, 33; tối đa 100 epoch, chọn epoch theo validation MSE. Đối chứng NoGraph dùng cùng kiến trúc, cùng seed, chỉ đặt ma trận cạnh bằng 0. Cạnh GNN dựa trên tương quan CPU của train.

## Kết quả test, trung bình ba seed

| Mô hình | MAE CPU | RMSE CPU | MAE RAM | RMSE RAM |
|---|---:|---:|---:|---:|
| Không cạnh đồ thị | 0,02646 | 0,03857 | 0,09433 | 0,10193 |
| GNN | 0,02718 | 0,03875 | 0,08814 | 0,09747 |

GNN giảm sai số RAM khoảng 0,00619 MAE và 0,00447 RMSE nhưng sai số CPU tăng khoảng 0,00072 MAE và 0,00018 RMSE so với cùng mạng không cạnh. Khác biệt trên 5 bước test không đủ chứng minh lợi ích tổng quát của GNN. [Kết quả từng seed](results.json) lưu cả số cửa sổ, số cạnh, dự báo và nhãn test.

## Biểu đồ

- [CPU đầu vào của bốn container](input_cpu.png)
- [RAM đầu vào của bốn container](input_ram.png)
- [CPU dự báo và thực tế, trung bình bốn container](forecast_cpu_seed11.png)
- [RAM dự báo và thực tế, trung bình bốn container](forecast_ram_seed11.png)
- [Đường học train/validation của GNN](learning_curve_seed11.png)

Các biểu đồ dự báo và đường học hiển thị seed 11 để nhìn diễn biến; bảng kết quả dùng cả ba seed. Biểu đồ đầu vào chỉ nối các ô có quan sát của từng container nên trục ngang là thứ tự mẫu quan sát, không phải khoảng thời gian đều khi thiếu dữ liệu.

## So với bài báo gốc

Bài báo gốc báo cáo RNN MAE 0,190 và RMSE 0,250 sau chuẩn hóa [0,1], nhưng không tách CPU/RAM, không nêu rõ horizon hoặc tập container. Dù cùng thang số [0,1] và tỷ lệ chia 80/20, các trị số ở đây **chưa thể so trực tiếp** để tuyên bố GNN tốt hơn bài báo: tập con đã chọn hậu nghiệm, chỉ có 4 container, horizon 20 phút và cách tổng hợp sai số khác. Đây là thí nghiệm GNN thăm dò trên dữ liệu thật.

Chạy lại từ thư mục gốc: `python Lab/gnn/run_80_20.py`.
