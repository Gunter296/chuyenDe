# Tổng quan kế hoạch thực nghiệm

## Bối cảnh

Bài báo gốc so sánh mô hình dự báo tải bằng MAE/RMSE, nhưng chưa mô tả đầy đủ cách biến dự báo thành quyết định đặt container và chưa đo chỉ số quá tải máy. Bài báo đề xuất GNN cho nghiên cứu tiếp theo. Ba hướng trong Lab lần lượt giải quyết khoảng trống về quyết định, bất định của dự báo và quan hệ giữa các container.

## Ma trận thí nghiệm

| Mã | Mô hình nhu cầu | Biên | Ý nghĩa |
|---|---|---|---|
| B0 | Tải gần nhất | Không | Đối chứng phản ứng |
| B1 | RNN | Không | Tác động của dự báo lên chọn máy |
| B2 | RNN | Phân vị sai số | Tác động của biên dự phòng |
| B3 | GNN | Không | Tác động của thông tin đồ thị |
| B4 | GNN | Phân vị sai số | Kết hợp ba hướng |

Mọi phương án phải dùng cùng trace, mốc thời gian, dung lượng máy và thuật toán lập lịch. Chia theo thời gian 60% train, 20% validation, 20% test; không dùng tương lai để tạo dự báo hoặc đồ thị. Chạy ba seed 11, 22, 33. Báo cáo MAE/RMSE theo CPU/RAM và bước dự báo; quá tải, mức thiếu tài nguyên, giờ máy, di chuyển và thời gian quyết định.

## Trạng thái bằng chứng

Đã có `Lab/data/container_usage.csv` và đã chạy B0–B4 trên cùng một tập con 4 container của trace Alibaba thật, ba seed. Chi tiết ở [bảng so sánh tổng hợp](04_so_sanh_bai_bao_goc.md) và báo cáo từng hướng. Tập con được chọn hậu nghiệm do trace thưa; chỉ có 5 cửa sổ train và 4 chu kỳ lập lịch, nên đây là phép thử thăm dò. Trong phép thử này, các chỉ số lập lịch B0–B4 trùng nhau; **chưa có bằng chứng cải thiện** so với đối chứng hoặc bài báo gốc.
