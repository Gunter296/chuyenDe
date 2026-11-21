# ChuyenDe

Kho mã gồm hai nhánh thí nghiệm về dự báo tải container và chọn máy. `Lab/` tổ chức mã theo từng hướng nghiên cứu; `GNN_Host_Selection/` là một pipeline riêng để so sánh dự báo và mô phỏng chọn máy. Hai nhánh dùng cấu hình và scheduler khác nhau, nên cần đọc README của từng thư mục trước khi so kết quả.

## Các thư mục chính

| Thư mục | Nội dung |
|---|---|
| [`Lab/`](Lab/README.md) | Bộ thí nghiệm theo các hướng RNN, GNN, chọn máy và biên dự phòng. README trong thư mục mô tả cách chạy và giới hạn chung. |
| [`Lab/gnn/`](Lab/gnn/README.md) | Thí nghiệm dự báo GNN, tạo đồ thị, so sánh với RNN và chạy trên các cohort. |
| [`Lab/host_selection/`](Lab/host_selection/README.md) | Huấn luyện RNN và mô phỏng chọn máy theo ngưỡng tải, giới hạn di chuyển và cooldown. |
| `Lab/safety_margin/` | Tính biên dự phòng từ residual validation và đánh giá ảnh hưởng của biên lên mô phỏng. |
| `Lab/reports/` | Báo cáo phương pháp, kết quả và nhận xét về giới hạn của các thí nghiệm Lab. |
| `GNN_Host_Selection/` | Pipeline độc lập dùng RNN/GNN để dự báo; RNN được nối với bộ chọn máy cost-aware, còn GNN trong pipeline chính hiện được đánh giá dự báo. Xem [README](GNN_Host_Selection/README.md) để biết cấu hình và giao thức. |
| `GNN_Host_Selection/gnn_compare_full/` | Cấu hình, checkpoint cohort, báo cáo và biểu đồ của lượt so sánh toàn cohort. |
| `GNN_Host_Selection/gnn_host_full/` | Báo cáo và checkpoint của lượt mô phỏng host toàn cohort trước đó. |
| `GNN_Host_Selection/results/` | Báo cáo tổng hợp và biểu đồ kết quả của lượt chạy nhỏ. |
| `GNN_Host_Selection/tests/` | Kiểm thử cho scheduler và pipeline. |

## Dữ liệu và tệp sinh khi chạy

Các trace CSV không được lưu trong repo vì dung lượng lớn. Đặt dữ liệu theo các đường dẫn được mô tả trong README của từng pipeline; ví dụ các script trong `GNN_Host_Selection/` mặc định tham chiếu tới `Lab/data/`. Các chỉ mục SQLite, cache Python và môi trường ảo cũng bị loại khỏi Git. Lần chạy toàn dữ liệu sẽ tạo lại SQLite cục bộ nếu chưa có chỉ mục phù hợp.

## Chạy và đọc kết quả

- Bắt đầu từ README của nhánh muốn chạy: [`Lab`](Lab/README.md) hoặc [`GNN_Host_Selection`](GNN_Host_Selection/README.md).
- Các phương án trong `Lab/host_selection/` và `GNN_Host_Selection/` sử dụng scheduler khác nhau; không gộp số liệu nếu cohort, split hoặc chính sách chọn máy không giống nhau.
- Xem các báo cáo trong `Lab/reports/` và các thư mục kết quả để biết giao thức, số lượng cohort và cách diễn giải chỉ số.
