# Chạy GNN tự động trên toàn bộ tệp container_usage.csv

Từ thư mục `Lab` trên máy Linux/VMware, **đặt SQLite trên ổ cục bộ của VM** để tránh tốc độ ghi rất chậm của `/mnt/hgfs`:

```bash
python3 gnn/run_full_auto.py --input data/container_usage.csv --output ~/gnn_full_run --progress-seconds 10
```

Nếu chạy từ thư mục gốc dự án trên Windows:

```powershell
python Lab/gnn/run_full_auto.py --input Lab/data/container_usage.csv --output Lab/gnn/full_run
```

Python cần `numpy` và `Pillow`. Script quét **mọi dòng** của CSV 1,56 GB và gộp theo `container_id`/ô 20 phút vào SQLite trên đĩa. Nó xét mọi cửa sổ 24 giờ, chọn container có ít nhất 90% quan sát trong phần train của cửa sổ, nhóm các container có thời điểm quan sát trùng nhau thành cohort 4 nút, rồi chạy GNN và mạng bỏ cạnh với seed 11/22/33. Trong từng cohort, 80% thời gian đầu dùng phát triển mô hình (train và validation), 20% cuối dùng test. CPU/RAM phần trăm được chia cho 100; cạnh chỉ dựng từ train.

Script in tiến độ ra stderr mỗi 10 giây: phần trăm byte CSV đã đọc, số dòng, tốc độ, ETA ước tính và trạng thái ghi SQLite. Sau nhập liệu, nó báo từng cửa sổ và cohort. **Ctrl+C một lần** sẽ dừng; chạy lại đúng lệnh với cùng nguồn, `--output` và `--bucket-seconds` để tiếp tục từ batch SQLite đã commit. Nếu thư mục output chứa cơ sở dữ liệu cũ không có checkpoint, script sẽ thông báo và quét lại từ đầu một lần. Nếu nguồn dữ liệu đổi, dùng thư mục output mới.

**Toàn bộ CSV được quét nhưng không phải mọi container đều được huấn luyện.** Container thưa, không đủ lịch sử, thiếu nhãn test hoặc không ghép được cohort bị loại. `report.json` ghi số dòng quét, số dòng hợp lệ, số ID, số ID đủ điều kiện, số ID thực được đánh giá, lý do loại từng cohort và MAE/RMSE tổng hợp có trọng số theo số container-cửa sổ test. `mae_comparison.png` hiển thị so sánh GNN với đối chứng bỏ cạnh.

Tùy chọn `--max-cohorts 10` dùng cho chạy thử nhanh; mặc định `0` chạy tất cả cohort đủ điều kiện. `--bucket-seconds`, `--window-seconds`, `--cohort-size` và `--min-coverage` cho phép thay đổi giao thức; ghi lại các giá trị này khi báo cáo kết quả. Script cần thêm dung lượng đĩa cho `usage.sqlite` và có thể chạy lâu trên toàn bộ dữ liệu.

Kết quả các cohort là phép đánh giá GNN trên trace thật theo nhiều nhóm, **không phải một đồ thị duy nhất của toàn bộ 71.476 container** và không phải tái lập chính xác MAE/RMSE của bài báo gốc. Chỉ có thể so trực tiếp GNN với mạng bỏ cạnh khi hai bên dùng cùng cohort và test.
