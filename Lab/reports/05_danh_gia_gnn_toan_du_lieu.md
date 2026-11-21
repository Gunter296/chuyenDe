# Đánh giá output GNN quét toàn bộ CSV

Nguồn kiểm tra: `Lab/gnn/full_run/gnn_full_run/report.json`, `usage.sqlite` và `mae_comparison.png`. SQLite xác nhận checkpoint ở đúng cuối tệp (1.563.196.258 byte).

## Phạm vi dữ liệu

| Chỉ số | Giá trị |
|---|---:|
| Dòng CSV đã quét | 31.670.001 |
| Dòng hợp lệ | 31.666.484 |
| Bản ghi container–ô thời gian sau gộp | 14.312.612 |
| Container khác nhau trong usage | 33.519 |
| Container đạt điều kiện train ở ít nhất một cửa sổ | 25.214 |
| Container thực được đánh giá | 7.386 |
| Cohort đánh giá được / bị loại | 2.067 / 8.979 |
| Container–cửa sổ test có trọng số | 24.480 |

Mặc định: ô 20 phút, cửa sổ 24 giờ, cohort 4 container, tối thiểu 90% độ phủ ở train, seed 11/22/33, chia 80% phát triển mô hình và 20% test. Dù đã đọc toàn bộ CSV, chỉ khoảng 22% container trong bảng usage được đánh giá. Phần lớn cohort bị loại vì không có đủ cửa sổ validation/test đồng thời. Số bước test của mỗi cohort từ 1 đến 13; 653/2.067 cohort chỉ có **một** bước test.

## GNN so với cùng mạng bỏ cạnh

Các số là trung bình cohort có trọng số theo số container–cửa sổ test, trên thang phần trăm sử dụng chia 100:

| Chỉ số | Bỏ cạnh | GNN | GNN trừ bỏ cạnh |
|---|---:|---:|---:|
| MAE CPU | 0,061228 | 0,058229 | **−0,003000** |
| RMSE CPU | 0,077011 | 0,073979 | **−0,003032** |
| MAE RAM | 0,060312 | 0,060429 | +0,000117 |
| RMSE RAM | 0,070906 | 0,071264 | +0,000359 |

GNN giảm MAE CPU khoảng 4,9% và RMSE CPU khoảng 3,9% so với đối chứng bỏ cạnh. MAE CPU tốt hơn ở 1.336 cohort, kém hơn ở 711 và hòa ở 20. Chênh lệch MAE CPU theo tám cửa sổ ngày đều âm. RAM gần như không đổi và hơi kém hơn: MAE RAM tốt hơn ở 995 cohort, kém hơn ở 1.052, hòa 20. [Biểu đồ so sánh MAE](../gnn/full_run/gnn_full_run/mae_comparison.png).

## Kết luận về cải tiến

**Có tín hiệu cải thiện dự báo CPU** do cạnh đồ thị trong giao thức này. Chưa có bằng chứng cải thiện RAM, và output hiện không đo quá tải, giờ máy, di chuyển hoặc chi phí suy luận/huấn luyện trên toàn bộ cohort. Vì vậy chưa thể kết luận GNN tối ưu phân bổ tài nguyên hoặc cải thiện bài báo gốc về hiệu năng hệ thống.

Bài báo gốc báo RNN MAE 0,190 và RMSE 0,250 chung sau chuẩn hóa, không nêu đủ cách chọn mẫu, horizon hay cách gộp CPU/RAM. Output mới là GNN so với cùng mạng bỏ cạnh, tách CPU/RAM trên các cohort 4 container với horizon 20 phút. Không được so trực tiếp trị số 0,058229 với 0,190 để tuyên bố vượt bài gốc.

## Điều cần làm trước khi gọi là kết quả mạnh

1. Bổ sung RNN baseline trên đúng các cohort và bước test GNN dùng, cùng chuẩn hóa [0,1].
2. Báo số bước test tối thiểu theo cohort; chạy kiểm tra độ nhạy khi yêu cầu ít nhất 5 hoặc 10 bước, vì nhiều cohort hiện chỉ có 1.
3. Báo kết quả theo ngày và khoảng biến thiên, rồi đánh giá thống kê theo nhóm thời gian thay vì coi 2.067 cohort là 2.067 quan sát độc lập.
4. Đưa dự báo GNN qua cùng bộ chọn máy và đo quá tải, giờ máy, di chuyển và thời gian suy luận.
5. Xác minh đơn vị CPU/RAM và dung lượng máy trước khi tuyên bố hiệu quả vật lý trên cụm Alibaba.
