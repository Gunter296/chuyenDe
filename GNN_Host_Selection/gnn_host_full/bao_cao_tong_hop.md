# Tổng hợp kết quả chạy toàn bộ cohort đủ điều kiện

**Nguồn:** `full_report.json`, `cohorts.jsonl`, `run_config.json` trong cùng thư mục. Báo cáo này phân tích lần chạy đã hoàn tất (`complete: true`) của chế độ `--all-cohorts`. “Toàn bộ” nghĩa là chương trình quét toàn bộ bảng `container_usage.csv` rồi đánh giá những cohort vượt qua điều kiện dữ liệu; **không phải mọi container đều được đưa vào mô hình**.

## 1. Phạm vi dữ liệu

| Chỉ số | Kết quả | Nguồn/cách hiểu |
|---|---:|---|
| Dòng usage hợp lệ đã nhập | 31.666.484 | `source_valid_rows` |
| Container xuất hiện trong các khoảng xét | 33.519 | `containers_seen_in_windows` |
| Container đạt độ phủ train ≥90% ở ít nhất một khoảng | 25.214 | `containers_meeting_train_coverage` |
| Cohort hoàn tất | 138 | `cohorts_evaluated`; mỗi cohort có 4 container |
| Container duy nhất thực sự được đánh giá | 551 | `containers_evaluated`; một container xuất hiện ở hơn một cohort/khoảng |
| Cặp container–cửa sổ test | 3.212 | `test_container_windows`; tương ứng 803 cửa sổ cohort vì mỗi cohort có 4 container |
| Cửa sổ test mỗi cohort | 5–10 | Đếm từ 138 dòng `cohorts.jsonl`: 76 cohort có 5, 36 có 6, 12 có 7, 6 có 8, 5 có 9, 3 có 10 |

Mỗi khoảng dài 72 ô × 20 phút = **24 giờ**, chia 60/20/20 theo thời gian; lịch sử 12 ô và dự báo **5 bước × 20 phút = 100 phút**. Chỉ có một seed (`11`), 15 epoch, GRU 24 chiều ẩn. Số 203.414 trường hợp thiếu độ phủ train và 43.523 trường hợp thiếu cửa sổ là **lượt container–khoảng**, không phải số container duy nhất bị loại; thêm 121 lượt không ghép được cohort. Các số loại không được cộng để suy ra số ID bị loại.

## 2. Sai số dự báo

| Phương án | MAE CPU | RMSE CPU | MAE RAM | RMSE RAM |
|---|---:|---:|---:|---:|
| B0 · RNN, chỉ dự báo | 0,06795 | 0,10754 | 0,36812 | 0,47320 |
| **B1 · tải gần nhất** | **0,02260** | **0,05301** | **0,01080** | **0,03440** |
| B2 · RNN + chọn máy | 0,06795 | 0,10754 | 0,36812 | 0,47320 |
| B3 · NoGraph + chọn máy | 0,06795 | 0,10754 | 0,36812 | 0,47320 |
| B4 · GNN động + chọn máy | 0,08179 | 0,11387 | 0,20828 | 0,26226 |
| Ablation · FixedGraph | 0,08127 | 0,11343 | 0,20664 | 0,26109 |

**Nguồn và phép tính:** `full_report.json → methods → [mã] → mae_by_horizon_cpu_ram` và `rmse_by_horizon_cpu_ram`. Bảng lấy trung bình 5 MAE horizon; RMSE tổng hợp là `sqrt(trung bình của 5 RMSE_h²)`. Trong report gốc, mỗi horizon đã được gộp qua cohort bằng trọng số số container–cửa sổ test; RMSE gộp từ bình phương trước khi lấy căn. Đơn vị là tỷ lệ sử dụng từ CSV chia 100, **không phải core/GB**.

**Diễn giải:** GNN B4 giảm MAE RAM khoảng **43,4%** so với RNN B2, nhưng MAE CPU **cao hơn 20,4%**. FixedGraph hơi tốt hơn GNN động ở cả CPU và RAM; do đó dữ liệu này **không ủng hộ lợi ích của cạnh động**. Quan trọng hơn, B1 tải gần nhất có MAE/RMSE thấp nhất ở cả CPU và RAM. B0/B2 là cùng dự báo RNN; B3 trùng B2 vì trong mã hiện tại NoGraph và RNN cùng GRU/head, cùng seed và không truyền tin. Vì vậy không diễn giải B2–B3 là hai đối chứng độc lập.

## 3. Kết quả mô phỏng chọn máy

| Phương án | Máy–bước quá tải / đang hoạt động | Tỷ lệ quá tải | Tổng giờ máy qua cohort | Di chuyển |
|---|---:|---:|---:|---:|
| B1 · tải gần nhất | 3.187 / 3.212 | 99,22% | 1.070,67 | 3 |
| B2 · RNN | 3.200 / 3.212 | 99,63% | 1.070,67 | 0 |
| B3 · NoGraph | 3.200 / 3.212 | 99,63% | 1.070,67 | 0 |
| B4 · GNN động | 3.200 / 3.212 | 99,63% | 1.070,67 | 0 |
| FixedGraph | 3.200 / 3.212 | 99,63% | 1.070,67 | 0 |

**Nguồn và phép tính:** `full_report.json → methods → [mã] → system`. Số máy–bước quá tải = `overload_rate × active_host_steps`, làm tròn về số nguyên; `active_host_steps = 3212` ở mỗi phương án. Tổng giờ máy là tổng **qua 138 lần mô phỏng cohort**, có thể chứa các khoảng thời gian chồng nhau; **không phải 1.070,67 giờ máy thực của cụm Alibaba**. B0 không chạy bộ chọn máy nên không có chỉ số hệ thống.

Tỷ lệ quá tải gần 100% ở mọi phương án là dấu hiệu mô phỏng đang ở trạng thái bão hòa; B1 chỉ thấp hơn B2/B4 **13 máy–bước trên 3.212**, tương đương khoảng **0,40 điểm phần trăm**. Giờ máy không đổi, còn số di chuyển gần như bằng 0. Vì vậy lần chạy này **không chứng minh thuật toán hay GNN cải thiện vận hành**. Cần kiểm tra mô hình tải nền, dung lượng và điều kiện chọn máy trước khi diễn giải kết quả như hiệu năng hệ thống.

Mô phỏng dùng một pool chung gồm 8 máy (`m_1`, `m_10`, `m_100`, `m_1000` đến `m_1004`); dung lượng chuẩn hóa của cả 8 đều `(1,1)`. Tải nền máy thiếu **17,34%** số ô và được điền bằng quan sát quá khứ, phần đầu không có quan sát dùng 0. Mã cộng tỷ lệ tải nền máy với tỷ lệ tải container như một **giả định kịch bản**. Trong chế độ toàn cohort, pool máy chung không được loại máy gốc của từng container nên còn nguy cơ đếm trùng tải. Các giới hạn này có thể góp phần tạo tỷ lệ quá tải rất cao.

## 4. Chi phí mô hình

| Mô hình | Huấn luyện trung bình ± độ lệch chuẩn (giây/cohort) | Suy luận trung bình ± độ lệch chuẩn (ms/bước) |
|---|---:|---:|
| RNN | 0,0918 ± 0,0717 | 0,3429 ± 0,0442 |
| NoGraph | 0,0911 ± 0,0715 | 0,3379 ± 0,0281 |
| FixedGraph | 0,0985 ± 0,0780 | 0,3663 ± 0,0371 |
| GNN động | 0,1131 ± 0,0862 | 0,3808 ± 0,0235 |

Nguồn: `full_report.json → model_cost`. GNN động huấn luyện lâu hơn RNN khoảng **23%** và suy luận lâu hơn khoảng **11%** theo trung bình cohort; đây là thời gian trên cohort 4 container trong môi trường chạy cụ thể, chưa ngoại suy được cho cụm lớn.

## 5. Trả lời câu hỏi nghiên cứu trong phạm vi lần chạy

- **RQ1:** B1 tải gần nhất có sai số thấp hơn RNN và tỷ lệ quá tải mô phỏng thấp hơn 0,40 điểm phần trăm. Không có bằng chứng RNN giúp chọn máy hơn cách dùng tải hiện tại trong cấu hình này.
- **RQ2:** GNN cải thiện dự báo RAM so với RNN nhưng làm xấu CPU; cả hai cho cùng tỷ lệ quá tải 99,63%. Không thấy lợi ích hệ thống của chênh lệch sai số dự báo.
- **RQ3:** GNN tốn thêm thời gian huấn luyện/suy luận trên cohort 4 nút, còn giờ máy và di chuyển không thay đổi. FixedGraph hơi tốt hơn GNN động trong sai số dự báo.

Các nhận xét này là **mô tả một lần chạy**, không phải khẳng định thống kê: chỉ một seed, cohort có thể trùng container, và giả định tải nền/dung lượng hiện chưa được hiệu chuẩn theo đơn vị vật lý. Bài báo RNN gốc báo MAE 0,190 và RMSE 0,250 chung trên cách chuẩn hóa/tập test khác; **không so chênh lệch trực tiếp** với các cột CPU/RAM tại đây. Bài gốc cũng không có chỉ số chọn máy để đối chiếu.

## 6. Việc cần xác minh trước khi viết kết luận bài báo

1. Kiểm tra vì sao B1 vượt xa các mô hình học, đặc biệt RAM; rà lại biến đích, chuẩn hóa, huấn luyện và phân bố tải theo thời gian.
2. Chạy nhiều seed, khoảng thời gian và kiểm tra độ nhạy với kích thước cohort, dung lượng máy, tải nền, `alpha`/`beta` và trễ mở máy.
3. Hiệu chuẩn tải container và tải máy về cùng đơn vị, tránh đếm trùng máy gốc của container; báo thêm thiếu CPU/RAM và độ bao phủ biên an toàn.
4. Đánh giá theo khoảng thời gian/cohort độc lập và khoảng tin cậy thay vì coi 3.212 cửa sổ chồng nhau là 3.212 quan sát độc lập.
