# So sánh rút gọn từ lần chạy toàn cohort đã có

Bảng dưới lấy từ kết quả lịch sử `gnn_host_full/full_report.json` (138 cohort, 551 container duy nhất, 3.212 cặp container–cửa sổ test). Mã hiện đã được rút gọn; bảng này dùng số đã chạy trước khi sửa mã, chưa phải kết quả của lần chạy mới.

| Phương án | MAE CPU | RMSE CPU | MAE RAM | RMSE RAM | Quá tải mô phỏng | Giờ máy cộng qua cohort | Di chuyển |
|---|---:|---:|---:|---:|---:|---:|---:|
| RNN dự báo + thuật toán chọn máy | 0,06795 | 0,10754 | 0,36812 | 0,47320 | 99,63% | 1.070,67 | 0 |
| GNN chỉ dự báo | 0,08179 | 0,11387 | 0,20828 | 0,26226 | Không áp dụng | Không áp dụng | Không áp dụng |
| RNN bài báo gốc | MAE chung 0,190 | RMSE chung 0,250 | Không tách | Không tách | Không công bố | Không công bố | Không công bố |

**Nguồn:** hàng RNN + chọn máy là `methods.B2`; hàng GNN chỉ lấy `methods.B4` → `mae_by_horizon_cpu_ram`, `rmse_by_horizon_cpu_ram` và **bỏ toàn bộ `system`** vì yêu cầu hiện tại là GNN chỉ dự báo. Bài báo gốc: *Time-Series Load Prediction for Cloud Resource Allocation Using Recurrent Neural Networks*, trang 4, Bảng 1, hàng RNN(Ours). MAE trong bảng là trung bình 5 horizon; RMSE là căn của trung bình 5 RMSE horizon bình phương. Các số dự báo của hai hàng đầu cùng test nên so được; số bài báo gốc dùng giao thức khác nên chỉ là mốc tham khảo, **không tính phần trăm cải thiện**.

Trong lần chạy này, GNN có MAE RAM thấp hơn RNN 43,4% nhưng MAE CPU cao hơn 20,4%. Chỉ RNN + chọn máy có chỉ số mô phỏng; do đó bảng **không kiểm chứng GNN cải thiện chọn máy**. Tỷ lệ quá tải 99,63% phản ánh giả định tải nền cộng với tải container, không phải số đo trực tiếp của cụm Alibaba. Tổng giờ máy cộng nhiều cohort có thể trùng khoảng thời gian, không phải giờ máy thực của cả cụm.
