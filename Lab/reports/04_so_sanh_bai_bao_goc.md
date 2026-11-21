# Đánh giá riêng ba hướng so với bài báo gốc

Bài báo *Time-Series Load Prediction for Cloud Resource Allocation Using
Recurrent Neural Networks* công bố RNN MAE **0,190**, RMSE **0,250** trong
[Bảng 1, hàng RNN(Ours), trang 4 của PDF người dùng cung cấp](../../Time-SeriesLoadPredictionforCloudResourceAllocationUsingRecurrentNeuralNetworks-1.pdf#page=4).
Bài báo cho biết dùng Alibaba Cluster Trace, chia 80/20 theo thời gian
và chuẩn hóa [0,1], nhưng thiếu phiên bản/tập mẫu, biến đích, horizon, cách
chuẩn hóa chi tiết và mã đánh giá. Bài báo cũng không công bố số định lượng về
quá tải, máy hoạt động hoặc di chuyển container. Do đó không thể lấy hiệu số
giữa số Lab và số công bố làm mức cải thiện trực tiếp.

## Hướng 1: chọn máy chủ từ dự báo

[Report HTML](../host_selection/paper_comparison/report.html) ·
[JSON](../host_selection/paper_comparison/report.json) ·
[mã chạy](../host_selection/compare_to_paper.py)

[Report 100 cohort](../host_selection/paper_comparison/preview.html) ·
[JSON 100 cohort](../host_selection/paper_comparison/preview.json) ·
[lệnh chạy toàn bộ](../host_selection/compare_full.py)

Trên tập thăm dò **4 container**, RNN đối chứng do Lab cài đặt có MAE 0,12660 và RMSE 0,17095
trên thang [0,1]. Đối chứng giữ nguyên vị trí có 8/8 máy-bước quá tải và thiếu
69 đơn vị RAM mô phỏng; RNN + chọn máy có 4/12 máy-bước quá tải, thiếu 50 đơn
vị RAM, dùng 4,667 thay vì 2,667 giờ máy và tạo 2 lần di chuyển trung bình
qua ba seed. Đối chứng giữ vị trí là thiết kế **mới của Lab**, không phải thuật
toán tác giả. Kết quả nhỏ này chưa chứng minh cải thiện hệ thống gốc.

Trên **100 cohort đầu** của catalog, giữ vị trí có 1.398/1.434 máy-bước quá
tải, thiếu 89.709 đơn vị RAM; RNN + chọn máy có 703/2.194 máy-bước quá tải,
thiếu 45.073 RAM. Đổi lại giờ máy tăng từ 478 lên 846,7 và có 264 lần di
chuyển. Đây là đánh đổi trong mô phỏng, không phải so với scheduler của tác giả.

## Hướng 2: biên theo sai số RNN

[Report HTML](../safety_margin/paper_comparison/report.html) ·
[JSON](../safety_margin/paper_comparison/report.json) ·
[mã chạy](../safety_margin/compare_to_paper.py)

[Report 100 cohort](../safety_margin/paper_comparison/preview.html) ·
[JSON 100 cohort](../safety_margin/paper_comparison/preview.json) ·
[lệnh chạy toàn bộ](../safety_margin/compare_full.py)

Trên cùng tập 4 container, q=0,95 tăng độ bao phủ CPU/RAM trung bình từ
0,563/0,500 lên 0,875/0,833. Quá tải, thiếu RAM, giờ máy và số di chuyển
**không thay đổi**. Đây là phân tích nhỏ về phần mở rộng; bài báo không có
chỉ số biên hoặc bao phủ để đối chiếu định lượng.

Trên **100 cohort đầu**, bao phủ tổng CPU/RAM tăng từ 0,498 lên 0,899, nhưng
máy-bước quá tải tăng từ 703/2.194 lên 729/2.193 và thiếu RAM tăng từ
45.073 lên 47.961 đơn vị mô phỏng. Chưa có bằng chứng biên giúp lập lịch tốt
hơn trong mẫu này.

## Hướng 3: GNN thay RNN

[Report HTML](../gnn/paper_comparison/report.html) ·
[JSON](../gnn/paper_comparison/report.json) ·
[mã chạy](../gnn/compare_to_paper.py)

Trên **2.067 cohort** cùng test, RNN đối chứng của Lab có MAE 0,09305, RMSE 0,13330;
GNN MAE 0,05764, RMSE 0,08261. MAE GNN giảm **38,1% so với RNN đối chứng của Lab**.
Đây là so sánh hợp lệ *trong giao thức Lab*. Không được viết “giảm 69,7% so
với bài báo” bằng cách trừ 0,05764 khỏi 0,190, vì điều kiện thử không được
xác minh là giống nhau.

## Kết luận có thể bảo vệ

- GNN cải thiện sai số dự báo tổng hợp so với RNN đối chứng trên cùng dữ liệu Lab.
- Thuật toán chọn máy và biên đã được chạy, nhưng chưa có phép đo định lượng
  tương ứng trong bài báo để chứng minh phần trăm cải thiện so với tác giả.
- Hướng 1/2 đã có phép thử 100 cohort đầu; đây là mẫu theo thứ tự catalog,
  chưa phải kết quả của toàn bộ 2.067 cohort. Hai lệnh full có checkpoint để
  hoàn thành phép đánh giá trước khi rút kết luận chung.

Xem [hướng dẫn chạy](../README.md) để tái tạo ba report riêng.
