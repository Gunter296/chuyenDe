# Kết quả hướng 2: biên dự phòng theo sai số

## Nguồn và phạm vi

Đã chạy trên `Lab/data/container_usage.csv`, qua cùng tập trích `Lab/host_selection/prepared_shared.json` và các dự báo RNN trong `Lab/host_selection/results_real/comparison_seed_{11,22,33}.json`. Tập trích gồm **4 container, 43 ô thời gian 20 phút** trong khoảng [86400, 138000) giây của trace; train/validation/test lần lượt 25/9/9 ô. Độ phủ train của 4 container là 100%, 92%, 92%, 96%. Do thiếu dữ liệu đồng thời, test lập lịch chỉ còn **5 ô liên tiếp**, trong đó `run.py` đánh giá 4 quyết định; có 12 máy-ô sẵn sàng được tính vào tỷ lệ quá tải. Mẫu này được chọn sau khi khảo sát tính khả thi, nên chỉ là phân tích thăm dò, không đại diện toàn bộ Alibaba.

## Thiết kế đã khóa

- B1 và B2 dùng đúng cùng dự báo RNN, cùng test, dung lượng máy, vị trí ban đầu và bộ lập lịch. B2 chỉ cộng biên dự phòng.
- Nửa đầu validation chọn epoch; nửa sau hiệu chỉnh sai số một phía `tải thực − dự báo`. Không dùng test để tạo biên.
- Phân vị **q = 0,95** được định trước; biên được tách theo CPU/RAM và horizon. Vì trace trích có ô 20 phút, horizon 1 ở đây là 20 phút, lệch với mục tiêu 5 phút ban đầu.
- Kết quả gốc của từng seed và tỷ lệ bao phủ nằm trong `Lab/safety_margin/real_result.json`.

## Kết quả đo được

| Chỉ số | B1 RNN, không biên | B2 RNN, biên q=0,95 |
|---|---:|---:|
| Máy-ô quá tải / máy-ô sẵn sàng | 4 / 12 | 4 / 12 |
| Tỷ lệ quá tải | 33,3% | 33,3% |
| Tổng thiếu CPU | 0 | 0 |
| Tổng thiếu RAM | 50 đơn vị mô phỏng | 50 đơn vị mô phỏng |
| Giờ máy hoạt động, kể cả máy đang khởi động | 4,667 | 4,667 |
| Di chuyển / 100 container-giờ | 37,5 | 37,5 |

Các chỉ số lập lịch trên giống nhau ở cả seed 11, 22 và 33. Tuy nhiên, **biên đã thay đổi dự báo**: biên CPU/RAM tương ứng theo seed là (8,14; 28,11), (7,37; 12,69), (8,16; 23,96). Tỷ lệ quan sát không vượt dự báo trên test tăng từ CPU 31,25–68,75% và RAM 50% ở B1 lên CPU 87,5% và RAM 68,75–93,75% ở B2. Tỷ lệ bao phủ này được đo trên 16 dự báo container-ô mỗi seed, nên rất không ổn định.

## Diễn giải

Trong mẫu thử này, biên dự phòng làm dự báo thận trọng hơn nhưng **không làm thay đổi các chỉ số lập lịch đo được**. Không thể kết luận biên vô ích nói chung: chỉ có 4 quyết định test, 2 máy ban đầu, giới hạn di chuyển và độ trễ mở máy; các ràng buộc này có thể khiến dự báo thay đổi mà quyết định cuối không đổi. Đồng thời, việc thêm biên không chứng minh giảm quá tải nếu số giờ máy và vị trí container không đổi.

So với bài báo gốc, đây là chỉ số hệ thống mới; bài gốc chỉ cho MAE 0,190 và RMSE 0,250 của RNN, không cung cấp tỷ lệ quá tải, giờ máy hoặc số di chuyển trên cùng điều kiện. Không so trực tiếp các số MAE/RMSE gốc với chỉ số trên vì khác đơn vị, mẫu và bước thời gian.

## Giới hạn và bước tiếp theo

Mẫu chỉ 4 container và ô 20 phút; phương pháp dự kiến 5 phút. Phân vị từ phần calibration ngắn có độ bất định cao, không có bảo đảm conformal. CPU/RAM là phần trăm sử dụng từ trace và được dùng như đơn vị mô phỏng, chưa chứng minh tương ứng với core/GB hay dung lượng thật của máy. Cần trace liên tục rộng hơn, nhiều container hơn và khóa quy tắc chọn mẫu trước khi xem test để đánh giá thực sự hiệu quả của B2.
