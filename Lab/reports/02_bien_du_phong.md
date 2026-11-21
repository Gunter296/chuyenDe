# Hướng 2: biên dự phòng theo sai số dự báo

## Suy luận

Sai số dự báo thiếu làm bộ lập lịch ghép quá nhiều container lên một máy. Cộng biên cố định cho mọi tình huống có thể lãng phí tài nguyên, trong khi sai số CPU và RAM, cũng như từng horizon, khác nhau.

## Mục tiêu

Giảm tỷ lệ máy quá dung lượng và tổng mức thiếu tài nguyên so với B1, đồng thời định lượng số giờ máy và số lần di chuyển tăng thêm.

## Hướng làm

Tính sai số một phía `e = tải thực - dự báo` từ phần calibration sau khi đã chọn epoch. Với mỗi CPU/RAM và horizon, lấy `max(0, Q_q(e))`, cộng vào dự báo rồi chạy đúng bộ chọn máy của hướng 1. Khảo sát q = 0.90, 0.95, 0.99; chọn q bằng quy tắc đặt trước hoặc validation độc lập, khóa trước test. So B1 với B2, và nên thêm biên cố định làm đối chứng.

## Ý nghĩa

Biên dự phòng kết nối độ bất định của mô hình với quyết định phân bổ. Đồ thị quá tải–giờ máy cho biết mức an toàn đạt được bằng bao nhiêu tài nguyên bổ sung, hữu ích hơn chỉ báo MAE/RMSE.

## Giới hạn

Phân vị sai số thực nghiệm không có bảo đảm bao phủ khi phân phối tải đổi theo thời gian. Calibration dùng chung nhiều container có thể che giấu container hiếm đột biến. Chưa có kết quả Alibaba để xác nhận lợi ích.
