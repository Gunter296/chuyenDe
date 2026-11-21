# Hướng 1: chuyển dự báo thành quyết định chọn máy chủ

## Suy luận

Sai số dự báo thấp không tự chứng minh hệ thống phân bổ tài nguyên tốt. Cần một quy tắc xác định máy đích và đo kết quả sau khi tải thực xuất hiện.

## Mục tiêu

Trả lời liệu cùng thuật toán chọn máy, dùng dự báo RNN có giảm quá tải hoặc số giờ máy so với dùng tải gần nhất hay không.

## Hướng làm

Tại mỗi bước, cộng dự báo CPU/RAM của container trên từng máy cho toàn horizon. Nếu máy có nguy cơ vượt ngưỡng, xét container có nhu cầu trội cao trước. Lọc máy đích đủ cả CPU/RAM ở mọi bước, chọn máy có phần dư chuẩn hóa nhỏ nhất; nếu không có, yêu cầu mở máy. Ghi nhận tải thiếu trong thời gian khởi động. Cố định giới hạn di chuyển và cooldown cho mọi phương án. So B0 với B1 qua ba seed trên cùng tập test.

## Ý nghĩa

Phương pháp biến một bài toán dự báo thành thuật toán vận hành có thể tái lập và kiểm tra. Nếu B1 cải thiện, bài báo có bằng chứng ở mức hệ thống; nếu không, kết quả chỉ ra dự báo chính xác hơn chưa chắc giúp lập lịch.

## Giới hạn

Mô phỏng chưa đo thời gian live migration, SLA, điện năng hay giá tiền thực. Ví dụ tổng hợp chỉ kiểm tra mã.
