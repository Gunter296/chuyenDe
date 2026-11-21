# Trọng tâm cải tiến theo Future Work của bài báo gốc

## Mục tiêu chính

Kiểm chứng liệu GNN cải thiện **độ chính xác dự báo tải tổng thể** so với mô hình không dùng đồ thị, rồi liệu dự báo đó giúp thuật toán lập lịch ra quyết định tốt hơn. CPU và RAM là hai thành phần của cùng bài toán dự báo đa tài nguyên; không dùng việc một thành phần thắng thành phần kia làm kết luận chính.

## Chỉ số kết luận

1. Chuẩn hóa CPU và RAM theo cùng giao thức train, tính MAE và RMSE tổng hợp trên mọi container, thời điểm, horizon và hai tài nguyên. Báo riêng CPU/RAM để giải thích nguồn gốc thay đổi.
2. So GNN với RNN và cùng mạng bỏ cạnh trên đúng cohort và bước test, qua ba seed.
3. Đưa từng dự báo vào cùng bộ chọn máy; đo quá tải, tổng thiếu tài nguyên, giờ máy, di chuyển và độ trễ quyết định.
4. Chỉ kết luận cải thiện theo Future Work khi chỉ số dự báo tổng hợp tốt hơn đối chứng **và** ít nhất một chỉ số lập lịch tốt hơn mà các chỉ số còn lại không suy giảm đáng kể. Chốt ngưỡng đánh đổi trên validation.

## Thứ tự tối ưu mô hình

Giữ mô hình hiện tại làm mốc. Thử (a) cạnh theo cùng ứng dụng/máy từ metadata, (b) cạnh học được nhưng thưa, (c) mã hóa thời gian RNN/GRU trước truyền thông điệp, (d) attention như một đối chứng riêng. Mỗi lần chỉ đổi một thành phần, chọn trên validation và khóa trước test. Ghi cả chi phí huấn luyện/suy luận.

## Giới hạn hiện tại

Output full_run mới so GNN với mạng bỏ cạnh về dự báo; chưa có RNN baseline và kết quả lập lịch trên toàn bộ cohort. Vì vậy kết luận Future Work **chưa đạt**. Lập lịch đa nền tảng là hướng khác của tác giả, chưa được hiện thực trong Lab.
