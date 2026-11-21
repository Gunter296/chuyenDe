# Hướng 3: mạng nơ-ron đồ thị cho dự báo tải

## Suy luận

RNN dự báo từng chuỗi có thể bỏ qua biến động đồng thời của các container. Quan hệ giữa container có thể giúp nhận biết tải chung của một nhóm, nhưng chỉ có ý nghĩa nếu cạnh đồ thị được tạo từ thông tin sẵn có trước quyết định.

## Mục tiêu

Kiểm tra liệu thêm truyền thông điệp giữa container có giảm MAE/RMSE và cải thiện kết quả lập lịch so với mô hình không dùng cạnh, rồi so B3/B4 với B1/B2. Đồng thời đo chi phí tính toán.

## Hướng làm

Tạo đồ thị container–container từ tương quan CPU dương trên train; chỉ giữ tối đa hai láng giềng mỗi nút. Với mỗi nút, đầu vào là lịch sử CPU/RAM. Một lớp GNN học kết hợp đặc trưng nút với tổng có trọng số của láng giềng, sau đó dự báo nhiều bước. Dùng cùng kiến trúc nhưng ma trận cạnh bằng 0 làm ablation. Chạy cùng seed, tập test và bộ chọn máy; B4 cộng biên từ sai số calibration của chính GNN.

## Ý nghĩa

Thí nghiệm tách được câu hỏi liệu quan hệ đồ thị thực sự mang thông tin hữu ích cho dự báo và lập lịch. Nếu đồ thị không cải thiện, bài báo vẫn có kết luận có giá trị về giới hạn của quan hệ dựa trên tương quan tải.

## Giới hạn

Tương quan tải không chứng minh phụ thuộc nhân quả; cạnh cố định có thể lỗi thời. Kiến trúc hiện là GNN tối thiểu một lớp, chưa dùng đồ thị container–máy hay đồ thị động. Đối chứng bỏ cạnh kiểm tra riêng tác động truyền thông điệp; so sánh với RNN còn phải tính khác biệt kiến trúc và tham số. Chưa có kết quả Alibaba.
