# Hướng 2: biên dự phòng theo sai số dự báo RNN

Chạy từ `Lab`:

```bash
python3 safety_margin/compare_to_paper.py --q 0.95
```

`compare_to_paper.py` huấn luyện RNN đối chứng do Lab cài đặt trên dữ liệu đã chuẩn bị, dùng
`margin.py` để tính phân vị sai số một phía từ phần calibration riêng trong
validation, rồi chạy cùng bộ chọn máy với và không có biên. Report HTML/JSON
nằm trong `paper_comparison/`. `experiment.py` và `test_margin.py` dùng cho
thử nghiệm/kiểm thử riêng của hướng này.

Bài gốc không công bố biên dự phòng, độ bao phủ hoặc tỷ lệ quá tải nên không
thể tính mức cải thiện trực tiếp theo các chỉ số này. Report kiểm tra liệu
phần mở rộng có ích trên cùng test; mẫu mặc định chỉ có bốn container.
`--q` phải chốt trước khi xem test. Xem [hướng dẫn tổng](../README.md).
Để chạy mọi cohort đã quét, dùng
`python3 safety_margin/compare_full.py --q 0.95` từ `Lab`.
