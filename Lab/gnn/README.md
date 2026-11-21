# Hướng 3: GNN thay RNN trong dự báo tải

Lệnh và cách đọc report so với RNN bài báo nằm ở [Lab/README.md](../README.md).
Điểm vào chính là `compare_to_paper.py`; `run_full_auto.py` tạo catalog/SQLite
từ CSV đầy đủ; `run_80_20.py`, `graph_model.py` và `forecast_core.py` cung
cấp mô hình GNN. RNN và hàm nạp cohort dùng chung nằm trong
`host_selection/cohort_baseline.py`. Checkpoint GNN/RNN hoàn chỉnh từ lần thử
trước được giữ lại để tạo report mới, chỉ lấy kết quả dự báo RNN/GNN.

Bài gốc **chưa thử GNN**. Report mới đo GNN so với RNN đối chứng do Lab cài đặt trên đúng cùng
cohort và thời điểm test, còn 0,190/0,250 chỉ là số tham khảo đã công bố.
