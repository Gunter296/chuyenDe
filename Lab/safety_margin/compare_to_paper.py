"""Isolated RNN versus RNN plus a prespecified residual margin."""
import argparse
import html
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host_selection"))
from train_rnn import build
from margin import compare

PAPER = {"mae": 0.190, "rmse": 0.250,
         "source": "User-provided article PDF, page 4, Table 1, RNN(Ours) row"}


def average_schedule(rows, key):
    fields = ("overloaded_host_steps", "observed_host_steps", "cpu_shortfall",
              "ram_shortfall", "active_host_hours", "moves")
    return {field: mean(row[key][field] for row in rows) for field in fields}


def average_coverage(rows, key):
    return [mean(row[key][0][resource] for row in rows) for resource in (0, 1)]


def run(prepared, output, q=.95, seeds=(11, 22, 33), epochs=30):
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        baseline = build(prepared, epochs=epochs, seed=seed)
        rows.append({"seed": seed, "rnn_forecast": baseline["metadata"],
                     **compare(baseline, q)})
    a = average_schedule(rows, "B1")
    b = average_schedule(rows, "B2")
    ca = average_coverage(rows, "empirical_test_coverage_B1")
    cb = average_coverage(rows, "empirical_test_coverage_B2")
    result = {"direction": "safety_margin", "paper_published": PAPER,
              "rnn_origin": "Lab implementation trained on its own sample; not author code or weights",
              "rnn_without_margin": a, "rnn_with_margin": b,
              "coverage_cpu_ram_without_margin": ca,
              "coverage_cpu_ram_with_margin": cb,
              "q_prespecified": q, "seeds": list(seeds),
              "source": prepared.get("source"),
              "comparison_status": "Paper has no published margin, coverage or scheduling metrics."}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    output.write_text(f"""<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Hướng 2: biên dự phòng</title><style>body{{font:16px system-ui;max-width:940px;margin:36px auto;padding:0 20px}}td,th{{border:1px solid #bbb;padding:8px}}table{{border-collapse:collapse}}th{{background:#eef}}</style>
<h1>Hướng 2: biên dự phòng cho dự báo RNN</h1>
<p>Bài báo công bố RNN MAE 0,190; RMSE 0,250; không công bố biên dự phòng,
độ bao phủ hay chỉ số lập lịch. Chỉ thay đổi: cộng phân vị sai số validation
q={q:.2f} vào dự báo RNN; cùng test và thuật toán chọn máy.</p>
<p>Nguồn 0,190/0,250: <a href="../../../Time-SeriesLoadPredictionforCloudResourceAllocationUsingRecurrentNeuralNetworks-1.pdf#page=4">PDF bạn cung cấp, trang 4, Bảng 1, hàng RNN(Ours)</a>; được nhập thủ công làm tham khảo, không phải kết quả chạy lại mã tác giả.</p>
<p>RNN đối chứng do Lab tự cài đặt và huấn luyện; không dùng mã, trọng số hoặc tập test gốc của tác giả.</p>
<table><tr><th>Phương án</th><th>Bao phủ CPU/RAM</th><th>Quá tải / máy-bước</th><th>Thiếu CPU/RAM</th><th>Giờ máy</th><th>Di chuyển</th></tr>
<tr><td>RNN không biên</td><td>{ca[0]:.3f}/{ca[1]:.3f}</td><td>{a['overloaded_host_steps']:.1f}/{a['observed_host_steps']:.1f}</td><td>{a['cpu_shortfall']:.1f}/{a['ram_shortfall']:.1f}</td><td>{a['active_host_hours']:.1f}</td><td>{a['moves']:.1f}</td></tr>
<tr><td>RNN + biên</td><td>{cb[0]:.3f}/{cb[1]:.3f}</td><td>{b['overloaded_host_steps']:.1f}/{b['observed_host_steps']:.1f}</td><td>{b['cpu_shortfall']:.1f}/{b['ram_shortfall']:.1f}</td><td>{b['active_host_hours']:.1f}</td><td>{b['moves']:.1f}</td></tr></table>
<p>Trung bình {len(rows)} seed; nguồn {html.escape(str(prepared.get('source')))}.
Biên được hiệu chỉnh trên phần validation riêng với chọn epoch; bao phủ là số
đo thực nghiệm, không phải bảo đảm xác suất. Không thể tính phần trăm cải thiện
so với bài báo vì bài báo không đo các chỉ số này.</p></html>""", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=Path("host_selection/prepared_shared.json"))
    parser.add_argument("--output", type=Path, default=Path("safety_margin/paper_comparison/report.html"))
    parser.add_argument("--q", type=float, default=.95)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    args = parser.parse_args()
    run(json.loads(args.prepared.read_text(encoding="utf-8")), args.output,
        args.q, tuple(args.seeds), args.epochs)
    print(args.output)
