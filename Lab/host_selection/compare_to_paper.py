"""Evaluate the host-choice extension with a reconstructed RNN baseline.

The paper publishes forecast MAE/RMSE but no scheduling numbers. The static
placement control here is an explicit new control, not the paper's scheduler.
"""
import argparse
import html
import json
from pathlib import Path
from statistics import mean

import numpy as np

from experiment import experiment

PAPER = {"mae": 0.190, "rmse": 0.250,
         "source": "User-provided article PDF, page 4, Table 1, RNN(Ours) row"}


def forecast_errors(rows):
    all_errors = []
    for row in rows:
        data = json.loads(Path(row["comparison_file"]).read_text(encoding="utf-8"))
        for actual, forecast in zip(data["actual"], data["forecast"]):
            for container in actual:
                all_errors.extend((np.asarray(forecast[container][0]) -
                                   np.asarray(actual[container])).tolist())
    values = np.asarray(all_errors, dtype=float) / 100
    return float(np.abs(values).mean()), float(np.sqrt(np.mean(values**2)))


def static_placement(data):
    capacity = data["capacity"]
    placement = data["placement"]
    overloaded = observed = 0
    short_cpu = short_ram = 0.
    for actual in data["actual"][1:]:
        loads = {host: [0., 0.] for host in capacity}
        for container, host in placement.items():
            loads[host][0] += actual[container][0]
            loads[host][1] += actual[container][1]
        for host, load in loads.items():
            observed += 1
            overloaded += int(any(load[r] > capacity[host][r] for r in (0, 1)))
            short_cpu += max(0., load[0]-capacity[host][0])
            short_ram += max(0., load[1]-capacity[host][1])
    return {"overloaded_host_steps": overloaded, "observed_host_steps": observed,
            "cpu_shortfall": short_cpu, "ram_shortfall": short_ram,
            "active_host_hours": observed * data["bucket_seconds"] / 3600,
            "moves": 0}


def run(prepared, output, seeds=(11, 22, 33), epochs=30):
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / (output.stem + "_inputs")
    old = experiment(prepared, work, seeds=seeds, epochs=epochs)
    rows = []
    for row in old["per_seed"]:
        path = work / f"comparison_seed_{row['seed']}.json"
        comparison = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"seed": row["seed"], "comparison_file": str(path),
                     "static": static_placement(comparison),
                     "rnn_scheduler": row["B1"]})
    mae, rmse = forecast_errors(rows)
    def metrics(key):
        values = [r[key] for r in rows]
        return {field: mean(v[field] for v in values)
                for field in ("overloaded_host_steps", "observed_host_steps",
                              "cpu_shortfall", "ram_shortfall", "active_host_hours", "moves")}
    result = {"direction": "host_selection", "paper_published": PAPER,
              "local_rnn": {"mae": mae, "rmse": rmse},
              "rnn_origin": "Lab implementation trained on its own sample; not author code or weights",
              "static_placement": metrics("static"),
              "rnn_host_selection": metrics("rnn_scheduler"),
              "seeds": list(seeds), "source": prepared.get("source"),
              "comparison_status": "Paper has no published scheduling metrics; static placement is a new control."}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    a, b = result["static_placement"], result["rnn_host_selection"]
    output.write_text(f"""<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Hướng 1: chọn máy chủ</title><style>body{{font:16px system-ui;max-width:940px;margin:36px auto;padding:0 20px}}td,th{{border:1px solid #bbb;padding:8px}}table{{border-collapse:collapse}}th{{background:#eef}}</style>
<h1>Hướng 1: từ dự báo RNN đến chọn máy chủ</h1>
<p>Bài báo công bố RNN MAE 0,190; RMSE 0,250, nhưng không có số liệu lập lịch.
RNN đối chứng của Lab trên tập này: MAE {mae:.5f}, RMSE {rmse:.5f}. Hai cặp số không
thể trừ trực tiếp vì giao thức gốc chưa được mô tả đủ.</p>
<p>Nguồn số công bố: <a href="../../../Time-SeriesLoadPredictionforCloudResourceAllocationUsingRecurrentNeuralNetworks-1.pdf#page=4">PDF bạn cung cấp, trang 4, Bảng 1, hàng RNN(Ours)</a>; được nhập thủ công làm tham khảo, không phải kết quả chạy lại mã tác giả.</p>
<p>RNN của Lab do chúng tôi tự cài đặt và huấn luyện; không dùng mã, trọng số hoặc tập test gốc của tác giả.</p>
<table><tr><th>Phương án</th><th>Quá tải / máy-bước</th><th>Thiếu CPU</th><th>Thiếu RAM</th><th>Giờ máy</th><th>Di chuyển</th></tr>
<tr><td>Giữ vị trí cố định (đối chứng mới)</td><td>{a['overloaded_host_steps']:.1f}/{a['observed_host_steps']:.1f}</td><td>{a['cpu_shortfall']:.1f}</td><td>{a['ram_shortfall']:.1f}</td><td>{a['active_host_hours']:.1f}</td><td>0</td></tr>
<tr><td>RNN + chọn máy</td><td>{b['overloaded_host_steps']:.1f}/{b['observed_host_steps']:.1f}</td><td>{b['cpu_shortfall']:.1f}</td><td>{b['ram_shortfall']:.1f}</td><td>{b['active_host_hours']:.1f}</td><td>{b['moves']:.1f}</td></tr></table>
<p>Trung bình {len(rows)} seed; nguồn {html.escape(str(prepared.get('source')))}.
Đây là mô phỏng hai tài nguyên trên tập đã chuẩn bị, không phải kết quả vận hành
hay mức cải thiện định lượng so với bộ lập lịch của bài báo gốc.</p></html>""", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=Path("host_selection/prepared_shared.json"))
    parser.add_argument("--output", type=Path, default=Path("host_selection/paper_comparison/report.html"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    args = parser.parse_args()
    run(json.loads(args.prepared.read_text(encoding="utf-8")), args.output,
        tuple(args.seeds), args.epochs)
    print(args.output)
