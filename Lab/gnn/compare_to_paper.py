"""Isolated RNN versus optimized GNN forecast comparison on one catalog.

Published paper values are shown as context, never subtracted from local
scores unless the author's undocumented sampling/evaluation is recovered.
"""
import argparse
import html
import json
import sqlite3
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"host_selection"))
from cohort_baseline import fit_rnn, prepared_from_db, sums_for
from forecast_core import multiresource_graph
from run_80_20 import fit

PAPER = {"mae": .190, "rmse": .250,
         "source": "User-provided article PDF, page 4, Table 1, RNN(Ours) row"}


def fresh():
    return {name: {"abs": 0., "sq": 0., "n": 0, "train_seconds": 0.,
                   "inference_ms": 0., "runs": 0}
            for name in ("RNN", "GNN")}


def add(totals, name, result):
    errors = sums_for(result)
    for key in ("abs", "sq", "n"):
        totals[name][key] += errors[key]
    totals[name]["train_seconds"] += result["training_seconds"]
    totals[name]["inference_ms"] += result["inference_ms_per_step"]
    totals[name]["runs"] += 1


def summarize(totals):
    return {name: {"mae": value["abs"]/value["n"],
                   "rmse": (value["sq"]/value["n"])**.5,
                   "values": value["n"], "training_seconds": value["train_seconds"],
                   "inference_ms_per_step": value["inference_ms"]/value["runs"]}
            for name, value in totals.items() if value["n"] and value["runs"]}


def render(output, summary, meta):
    rnn, gnn = summary["RNN"], summary["GNN"]
    reduction = 100*(rnn["mae"]-gnn["mae"])/rnn["mae"]
    result = {"direction": "gnn", "paper_published": PAPER,
              "local_rnn": rnn, "local_gnn": gnn,
              "rnn_origin": "Lab implementation trained on its own sample; not author code or weights",
              "mae_reduction_vs_local_rnn_percent": reduction, "metadata": meta,
              "comparison_status": "Paired local RNN/GNN comparison; published values are reference only."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    width_r = 540*rnn["mae"]/max(rnn["mae"],gnn["mae"])
    width_g = 540*gnn["mae"]/max(rnn["mae"],gnn["mae"])
    output.write_text(f"""<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Hướng 3: GNN so với RNN đối chứng của Lab</title>
<style>body{{font:16px system-ui;max-width:900px;margin:36px auto;padding:0 20px}}td,th{{border:1px solid #bbb;padding:8px}}table{{border-collapse:collapse}}th{{background:#eef}}svg{{max-width:100%}}</style>
<h1>Hướng 3: thay RNN bằng GNN trong dự báo tải</h1>
<p>{meta['completed']} cohort đã xử lý, bỏ {meta['skipped']};
{rnn['values']} giá trị dự báo trên cùng tập test.</p>
<table><tr><th>Nguồn</th><th>Mô hình</th><th>MAE</th><th>RMSE</th><th>Train tổng (s)</th><th>Suy luận (ms/bước)</th></tr>
<tr><td>Bài báo, Bảng 1</td><td>RNN</td><td>0,190</td><td>0,250</td><td>Không báo</td><td>Không báo</td></tr>
<tr><td>Lab, cùng test</td><td>RNN đối chứng Lab</td><td>{rnn['mae']:.5f}</td><td>{rnn['rmse']:.5f}</td><td>{rnn['training_seconds']:.1f}</td><td>{rnn['inference_ms_per_step']:.3f}</td></tr>
<tr><td>Lab, cùng test</td><td>GNN</td><td>{gnn['mae']:.5f}</td><td>{gnn['rmse']:.5f}</td><td>{gnn['training_seconds']:.1f}</td><td>{gnn['inference_ms_per_step']:.3f}</td></tr></table>
<p>Nguồn hàng bài báo: <a href="../../../Time-SeriesLoadPredictionforCloudResourceAllocationUsingRecurrentNeuralNetworks-1.pdf#page=4">PDF bạn cung cấp, trang 4, Bảng 1, hàng RNN(Ours)</a>. Số được nhập thủ công để tham khảo; Lab không chạy lại mã tác giả.</p>
<svg viewBox="0 0 730 150" aria-label="MAE cùng test"><text x="10" y="21">MAE trên cùng test</text><text x="10" y="64">RNN</text><rect x="85" y="42" width="{width_r:.1f}" height="25" fill="#64748b"/><text x="640" y="61">{rnn['mae']:.5f}</text><text x="10" y="119">GNN</text><rect x="85" y="97" width="{width_g:.1f}" height="25" fill="#16a34a"/><text x="640" y="116">{gnn['mae']:.5f}</text></svg>
<p>GNN giảm MAE {reduction:.1f}% so với RNN đối chứng của Lab trên cùng cohort và bước test.
Không tính phần trăm thay đổi so với 0,190 vì bài báo thiếu chi tiết mẫu, target,
horizon và cách tổng hợp chỉ số. CPU/RAM được chia cho 100; 80% đầu phát triển
mô hình, 20% cuối test; cạnh GNN chọn bằng validation.</p>
<p>Catalog: {html.escape(str(meta['catalog']))}. Đây là phép thử dự báo độc lập;
không dùng kết quả lập lịch hoặc biên dự phòng.</p></html>""", encoding="utf-8")
    return result


def run(catalog_path, output, max_cohorts=0, reuse_checkpoint=None):
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    cohorts = [c for c in catalog["cohorts"] if c["status"] == "evaluated"]
    if max_cohorts:
        cohorts = cohorts[:max_cohorts]
    signature = [str(catalog_path.resolve()), catalog_path.stat().st_size,
                 catalog_path.stat().st_mtime_ns]
    output.parent.mkdir(parents=True, exist_ok=True)
    if reuse_checkpoint:
        saved = json.loads(reuse_checkpoint.read_text(encoding="utf-8"))
        if (saved["completed"] != len(cohorts) or saved["skipped"] or
                saved["signature"][1] != signature[1] or
                abs(saved["signature"][2]-signature[2]) > 1_000_000_000):
            raise ValueError("Existing checkpoint does not cover selected cohorts")
        totals = fresh()
        for name in totals:
            src = saved["totals"][name]
            for src_key, dst_key in (("abs","abs"),("sq","sq"),("n","n"),
                                     ("training_seconds","train_seconds"),
                                     ("inference_ms_sum","inference_ms"),
                                     ("fit_count","runs")):
                totals[name][dst_key] = src[src_key]
        return render(output, summarize(totals),
                      {"completed":len(cohorts),"skipped":0,"catalog":str(catalog_path),
                       "reused_future_work_checkpoint":str(reuse_checkpoint)})
    checkpoint = output.with_suffix(".checkpoint.json")
    totals = fresh(); completed = skipped = 0
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["signature"] != signature or state["completed"] > len(cohorts):
            raise ValueError("Checkpoint incompatible; use a new output path")
        totals, completed, skipped = state["totals"], state["completed"], state["skipped"]
        print(f"[gnn-vs-paper] resume {completed}/{len(cohorts)}", flush=True)
    settings = catalog["settings"]
    bins = settings["window_seconds"]//settings["bucket_seconds"]
    db = catalog_path.parent/"usage.sqlite"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for index, cohort in enumerate(cohorts, 1):
            if index <= completed:
                continue
            prepared = prepared_from_db(con, cohort, bins)
            train_end = int(int(bins*.8)*.75)
            adjacency = multiresource_graph(prepared["actual"],
                                            prepared["containers"], train_end)
            local = fresh()
            try:
                for seed in settings["seeds"]:
                    rnn = fit_rnn(prepared, seed)
                    fixed = fit(prepared, seed, True, return_full=True)
                    adaptive = fit(prepared, seed, True,
                                   adjacency_override=adjacency, return_full=True)
                    chosen = min((fixed, adaptive), key=lambda x:x["validation_mse"])
                    chosen = dict(chosen)
                    chosen["training_seconds"] = fixed["training_seconds"]+adaptive["training_seconds"]
                    if rnn["test_times"] != chosen["test_times"]:
                        raise ValueError("RNN/GNN test times differ")
                    add(local,"RNN",rnn); add(local,"GNN",chosen)
                for name in totals:
                    for key in totals[name]:
                        totals[name][key] += local[name][key]
            except ValueError:
                skipped += 1
            state = {"signature":signature,"completed":index,"skipped":skipped,"totals":totals}
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(state), encoding="utf-8")
            temporary.replace(checkpoint)
            if index%10 == 0 or index == len(cohorts):
                print(f"[gnn-vs-paper] {index}/{len(cohorts)}, skipped={skipped}", flush=True)
    finally:
        con.close()
    if not totals["RNN"]["n"] or not totals["GNN"]["n"]:
        raise ValueError("No eligible test values")
    return render(output,summarize(totals),
                  {"completed":len(cohorts),"skipped":skipped,"catalog":str(catalog_path)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("gnn/full_run/gnn_full_run/report.json"))
    parser.add_argument("--output", type=Path, default=Path("gnn/paper_comparison/report.html"))
    parser.add_argument("--max-cohorts", type=int, default=0)
    parser.add_argument("--reuse-checkpoint", type=Path)
    args = parser.parse_args()
    run(args.catalog,args.output,args.max_cohorts,args.reuse_checkpoint)
    print(args.output)
