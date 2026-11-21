"""Full-catalog RNN residual-margin experiment, no GNN or host ablation."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"host_selection"))
from cohort_baseline import fit_rnn, prepared_from_db, schedule_metrics

FIELDS=("overloaded_host_steps","observed_host_steps","cpu_shortfall",
        "ram_shortfall","active_host_hours","moves")


def margin_result(rnn,q):
    residuals=np.asarray(rnn["validation_errors"],dtype=float)
    if residuals.ndim!=3 or residuals.shape[-1]!=2 or not len(residuals):
        raise ValueError("Missing validation residuals")
    margin=np.maximum(np.quantile(residuals.reshape(-1,2),q,axis=0),0)
    adjusted=dict(rnn)
    adjusted["forecast_full"]=(np.asarray(rnn["forecast_full"])+margin).tolist()
    return adjusted


def fresh():
    return {name:{"covered":0,"coverage_n":0,"schedule_runs":0,
                  **{field:0 for field in FIELDS}}
            for name in ("RNN","RNN_margin")}


def render(output,totals,meta):
    a,b=totals["RNN"],totals["RNN_margin"]
    result={"direction":"safety_margin","paper_published":{"mae":.190,"rmse":.250,
            "source":"User-provided article PDF, page 4, Table 1, RNN(Ours) row"},
            "rnn_origin":"Lab implementation trained on its own sample; not author code or weights",
            "rnn_without_margin":a,"rnn_with_margin":b,
            "coverage_without_margin":a["covered"]/a["coverage_n"],
            "coverage_with_margin":b["covered"]/b["coverage_n"],
            "q":meta["q"],"metadata":meta,
            "comparison_status":"Paper has no margin, coverage or scheduling metrics."}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    output.write_text(f"""<!doctype html><html lang="vi"><meta charset="utf-8"><title>Biên dự phòng: toàn bộ cohort</title>
<style>body{{font:16px system-ui;max-width:900px;margin:36px auto;padding:0 20px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:8px}}th{{background:#eef}}</style>
<h1>Hướng 2: biên dự phòng theo sai số RNN</h1>
<p>{meta['completed']} cohort xét, {meta['skipped']} bỏ. q={meta['q']:.2f}; chỉ thay đổi là cộng biên vào dự báo RNN trước khi chọn máy.</p>
<p>Bài gốc công bố RNN MAE 0,190 và RMSE 0,250, nhưng không có số liệu biên, độ bao phủ hay lập lịch. Không thể tính mức cải thiện trực tiếp từ Bảng 1.</p>
<p>Nguồn 0,190/0,250: <a href="../../../Time-SeriesLoadPredictionforCloudResourceAllocationUsingRecurrentNeuralNetworks-1.pdf#page=4">PDF bạn cung cấp, trang 4, Bảng 1, hàng RNN(Ours)</a>; được nhập thủ công làm tham khảo, không phải kết quả chạy lại mã tác giả.</p>
<p>RNN đối chứng do Lab tự cài đặt và huấn luyện; không dùng mã, trọng số hoặc tập test gốc của tác giả.</p>
<table><tr><th>Phương án Lab</th><th>Độ bao phủ</th><th>Quá tải / máy-bước</th><th>Thiếu CPU/RAM</th><th>Giờ máy</th><th>Di chuyển</th></tr>
<tr><td>RNN</td><td>{result['coverage_without_margin']:.3f}</td><td>{a['overloaded_host_steps']}/{a['observed_host_steps']}</td><td>{a['cpu_shortfall']:.1f}/{a['ram_shortfall']:.1f}</td><td>{a['active_host_hours']:.1f}</td><td>{a['moves']}</td></tr>
<tr><td>RNN + biên</td><td>{result['coverage_with_margin']:.3f}</td><td>{b['overloaded_host_steps']}/{b['observed_host_steps']}</td><td>{b['cpu_shortfall']:.1f}/{b['ram_shortfall']:.1f}</td><td>{b['active_host_hours']:.1f}</td><td>{b['moves']}</td></tr></table>
<p>{a['schedule_runs']} lượt cohort–seed đủ hai bước test liên tiếp. Độ bao phủ gộp CPU/RAM trên mọi test; các chỉ số máy là tổng mô phỏng qua cohort và seed, không phải một cụm thật. Biên dùng validation đã chọn epoch nên không có bảo đảm xác suất; không dùng nhãn test để tính biên.</p></html>""",encoding="utf-8")


def run(catalog_path,output,q=.95,max_cohorts=0):
    if not 0<q<1:raise ValueError("q must be between 0 and 1")
    catalog=json.loads(catalog_path.read_text(encoding="utf-8"))
    cohorts=[c for c in catalog["cohorts"] if c["status"]=="evaluated"]
    if max_cohorts:cohorts=cohorts[:max_cohorts]
    settings=catalog["settings"]
    bins=settings["window_seconds"]//settings["bucket_seconds"]
    signature=[catalog_path.stat().st_size,catalog_path.stat().st_mtime_ns,q]
    output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint=output.with_suffix(".checkpoint.json")
    totals=fresh();completed=skipped=0
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text(encoding="utf-8"))
        old=state["signature"]
        if (old[0]!=signature[0] or abs(old[1]-signature[1])>1_000_000_000 or
                old[2]!=q or state["completed"]>len(cohorts)):
            raise ValueError("Checkpoint incompatible; choose another output")
        totals,completed,skipped=state["totals"],state["completed"],state["skipped"]
        print(f"[margin] resume {completed}/{len(cohorts)}",flush=True)
    con=sqlite3.connect(f"file:{catalog_path.parent/'usage.sqlite'}?mode=ro",uri=True)
    try:
        for index,cohort in enumerate(cohorts,1):
            if index<=completed:continue
            prepared=prepared_from_db(con,cohort,bins)
            local=fresh()
            try:
                for seed in settings["seeds"]:
                    rnn=fit_rnn(prepared,seed)
                    adjusted=margin_result(rnn,q)
                    for name,result in (("RNN",rnn),("RNN_margin",adjusted)):
                        truth=np.asarray(result["truth_full"])
                        forecast=np.asarray(result["forecast_full"])
                        local[name]["covered"]+=int(np.count_nonzero(truth<=forecast))
                        local[name]["coverage_n"]+=int(truth.size)
                        metrics=schedule_metrics(result,prepared["containers"],settings["bucket_seconds"])
                        if metrics:
                            local[name]["schedule_runs"]+=1
                            for field in FIELDS:local[name][field]+=metrics[field]
                for name in totals:
                    for field in totals[name]:totals[name][field]+=local[name][field]
            except ValueError:
                skipped+=1
            state={"signature":signature,"completed":index,"skipped":skipped,"totals":totals}
            temporary=checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(state),encoding="utf-8")
            temporary.replace(checkpoint)
            if index%10==0 or index==len(cohorts):
                print(f"[margin] {index}/{len(cohorts)}, skipped={skipped}",flush=True)
    finally:con.close()
    if not totals["RNN"]["coverage_n"]:raise ValueError("No test values")
    render(output,totals,{"completed":len(cohorts),"skipped":skipped,
                          "catalog":str(catalog_path),"q":q})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--catalog",type=Path,default=Path("gnn/full_run/gnn_full_run/report.json"))
    parser.add_argument("--output",type=Path,default=Path("safety_margin/paper_comparison/full.html"))
    parser.add_argument("--q",type=float,default=.95)
    parser.add_argument("--max-cohorts",type=int,default=0)
    args=parser.parse_args()
    run(args.catalog,args.output,args.q,args.max_cohorts)
    print(args.output)
