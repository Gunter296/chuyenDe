"""Full-catalog host-selection experiment, separate from GNN/margin changes."""
import argparse
import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
from cohort_baseline import fit_rnn, prepared_from_db, schedule_metrics, sums_for

FIELDS = ("overloaded_host_steps", "observed_host_steps", "cpu_shortfall",
          "ram_shortfall", "active_host_hours", "moves")


def static_metrics(result, containers, bucket_seconds):
    times = result["test_times"]
    runs = []
    start = 0
    for i in range(1, len(times)+1):
        if i == len(times) or times[i] != times[i-1]+1:
            runs.append((start,i)); start=i
    lo, hi = max(runs, key=lambda pair:pair[1]-pair[0])
    if hi-lo < 2:
        return None
    metrics = {field:0 for field in FIELDS}
    for step in result["truth_full"][lo+1:hi]:
        hosts = [[0.,0.],[0.,0.]]
        for i, usage in enumerate(step):
            for resource in (0,1):
                hosts[i%2][resource] += usage[resource]*100
        for usage in hosts:
            metrics["observed_host_steps"] += 1
            metrics["overloaded_host_steps"] += int(any(v>100 for v in usage))
            metrics["cpu_shortfall"] += max(0.,usage[0]-100)
            metrics["ram_shortfall"] += max(0.,usage[1]-100)
            metrics["active_host_hours"] += bucket_seconds/3600
    return metrics


def fresh():
    return {"abs":0.,"sq":0.,"n":0,"runs":0,"schedule_runs":0,
            "static":{field:0 for field in FIELDS},
            "scheduler":{field:0 for field in FIELDS}}


def render(output, totals, meta):
    mae=totals["abs"]/totals["n"]
    rmse=(totals["sq"]/totals["n"])**.5
    a,b=totals["static"],totals["scheduler"]
    result={"direction":"host_selection","paper_published":{"mae":.190,"rmse":.250,
            "source":"User-provided article PDF, page 4, Table 1, RNN(Ours) row"},
            "local_rnn":{"mae":mae,"rmse":rmse,"values":totals["n"]},
            "rnn_origin":"Lab implementation trained on its own sample; not author code or weights",
            "static_placement":a,"rnn_host_selection":b,"metadata":meta,
            "comparison_status":"Paper has no scheduling numbers; static placement is a new control."}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    output.write_text(f"""<!doctype html><html lang="vi"><meta charset="utf-8"><title>Chọn máy: toàn bộ cohort</title>
<style>body{{font:16px system-ui;max-width:900px;margin:36px auto;padding:0 20px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:8px}}th{{background:#eef}}</style>
<h1>Hướng 1: RNN dự báo và chọn máy cụ thể</h1>
<p>{meta['completed']} cohort xét, {meta['skipped']} bỏ; {totals['schedule_runs']} lượt cohort–seed có ít nhất hai bước test liên tiếp.</p>
<p>Bài gốc công bố RNN MAE 0,190, RMSE 0,250. RNN đối chứng do Lab tự cài đặt: MAE {mae:.5f}, RMSE {rmse:.5f} trên {totals['n']} giá trị test. Không dùng mã, trọng số hoặc tập test gốc của tác giả; không tính chênh lệch trực tiếp với số công bố.</p>
<p>Nguồn số công bố: <a href="../../../Time-SeriesLoadPredictionforCloudResourceAllocationUsingRecurrentNeuralNetworks-1.pdf#page=4">PDF bạn cung cấp, trang 4, Bảng 1, hàng RNN(Ours)</a>; được nhập thủ công làm tham khảo, không phải kết quả chạy lại mã tác giả.</p>
<table><tr><th>Phương án Lab</th><th>Quá tải / máy-bước</th><th>Thiếu CPU/RAM</th><th>Giờ máy</th><th>Di chuyển</th></tr>
<tr><td>Giữ vị trí cố định</td><td>{a['overloaded_host_steps']}/{a['observed_host_steps']}</td><td>{a['cpu_shortfall']:.1f}/{a['ram_shortfall']:.1f}</td><td>{a['active_host_hours']:.1f}</td><td>0</td></tr>
<tr><td>RNN + chọn máy</td><td>{b['overloaded_host_steps']}/{b['observed_host_steps']}</td><td>{b['cpu_shortfall']:.1f}/{b['ram_shortfall']:.1f}</td><td>{b['active_host_hours']:.1f}</td><td>{b['moves']}</td></tr></table>
<p>Hai máy ban đầu, mỗi máy dung lượng (100,100). Chỉ số là tổng mô phỏng qua cohort và seed, không phải một cụm sản xuất. Bài gốc không có đối chứng lập lịch định lượng; giữ vị trí cố định là đối chứng mới của Lab.</p></html>""",encoding="utf-8")


def run(catalog_path, output, max_cohorts=0):
    catalog=json.loads(catalog_path.read_text(encoding="utf-8"))
    cohorts=[c for c in catalog["cohorts"] if c["status"]=="evaluated"]
    if max_cohorts:cohorts=cohorts[:max_cohorts]
    settings=catalog["settings"]
    bins=settings["window_seconds"]//settings["bucket_seconds"]
    signature=[catalog_path.stat().st_size,catalog_path.stat().st_mtime_ns]
    output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint=output.with_suffix(".checkpoint.json")
    totals=fresh();completed=skipped=0
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text(encoding="utf-8"))
        if (state["signature"][0]!=signature[0] or
                abs(state["signature"][1]-signature[1])>1_000_000_000 or
                state["completed"]>len(cohorts)):
            raise ValueError("Checkpoint incompatible; choose another output")
        totals,completed,skipped=state["totals"],state["completed"],state["skipped"]
        print(f"[host-selection] resume {completed}/{len(cohorts)}",flush=True)
    db=catalog_path.parent/"usage.sqlite"
    con=sqlite3.connect(f"file:{db}?mode=ro",uri=True)
    try:
        for index,cohort in enumerate(cohorts,1):
            if index<=completed:continue
            prepared=prepared_from_db(con,cohort,bins)
            local=fresh()
            try:
                for seed in settings["seeds"]:
                    rnn=fit_rnn(prepared,seed)
                    errors=sums_for(rnn)
                    for field in ("abs","sq","n"):local[field]+=errors[field]
                    local["runs"]+=1
                    static=static_metrics(rnn,prepared["containers"],settings["bucket_seconds"])
                    dynamic=schedule_metrics(rnn,prepared["containers"],settings["bucket_seconds"])
                    if (static is None)!=(dynamic is None):
                        raise ValueError("Static/scheduler test mismatch")
                    if static:
                        local["schedule_runs"]+=1
                        for label,metrics in (("static",static),("scheduler",dynamic)):
                            for field in FIELDS:local[label][field]+=metrics[field]
                for field in ("abs","sq","n","runs","schedule_runs"):
                    totals[field]+=local[field]
                for label in ("static","scheduler"):
                    for field in FIELDS:totals[label][field]+=local[label][field]
            except ValueError:
                skipped+=1
            state={"signature":signature,"completed":index,"skipped":skipped,"totals":totals}
            temporary=checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(state),encoding="utf-8")
            temporary.replace(checkpoint)
            if index%10==0 or index==len(cohorts):
                print(f"[host-selection] {index}/{len(cohorts)}, skipped={skipped}",flush=True)
    finally:con.close()
    if not totals["n"]:raise ValueError("No test values")
    render(output,totals,{"completed":len(cohorts),"skipped":skipped,"catalog":str(catalog_path)})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--catalog",type=Path,default=Path("gnn/full_run/gnn_full_run/report.json"))
    parser.add_argument("--output",type=Path,default=Path("host_selection/paper_comparison/full.html"))
    parser.add_argument("--max-cohorts",type=int,default=0)
    args=parser.parse_args()
    run(args.catalog,args.output,args.max_cohorts)
    print(args.output)
