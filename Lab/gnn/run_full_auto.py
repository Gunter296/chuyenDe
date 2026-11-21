"""Stream the complete container_usage CSV, then train bounded GNN cohorts.

Every input row is scanned. A row/container contributes to model evaluation
only if it passes the documented quality and simultaneous-window gates.
No 71k-by-71k dense graph is created.
"""
import argparse
import csv
import json
import sqlite3
import sys
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
try:
    from .run_80_20 import fit
except ImportError:
    from run_80_20 import fit


UPSERT = ("INSERT INTO usage VALUES (?,?,?,?) ON CONFLICT(container,bucket) "
          "DO UPDATE SET cpu=max(cpu,excluded.cpu),ram=max(ram,excluded.ram)")


def log(message):
    print(message, file=sys.stderr, flush=True)


def ingest(source, database, bucket_seconds, progress_seconds=10):
    source = source.resolve()
    size = source.stat().st_size
    modified_ns = source.stat().st_mtime_ns
    con = sqlite3.connect(database)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("CREATE TABLE IF NOT EXISTS usage (container TEXT, bucket INTEGER, cpu REAL, ram REAL, PRIMARY KEY(container,bucket))")
    con.execute("CREATE TABLE IF NOT EXISTS ingest_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    state = dict(con.execute("SELECT key,value FROM ingest_state"))
    signature = json.dumps([str(source), size, modified_ns, bucket_seconds])
    if state and state.get("signature") != signature:
        raise ValueError("Source or bucket size changed for this SQLite database; use a new --output directory")
    offset = int(state.get("offset", 0))
    total = int(state.get("total", 0))
    valid = int(state.get("valid", 0))
    if offset >= size:
        log(f"[ingest] already complete: {total:,} rows, {valid:,} valid; building index")
        con.execute("CREATE INDEX IF NOT EXISTS usage_bucket ON usage(bucket)")
        return con, total, valid
    if offset:
        log(f"[ingest] resuming at {offset/size:.1%} ({offset:,}/{size:,} bytes)")
    elif con.execute("SELECT 1 FROM usage LIMIT 1").fetchone():
        log("[ingest] older partial database has no checkpoint; rescanning from byte 0 safely")
    progress = {"stage": "reading CSV", "bytes": offset, "rows": total,
                "valid": valid, "last_commit": offset}
    stop = threading.Event()
    started = time.monotonic()

    def heartbeat():
        while not stop.wait(progress_seconds):
            elapsed = max(time.monotonic() - started, 1)
            done = progress["bytes"] - offset
            speed = done / elapsed
            eta = (size - progress["bytes"]) / speed if speed > 0 else None
            log(f"[ingest] {progress['bytes']/size:.1%} "
                f"({progress['bytes']:,}/{size:,} bytes), "
                f"{progress['rows']:,} rows, {progress['valid']:,} valid; "
                f"{progress['stage']}; {speed/1048576:.2f} MiB/s"
                + (f"; ETA {eta/60:.1f} min" if eta is not None else ""))

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    batch = []
    try:
        with source.open("rb") as stream:
            stream.seek(offset)
            for raw in iter(stream.readline, b""):
                total += 1
                row = next(csv.reader([raw.decode("utf-8", errors="replace")]))
                if len(row) >= 5 and row[0].startswith("c_"):
                    try:
                        stamp, cpu, ram = float(row[2]), float(row[3]), float(row[4])
                    except ValueError:
                        pass
                    else:
                        if stamp >= 0 and 0 <= cpu <= 100 and 0 <= ram <= 100:
                            valid += 1
                            batch.append((row[0], int(stamp // bucket_seconds), cpu, ram))
                progress.update(bytes=stream.tell(), rows=total, valid=valid)
                if len(batch) >= 10000:
                    progress["stage"] = f"writing {len(batch):,} rows to SQLite"
                    con.executemany(UPSERT, batch)
                    con.executemany(
                        "INSERT OR REPLACE INTO ingest_state VALUES (?,?)",
                        [("signature",signature),("offset",str(stream.tell())),
                         ("total",str(total)),("valid",str(valid))])
                    con.commit()
                    batch.clear()
                    progress["last_commit"] = stream.tell()
                    progress["stage"] = "reading CSV"
        progress["stage"] = f"writing final {len(batch):,} rows to SQLite"
        if batch:
            con.executemany(UPSERT, batch)
        con.executemany("INSERT OR REPLACE INTO ingest_state VALUES (?,?)",
                        [("signature",signature),("offset",str(size)),
                         ("total",str(total)),("valid",str(valid))])
        con.commit()
    except KeyboardInterrupt:
        con.rollback()
        log(f"[ingest] interrupted. Safe checkpoint: {progress['last_commit']/size:.1%}. "
            "Run the same command to resume.")
        raise
    finally:
        stop.set()
        worker.join(timeout=1)
    log(f"[ingest] complete: {total:,} rows, {valid:,} valid. Building bucket index...")
    con.execute("CREATE INDEX IF NOT EXISTS usage_bucket ON usage(bucket)")
    log("[ingest] bucket index ready")
    return con, total, valid


def process(source, output, bucket_seconds=1200, window_seconds=86400,
            cohort_size=4, min_coverage=.9, max_cohorts=0, seeds=(11,22,33),
            progress_seconds=10):
    output.mkdir(parents=True, exist_ok=True)
    if str(output.resolve()).startswith("/mnt/hgfs/"):
        log("[warning] SQLite output is on VMware shared folder (/mnt/hgfs), "
            "which can be very slow. Prefer --output ~/gnn_full_run")
    log(f"[start] input={source} ({source.stat().st_size/1048576:.1f} MiB), "
        f"output={output}, progress every {progress_seconds}s")
    con,total,valid=ingest(source,output/"usage.sqlite",bucket_seconds,
                           progress_seconds)
    low, high = con.execute("SELECT min(bucket),max(bucket) FROM usage").fetchone()
    if low is None:
        raise ValueError("No valid container usage rows")
    bins=window_seconds//bucket_seconds
    if bins < 40 or cohort_size < 2:
        raise ValueError("Window needs >=40 bins and cohort size >=2")
    records=[]
    eligible_ids=set()
    used_ids=set()
    for window_number,start in enumerate(range(low,high+1,bins),1):
        log(f"[window {window_number}] buckets [{start},{start+bins}): "
            "finding eligible containers...")
        stop=start+bins
        train_stop=start+int(bins*.8*.75)
        candidates=con.execute(
            "SELECT container,count(*) AS n FROM usage WHERE bucket>=? AND bucket<? "
            "GROUP BY container HAVING n>=? ORDER BY n DESC,container",
            (start,train_stop,int((train_stop-start)*min_coverage+0.999999))).fetchall()
        eligible=[c for c,_ in candidates]
        log(f"[window {window_number}] {len(eligible)} containers meet "
            f"{min_coverage:.0%} train coverage")
        eligible_ids.update(eligible)
        if len(eligible)<cohort_size:
            continue
        rows=con.execute(
            "SELECT container,bucket,cpu,ram FROM usage WHERE bucket>=? AND bucket<? "
            "ORDER BY bucket,container",(start,stop)).fetchall()
        log(f"[window {window_number}] loaded {len(rows):,} aggregated observations")
        by_id={}
        for c,t,cpu,ram in rows:
            if c in eligible_ids:
                by_id.setdefault(c,{})[t]=(cpu,ram)
        remaining=[c for c in eligible if c in by_id]
        while len(remaining)>=cohort_size:
            anchor=remaining.pop(0)
            anchor_bins=set(by_id[anchor])
            ranked=sorted(remaining,key=lambda c:(-len(anchor_bins & set(by_id[c])),c))
            group=[anchor]+ranked[:cohort_size-1]
            for c in group[1:]:
                remaining.remove(c)
            actual=[{c:list(by_id[c][t]) for c in group if t in by_id[c]}
                    for t in range(start,stop)]
            prepared={"actual":actual,"containers":group,
                      "bucket_seconds":bucket_seconds,"source":str(source)}
            try:
                log(f"[cohort] window={window_number}, containers={group}; "
                    f"seeds={list(seeds)}")
                results=[fit(prepared,seed,mode) for seed in seeds
                         for mode in (False,True)]
            except ValueError as error:
                log(f"[cohort] excluded: {error}")
                records.append({"window_start_bucket":start,"containers":group,
                                "status":"excluded","reason":str(error)})
                continue
            used_ids.update(group)
            values={}
            for mode in ("NoGraph","GNN"):
                subset=[r for r in results if r["model"]==mode]
                values[mode]={
                    "mae_cpu_ram":np.mean([r["mae_cpu_ram"] for r in subset],axis=0).tolist(),
                    "rmse_cpu_ram":np.mean([r["rmse_cpu_ram"] for r in subset],axis=0).tolist()}
            records.append({"window_start_bucket":start,"containers":group,
                            "status":"evaluated","sample_counts":results[0]["sample_counts"],
                            "metrics":values})
            log(f"[cohort] evaluated={sum(r['status']=='evaluated' for r in records)}, "
                f"excluded={sum(r['status']=='excluded' for r in records)}, "
                f"remaining_in_window={len(remaining)}")
            if max_cohorts and sum(r["status"]=="evaluated" for r in records)>=max_cohorts:
                break
        if max_cohorts and sum(r["status"]=="evaluated" for r in records)>=max_cohorts:
            break
    evaluated=[r for r in records if r["status"]=="evaluated"]
    summary={}
    for mode in ("NoGraph","GNN"):
        summary[mode]={}
        for metric in ("mae_cpu_ram","rmse_cpu_ram"):
            if evaluated:
                weights=[r["sample_counts"]["test"]*len(r["containers"])
                         for r in evaluated]
                summary[mode][metric]=np.average(
                    [r["metrics"][mode][metric] for r in evaluated],
                    axis=0,weights=weights).tolist()
            else:
                summary[mode][metric]=None
    report={"input":str(source),"input_rows_scanned":total,"valid_rows":valid,
            "distinct_containers":con.execute("SELECT count(DISTINCT container) FROM usage").fetchone()[0],
            "eligible_container_ids":len(eligible_ids),"evaluated_container_ids":len(used_ids),
            "evaluated_cohorts":len(evaluated),"excluded_cohorts":
            sum(r["status"]=="excluded" for r in records),
            "settings":{"bucket_seconds":bucket_seconds,"window_seconds":window_seconds,
                        "cohort_size":cohort_size,"min_train_coverage":min_coverage,
                        "max_cohorts":max_cohorts,"seeds":list(seeds)},
            "summary_weighted_by_test_container_windows":summary,"cohorts":records}
    (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if evaluated:
        image=Image.new("RGB",(900,430),"white")
        draw=ImageDraw.Draw(image)
        draw.text((45,20),"MAE on 0-1 scale, weighted by test container-windows",fill="#222222")
        labels=("CPU","RAM")
        for i,label in enumerate(labels):
            x=100+i*400
            values=[summary[mode]["mae_cpu_ram"][i] for mode in ("NoGraph","GNN")]
            maximum=max(values)*1.2 or 1
            draw.text((x,65),label,fill="#222222")
            for j,(mode,value,color) in enumerate(zip(("NoGraph","GNN"),values,
                                                       ("#4e79a7","#e15759"))):
                y=115+j*115
                draw.rectangle((x,y,x+int(300*value/maximum),y+45),fill=color)
                draw.text((x+305,y+12),f"{mode}: {value:.5f}",fill="#222222")
        image.save(output/"mae_comparison.png")
    con.close()
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path("Lab/data/container_usage.csv"))
    p.add_argument("--output",type=Path,default=Path("Lab/gnn/full_run"))
    p.add_argument("--bucket-seconds",type=int,default=1200)
    p.add_argument("--window-seconds",type=int,default=86400)
    p.add_argument("--cohort-size",type=int,default=4)
    p.add_argument("--min-coverage",type=float,default=.9)
    p.add_argument("--max-cohorts",type=int,default=0,
                   help="0 evaluates every eligible cohort")
    p.add_argument("--progress-seconds",type=int,default=10)
    args=p.parse_args()
    result=process(args.input,args.output,args.bucket_seconds,args.window_seconds,
                   args.cohort_size,args.min_coverage,args.max_cohorts,
                   progress_seconds=args.progress_seconds)
    print(json.dumps({k:result[k] for k in ("input_rows_scanned","valid_rows",
          "distinct_containers","eligible_container_ids","evaluated_container_ids",
          "evaluated_cohorts","excluded_cohorts",
          "summary_weighted_by_test_container_windows")},
          indent=2))
