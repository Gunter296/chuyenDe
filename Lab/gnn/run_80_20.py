"""GNN vs no-graph on the shared real subset, 80/20 chronological split.

CPU/RAM utilization percentages are divided by 100. Validation is the last
quarter of the 80% development period; test is untouched until evaluation.
"""
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    from .graph_model import GraphNet, graph, examples
except ImportError:
    from graph_model import GraphNet, graph, examples

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "host_selection" / "prepared_shared.json"
OUT = ROOT / "gnn" / "results_80_20"


def fit(prepared, seed, use_graph, history=6, horizon=1, epochs=100,
        adjacency_override=None, return_full=False):
    actual, containers = prepared["actual"], prepared["containers"]
    n = len(actual)
    dev_end = int(n * .8)
    train_end = int(dev_end * .75)
    a, _ = graph(actual, containers, train_end, neighbors=2)
    if adjacency_override is not None:
        a = np.asarray(adjacency_override, dtype=float)
        if a.shape != (len(containers), len(containers)):
            raise ValueError("Adjacency shape does not match containers")
    if not use_graph:
        a = np.zeros_like(a)
    scale = np.array([100., 100.])
    train = list(examples(actual, containers, 0, train_end, history, horizon, scale))
    val = list(examples(actual, containers, train_end, dev_end, history, horizon, scale))
    test = list(examples(actual, containers, dev_end, n, history, horizon, scale))
    if not train or not val or not test:
        raise ValueError(f"Incomplete windows: train={len(train)}, val={len(val)}, test={len(test)}")
    model = GraphNet(history*2, horizon*2, 16, seed)
    rng = np.random.default_rng(seed)
    best, stale, params = float("inf"), 0, None
    curve = []
    training_started = time.perf_counter()
    for epoch in range(epochs):
        for index in rng.permutation(len(train)):
            _, x, y = train[index]
            model.step(x, y, a, .01)
        train_loss = float(np.mean([np.mean((model.predict(x,a)[0]-y)**2)
                                    for _,x,y in train]))
        val_loss = float(np.mean([np.mean((model.predict(x,a)[0]-y)**2)
                                  for _,x,y in val]))
        curve.append([epoch+1, train_loss, val_loss])
        if val_loss < best:
            best, stale = val_loss, 0
            params = [p.copy() for p in
                      (model.ws,model.wn,model.b,model.v,model.c)]
        else:
            stale += 1
            if stale >= 10:
                break
    model.ws,model.wn,model.b,model.v,model.c = params
    training_seconds = time.perf_counter() - training_started
    predicted, truth, times = [], [], []
    inference_started = time.perf_counter()
    for t,x,y in test:
        predicted.append(np.maximum(model.predict(x,a)[0].reshape(len(containers),horizon,2),0))
        truth.append(y.reshape(len(containers),horizon,2))
        times.append(t)
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    inference_ms_per_step = (time.perf_counter()-inference_started)*1000/len(test)
    error = predicted-truth
    result={"seed":seed,"model":"GNN" if use_graph else "NoGraph",
            "split":{"train_end":train_end,"development_end":dev_end,"total":n},
            "sample_counts":{"train":len(train),"validation":len(val),"test":len(test)},
            "edge_count":int(np.count_nonzero(a)),
            "mae_cpu_ram":np.mean(np.abs(error),axis=(0,1,2)).tolist(),
            "rmse_cpu_ram":np.sqrt(np.mean(error**2,axis=(0,1,2))).tolist(),
            "loss_curve":curve,"test_times":times,
            "test_truth":truth[:, :, 0, :].mean(axis=1).tolist(),
            "test_prediction":predicted[:, :, 0, :].mean(axis=1).tolist(),
            "validation_mse":best}
    result["training_seconds"] = training_seconds
    result["inference_ms_per_step"] = inference_ms_per_step
    if return_full:
        result["forecast_full"] = predicted[:, :, 0, :].tolist()
        result["truth_full"] = truth[:, :, 0, :].tolist()
        result["validation_errors"] = [
            (y.reshape(len(containers),horizon,2)[:,0,:] -
             np.maximum(model.predict(x,a)[0].reshape(len(containers),horizon,2)[:,0,:],0)).tolist()
            for _,x,y in val]
    return result


def line_chart(series, labels, path, title, y_label):
    w,h=1000,520
    im=Image.new("RGB",(w,h),"white")
    d=ImageDraw.Draw(im)
    left,right,top,bottom=85,w-35,65,h-65
    vals=[v for s in series for v in s]
    lo=min(vals); hi=max(vals)
    if hi==lo: hi=lo+1
    pad=(hi-lo)*.08;lo-=pad;hi+=pad
    d.line((left,top,left,bottom),fill="#333333",width=2)
    d.line((left,bottom,right,bottom),fill="#333333",width=2)
    d.text((left,20),title,fill="#111111")
    d.text((10,top),y_label,fill="#111111")
    colors=["#1f77b4","#d62728","#2ca02c","#9467bd"]
    for i,s in enumerate(series):
        if len(s)<2:continue
        pts=[(left+j*(right-left)/(len(s)-1),
              bottom-(v-lo)*(bottom-top)/(hi-lo)) for j,v in enumerate(s)]
        d.line(pts,fill=colors[i],width=3)
        d.text((left+i*220,bottom+18),labels[i],fill=colors[i])
    for i in range(5):
        y=bottom-i*(bottom-top)/4
        d.line((left,y,right,y),fill="#dddddd")
        d.text((8,y-7),f"{lo+i*(hi-lo)/4:.3f}",fill="#444444")
    im.save(path)


def main():
    prepared=json.loads(SOURCE.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True,exist_ok=True)
    results=[fit(prepared,seed,mode) for seed in (11,22,33)
             for mode in (False,True)]
    summary={}
    for mode in ("NoGraph","GNN"):
        rows=[r for r in results if r["model"]==mode]
        summary[mode]={
            "mae_cpu_ram":np.mean([r["mae_cpu_ram"] for r in rows],axis=0).tolist(),
            "rmse_cpu_ram":np.mean([r["rmse_cpu_ram"] for r in rows],axis=0).tolist()}
    report={"source":str(SOURCE),"normalization":"cpu_util_percent/100, mem_util_percent/100",
            "bucket_seconds":prepared["bucket_seconds"],
            "containers":prepared["containers"],"results":results,"summary":summary}
    (OUT/"results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    g=next(r for r in results if r["model"]=="GNN" and r["seed"]==11)
    for i,name in enumerate(("CPU","RAM")):
        line_chart([[row[i] for row in g["test_truth"]],
                    [row[i] for row in g["test_prediction"]]],
                   [f"{name} actual",f"{name} forecast"],
                   OUT/f"forecast_{name.lower()}_seed11.png",
                   f"GNN {name} forecast vs actual, mean of 4 containers",
                   "utilization / 100")
    line_chart([[row[i] for row in g["loss_curve"]] for i in (1,2)],
               ["train MSE","validation MSE"],OUT/"learning_curve_seed11.png",
               "GNN learning curve, seed 11","MSE on 0-1 scale")
    for i,name in enumerate(("CPU","RAM")):
        series=[[row[c][i]/100 for row in prepared["actual"] if c in row]
                for c in prepared["containers"]]
        line_chart(series,prepared["containers"],OUT/f"input_{name.lower()}.png",
                   f"Input {name} by container (observed bins only)","utilization / 100")
    print(json.dumps({"sample_counts":g["sample_counts"],"summary":summary},indent=2))


if __name__=="__main__":
    main()
