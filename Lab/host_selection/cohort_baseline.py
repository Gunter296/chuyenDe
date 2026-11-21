"""Shared RNN baseline and cohort scheduling helpers for full-catalog tests."""
import sqlite3
import time

import numpy as np

from train_rnn import RNN
from run import run as schedule_run


def examples(actual, containers, lo, hi, history, horizon, scale):
    for t in range(max(lo, history), hi-horizon+1):
        if any(any(c not in actual[k] for c in containers)
               for k in range(t-history, t+horizon)):
            continue
        x=np.asarray([[actual[k][c] for k in range(t-history,t)]
                      for c in containers])/scale
        y=np.asarray([[actual[k][c] for k in range(t,t+horizon)]
                      for c in containers])/scale
        yield t,x.reshape(len(containers),-1),y.reshape(len(containers),-1)


def fit_rnn(prepared,seed,history=6,horizon=1,epochs=100):
    actual,containers=prepared["actual"],prepared["containers"]
    n=len(actual);dev_end=int(n*.8);train_end=int(dev_end*.75)
    train=list(examples(actual,containers,0,train_end,history,horizon,
                        np.array([100.,100.])))
    val=list(examples(actual,containers,train_end,dev_end,history,horizon,
                      np.array([100.,100.])))
    test=list(examples(actual,containers,dev_end,n,history,horizon,
                       np.array([100.,100.])))
    if not train or not val or not test:
        raise ValueError("RNN missing aligned windows")
    model=RNN(16,horizon,seed);rng=np.random.default_rng(seed)
    best=float("inf");params=None;stale=0
    started=time.perf_counter()
    for _ in range(epochs):
        order=[(i,c) for i in range(len(train)) for c in range(len(containers))]
        for index in rng.permutation(len(order)):
            i,c=order[index];_,x,y=train[i]
            model.step(x[c].reshape(history,2),y[c].reshape(horizon,2),.001)
        loss=np.mean([np.mean((model.predict(x[c].reshape(history,2))[0]-
                              y[c])**2)
                      for _,x,y in val for c in range(len(containers))])
        if loss<best:
            best=float(loss);stale=0
            params=[p.copy() for p in (model.w,model.u,model.b,model.v,model.c)]
        else:
            stale+=1
            if stale>=10:break
    model.w,model.u,model.b,model.v,model.c=params
    training_seconds=time.perf_counter()-started
    predictions=[];truth=[];times=[]
    inference_started=time.perf_counter()
    for t,x,y in test:
        predictions.append([np.maximum(model.predict(x[c].reshape(history,2))[0]
                         .reshape(horizon,2),0)[0].tolist()
                            for c in range(len(containers))])
        truth.append(y.reshape(len(containers),horizon,2)[:,0,:].tolist())
        times.append(t)
    return {"model":"RNN","seed":seed,"test_times":times,
            "forecast_full":predictions,"truth_full":truth,
            "validation_errors":[
                (y.reshape(len(containers),horizon,2)[:,0,:] -
                 np.asarray([np.maximum(model.predict(x[c].reshape(history,2))[0]
                              .reshape(horizon,2),0)[0]
                             for c in range(len(containers))])).tolist()
                for _,x,y in val],
            "validation_mse":best,"training_seconds":training_seconds,
            "inference_ms_per_step":
            (time.perf_counter()-inference_started)*1000/max(1,len(test))}


def prepared_from_db(con,cohort,bins):
    start=cohort["window_start_bucket"]
    containers=cohort["containers"]
    placeholders=",".join("?" for _ in containers)
    rows=con.execute(
        f"SELECT container,bucket,cpu,ram FROM usage WHERE bucket>=? AND bucket<? "
        f"AND container IN ({placeholders})",(start,start+bins,*containers)).fetchall()
    actual=[{} for _ in range(bins)]
    for c,t,cpu,ram in rows:
        actual[t-start][c]=[cpu,ram]
    return {"actual":actual,"containers":containers}


def sums_for(result):
    pred=np.asarray(result["forecast_full"],dtype=float)
    truth=np.asarray(result["truth_full"],dtype=float)
    error=pred-truth
    return {"abs":float(np.abs(error).sum()),"sq":float((error**2).sum()),
            "n":int(error.size),"cpu_abs":float(np.abs(error[...,0]).sum()),
            "ram_abs":float(np.abs(error[...,1]).sum()),
            "cpu_n":int(error[...,0].size)}


def schedule_metrics(result,containers,bucket_seconds):
    times=result["test_times"]
    runs=[];start=0
    for i in range(1,len(times)+1):
        if i==len(times) or times[i]!=times[i-1]+1:
            runs.append((start,i));start=i
    lo,hi=max(runs,key=lambda pair:pair[1]-pair[0])
    if hi-lo<2:return None
    truth=result["truth_full"][lo:hi]
    forecast=result["forecast_full"][lo:hi]
    capacity={"h0":[100,100],"h1":[100,100]}
    placement={c:f"h{i%2}" for i,c in enumerate(containers)}
    data={"actual":[{c:[v*100 for v in row[i]] for i,c in enumerate(containers)}
                    for row in truth],
          "forecast":[{c:[[v*100 for v in row[i]]] for i,c in enumerate(containers)}
                      for row in forecast],
          "capacity":capacity,"placement":placement,
          "bucket_seconds":bucket_seconds,"boot_minutes":1}
    return schedule_run(data,"B1")
