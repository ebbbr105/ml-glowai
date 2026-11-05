"""Predeclared observational feature-ablation study; honest held-out-person protocol.
Run: PYTHONPATH=src python experiments/benchmark.py --data data/raw/mmc3.xlsx --sensor data/raw/mmc2.xlsx
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from glowai.dataset import load_pantavou, add_wearables, subject_splits, BASE, RAD, ACTIVITY
from glowai.estimators import OrdinalLogistic,fit_boosting,multiclass_probs,evaluate,CLASSES

# Frozen list, not selected by outer-test performance.
MODEL_SPECS=[
    ("ordinal_temp", "ordinal", ["air_temperature_c"]),
    ("ordinal_weather", "ordinal", BASE[:3]),
    ("ordinal_base4", "ordinal", BASE),
    ("ordinal_radiation_diagnostic", "ordinal", BASE+RAD),
    ("ordinal_wearables_diagnostic", "ordinal", BASE+ACTIVITY),
    ("xgboost_base4", "xgboost", BASE),
    ("catboost_base4", "catboost", BASE),
    ("catboost_radiation_diagnostic", "catboost", BASE+RAD),
]

def personal_shift(m,own,test,lam=10.0):
    """One regularized personal intercept, refitted between days; lam fixed ex ante.
    No held-out-day or same-day label is used in adaptation.
    """
    probs=[];ordered=[]
    for day in sorted(test.local_date.unique()):
        current=test[test.local_date==day].sort_values(["timestamp_local","source_row_id"])
        history=own[own.local_date<day]
        assert history.timestamp_local.max()<current.timestamp_local.min()
        sy=m.score_raw(history); y=history.tsv.to_numpy(dtype=int)+3
        def f(b):
            p=np.diff(1/(1+np.exp(-np.clip(m.thresholds[None,:]-sy[:,None]-b,-700,700))),axis=1)
            return -np.log(np.maximum(p[np.arange(len(y)),y],1e-14)).sum()+lam*b*b
        res=minimize_scalar(f,method="bounded",bounds=(-6,6),options={"xatol":1e-7})
        if not res.success: raise RuntimeError("adapter optimization")
        probs.append(m.predict_shift(current,res.x));ordered.extend(current.source_row_id.to_list())
    tmp=pd.DataFrame({"source_row_id":ordered,"idx":np.arange(len(ordered))})
    order=tmp.set_index("source_row_id").loc[test.source_row_id,"idx"].to_numpy(dtype=int)
    return np.vstack(probs)[order]


def run(data_path,sensor_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df=load_pantavou(data_path)
    df=add_wearables(df,sensor_path)
    print("SOURCE",len(df),"surveys",df.user_id.nunique(),"participants", flush=True)
    print("METS matched",df.mets_preceding.notna().sum(),"/",len(df),flush=True)
    metrics=[];predictions=[]
    for uid,train,own,test in subject_splits(df):
        print("FOLD",uid,"train",len(train),"test",len(test),flush=True)
        for name,kind,features in MODEL_SPECS:
            if kind=="ordinal":
                m=OrdinalLogistic(features).fit(train)
                p=m.predict_proba(test)
            else:
                m=fit_boosting(train,features,kind)
                p=multiclass_probs(m,test,features)
            models=[(name,p)]
            if name in ("ordinal_base4","ordinal_radiation_diagnostic"):
                models.append((name+"_personal_shift_l10",personal_shift(m,own,test,lam=10.0)))
            if name=="ordinal_base4":
                ytrain=train.tsv.to_numpy(dtype=int)
                hist=np.bincount(ytrain+3,minlength=7)
                prior=(hist+1)/(len(ytrain)+7)
                models.append(("dummy_most_frequent",np.eye(7)[np.full(len(test),np.argmax(hist))]))
                models.append(("dummy_prior",np.tile(prior,(len(test),1))))
                idx=np.searchsorted(np.cumsum(hist),sum(hist)*.5,side="left")
                models.append(("dummy_median",np.eye(7)[np.full(len(test),idx)]))
            for name2,p2 in models:
                s=evaluate(test.tsv,p2)
                metrics.append({"user_id":uid,"model":name2,**s})
                for j,(_,r) in enumerate(test.iterrows()):
                    predictions.append({"user_id":uid,"source_row_id":int(r.source_row_id),"local_date":r.local_date,"model":name2,"y":int(r.tsv),**{f"p_{k}":v for k,v in zip(CLASSES,p2[j])}})
    per=pd.DataFrame(metrics); preds=pd.DataFrame(predictions)
    rows=[]
    for name,g in per.groupby("model"):
        pooled=preds[preds.model==name].sort_values(["user_id","source_row_id"])
        allp=pooled[[f"p_{k}" for k in CLASSES]].to_numpy()
        pp=evaluate(pooled.y.to_numpy(),allp)
        rows.append({"model":name,"people":len(g),"test_rows":len(pooled),
                     **{f"mean_person_{c}":g[c].mean() for c in ["accuracy7","mae_argmax","mae_median","rps","logloss","macro_f1"]},
                     **{f"pooled_{c}":pp[c] for c in ["accuracy7","mae_argmax","mae_median","rps","logloss","macro_f1"]}})
    summary=pd.DataFrame(rows).sort_values("mean_person_mae_median")
    per.to_csv(out/"by_person.csv",index=False)
    summary.to_csv(out/"comparison.csv",index=False)
    preds.to_csv(out/"predictions.csv",index=False)
    df.groupby(["user_id","Season"]).agg(n=("tsv","size"),temp_min=("air_temperature_c","min"),temp_max=("air_temperature_c","max")).to_csv(out/"participant_seasons.csv")
    print("\nRESULTS per-person equal weight:\n", summary[["model","mean_person_accuracy7","mean_person_mae_median","mean_person_rps"]].to_string(index=False,float_format=lambda v:f"{v:.4f}"),flush=True)
    return summary,per,preds

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--sensor",required=True)
    ap.add_argument("--out",default="results")
    args=ap.parse_args()
    run(args.data,args.sensor,args.out)
