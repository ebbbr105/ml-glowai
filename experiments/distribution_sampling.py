"""Prequential personal adaptation: representativeness of observed *past* weather.

No target labels or future observations used to select calibration examples.
Leave-one-person-out trains population model; test one person on later days.
Compares 10/20 most recent vs weather-distribution quantiles vs distinct-day
coverage on identical test observations; all history reference.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from glowai.dataset import load_pantavou, subject_splits, BASE
from glowai.estimators import OrdinalLogistic, evaluate

LAMBDA=10.0
CLASSES=np.arange(-3,4)

def personal_shift(model,hist):
    y=hist.tsv.to_numpy(dtype=int)+3
    scores=model.score_raw(hist)
    th=model.thresholds
    def obj(shift):
        p=np.diff(expit(th[None,:]-scores[:,None]-shift),axis=1)
        return -np.log(np.maximum(p[np.arange(len(y)),y],1e-14)).sum()+LAMBDA*shift**2
    r=minimize_scalar(obj,bounds=(-6,6),method='bounded')
    if not r.success: raise RuntimeError('optimization failed')
    return r.x

def select_history(prev,k,strategy):
    if k is None or len(prev)<=k: return prev
    if strategy=='recent':return prev.tail(k)
    if strategy=='quantiles':
        # Data-adaptive quantiles based only on available historical weather,
        # chosen without looking at calibration labels or future weather.
        s=prev.sort_values(['air_temperature_c','timestamp_local']).reset_index(drop=False)
        # Unique evenly spaced temperature ranks, a deterministic sample
        loc=np.rint(np.linspace(0,len(s)-1,k)).astype(int)
        assert len(set(loc))==k
        return s.iloc[loc].set_index('index',drop=True).sort_values('timestamp_local')
    if strategy=='days':
        # Coverage across different preceding days; closest to the day's
        # median temperature, then fill remaining slots by quantile coverage.
        days=sorted(prev.local_date.unique())
        indices=np.rint(np.linspace(0,len(days)-1,min(k,len(days)))).astype(int)
        selected=[]
        for d in np.array(days)[indices]:
            group=prev.loc[prev.local_date==d]
            temp_mid=group.air_temperature_c.median()
            idx=(group.air_temperature_c-temp_mid).abs().idxmin()
            selected.append(idx)
        selected=list(dict.fromkeys(selected))
        if len(selected)<k:
            remain=prev.loc[~prev.index.isin(selected)].sort_values('air_temperature_c')
            add=np.rint(np.linspace(0,len(remain)-1,k-len(selected))).astype(int)
            selected.extend(remain.iloc[add].index.tolist())
        assert len(set(selected))==k
        return prev.loc[selected].sort_values('timestamp_local')
    raise ValueError(strategy)

def run(data_path,out_dir):
    df=load_pantavou(data_path)
    records=[]
    for uid,train,own,test in subject_splits(df):
        model=OrdinalLogistic(BASE).fit(train)
        for date,chunk in test.groupby('local_date',sort=True):
            past=own.loc[own.local_date<date].sort_values(['timestamp_local','source_row_id'])
            assert past.timestamp_local.max()<chunk.timestamp_local.min()
            methods=[('population',None,None),('all_past',None,'all')]
            for k in [10,20]:
                for strategy in ['recent','quantiles','days']:
                    methods.append((f'{strategy}_{k}',k,strategy))
            for name,k,strategy in methods:
                hist=select_history(past,k,strategy) if strategy not in [None,'all'] else past
                shift=0 if name=='population' else personal_shift(model,hist)
                probs=model.predict_shift(chunk,shift)
                preds=CLASSES[np.argmax(np.cumsum(probs,axis=1)>=.5,axis=1)]
                modes=CLASSES[probs.argmax(axis=1)]
                for i,(_,r) in enumerate(chunk.iterrows()):
                    records.append({'user_id':uid,'row_id':int(r.source_row_id),'test_day':str(date),
                                    'method':name,'n_prior':len(past),'n_used':0 if name=='population' else len(hist),
                                    'n_days_used':0 if name=='population' else hist.local_date.nunique(),
                                    'past_temp_min':past.air_temperature_c.min(),'past_temp_max':past.air_temperature_c.max(),
                                    'selected_temp_min':np.nan if name=='population' else hist.air_temperature_c.min(),
                                    'selected_temp_max':np.nan if name=='population' else hist.air_temperature_c.max(),
                                    'true_tsv':int(r.tsv),'mae':abs(int(r.tsv)-int(preds[i])),
                                    'correct':int(r.tsv==modes[i]),
                                    **{f'p_{j}':float(probs[i,z]) for z,j in enumerate(CLASSES)}})
        print('evaluated subject',uid,flush=True)
    pred=pd.DataFrame(records)
    assert pred.groupby('method').size().eq(350).all()
    assert not pred[['user_id','row_id','method']].duplicated().any()
    keys=[f'p_{j}' for j in CLASSES]
    summary=[];pers=[]
    for method,g in pred.groupby('method'):
        byuser=g.groupby('user_id').agg(n=('mae','size'),mae=('mae','mean'),accuracy=('correct','mean'),temp_span=('selected_temp_max',lambda q:q.mean())).reset_index()
        byuser['method']=method
        pers.append(byuser)
        met=evaluate(g.true_tsv.to_numpy(),g[keys].to_numpy())
        summary.append({'method':method,'n_test':len(g),'n_people':len(byuser),'mae_mean_subjects':byuser.mae.mean(),
                        'accuracy_mean_subjects':byuser.accuracy.mean(),'pooled_mae':met['mae_median'],
                        'pooled_accuracy':met['accuracy7'],'rps':met['rps']})
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    for fname,frame in [('calibration_distribution_predictions.csv',pred),('calibration_distribution_metrics.csv',pd.DataFrame(summary)),('calibration_distribution_by_person.csv',pd.concat(pers,ignore_index=True))]:
        frame.to_csv(out/fname,index=False)
    print(pd.DataFrame(summary).sort_values('mae_mean_subjects').round(4).to_string(index=False))
    return pred

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--data',required=True);pa.add_argument('--out',default='results')
    a=pa.parse_args();run(a.data,a.out)
