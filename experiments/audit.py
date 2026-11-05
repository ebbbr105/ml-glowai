"""Generate cluster-aware diagnostics from benchmark predictions.
No tuning or selection; all displayed comparisons are descriptive.

Usage: python experiments/audit.py --pred results/predictions.csv --out results
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

LABELS = np.arange(-3,4)
PROB_COLS = [f'p_{k}' for k in LABELS]


def one_model(d: pd.DataFrame, model: str) -> pd.DataFrame:
    out=d[d.model==model].copy()
    if len(out)==0:raise ValueError(model)
    p=out[PROB_COLS].to_numpy(float)
    out['pred7']=LABELS[p.argmax(axis=1)]
    out['pred_median7']=LABELS[(np.cumsum(p,axis=1)>=0.5).argmax(axis=1)]
    out['mae']=abs(out.pred_median7-out.y)
    out['error']=out.pred7-out.y
    out['true3']=np.sign(out.y)
    pp3=np.column_stack((p[:,:3].sum(axis=1),p[:,3],p[:,4:].sum(axis=1)))
    out['pred3']=np.array([-1,0,1])[pp3.argmax(axis=1)]
    out['hit3']=(out.true3==out.pred3).astype(int)
    out['hit7']=(out.y==out.pred7).astype(int)
    return out


def subject_mean_delta(d,a,b,metric):
    meanA=one_model(d,a).groupby('user_id')[metric].mean().sort_index()
    meanB=one_model(d,b).groupby('user_id')[metric].mean().sort_index()
    delta=(meanB-meanA)
    rng=np.random.default_rng(20261105)
    cluster_bs=rng.choice(delta.to_numpy(),size=(10000,len(delta)),replace=True).mean(axis=1)
    ci=np.quantile(cluster_bs,[.025,.975]).tolist()
    return {'a':a,'b':b,'metric':metric,'mean_A':float(meanA.mean()),'mean_B':float(meanB.mean()),'delta_B_minus_A':float(delta.mean()),'positive_people':int((delta>0).sum()),'negative_people':int((delta<0).sum()),'equal_people':int((delta==0).sum()),'ci95_people_bootstrap_low':ci[0],'ci95_people_bootstrap_high':ci[1]}


def run(pred,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(pred)
    assert d.groupby(['model','user_id']).size().groupby('model').sum().nunique()==1
    # Exactly matching test events between every tested model.
    anchors=None
    for model,g in d.groupby('model'):
        keys=set(zip(g.user_id,g.source_row_id))
        if anchors is None:anchors=keys
        assert keys==anchors,(model,len(keys),len(anchors))
        assert g[['user_id','source_row_id']].duplicated().sum()==0
    assessments=[]
    for name in sorted(d.model.unique()):
        p=one_model(d,name)
        person=p.groupby('user_id').agg(n=('y','size'),accuracy7=('hit7','mean'),mae=('mae','mean'),accuracy3=('hit3','mean'))
        assessments.append({'model':name,'n_test':len(p),'mean_person_accuracy7':person.accuracy7.mean(), 'mean_person_accuracy3':person.accuracy3.mean(),'mean_person_mae':person.mae.mean()})
    pd.DataFrame(assessments).sort_values('mean_person_mae').to_csv(out/'audit_3class.csv',index=False)
    pairs=[
        ('dummy_most_frequent','ordinal_base4'),
        ('dummy_median','ordinal_base4'),
        ('ordinal_temp','ordinal_base4'),
        ('ordinal_base4','xgboost_base4'),
        ('ordinal_base4','catboost_base4'),
        ('ordinal_base4','ordinal_radiation_diagnostic'),
        ('ordinal_base4','ordinal_wearables_diagnostic'),
        ('ordinal_base4','ordinal_base4_personal_shift_l10'),
        ('ordinal_radiation_diagnostic','ordinal_radiation_diagnostic_personal_shift_l10'),
    ]
    effects=[subject_mean_delta(d,a,b,'mae') for a,b in pairs]
    effects +=[subject_mean_delta(d,a,b,'hit7') for a,b in pairs]
    pd.DataFrame(effects).to_csv(out/'audit_paired_effects.csv',index=False)
    for name in ['ordinal_base4','ordinal_radiation_diagnostic']:
        p=one_model(d,name)
        errs=p.groupby('y').agg(n=('mae','size'),accuracy7=('hit7','mean'),mean_abs_error=('mae','mean')).reset_index()
        errs['model']=name
        errs.to_csv(out/f'audit_errors_{name}.csv',index=False)
        confusion=pd.DataFrame(confusion_matrix(p.y,p.pred7,labels=LABELS),index=[f'true_{z}' for z in LABELS],columns=[f'pred_{z}' for z in LABELS]);confusion.to_csv(out/f'audit_confusion_{name}.csv')
    print('\nThree-class diagnostic:')
    print(pd.DataFrame(assessments).query("model in ['ordinal_base4','ordinal_radiation_diagnostic','xgboost_base4','catboost_base4','ordinal_base4_personal_shift_l10']").to_string(index=False))
    print('\nSelected paired delta MAE (negative=improved):')
    print(pd.DataFrame(effects).query("metric=='mae'")[['a','b','delta_B_minus_A','positive_people','negative_people','ci95_people_bootstrap_low','ci95_people_bootstrap_high']].to_string(index=False))
    return effects

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--pred',default='results/predictions.csv');a.add_argument('--out',default='results');arg=a.parse_args()
    run(arg.pred,arg.out)
