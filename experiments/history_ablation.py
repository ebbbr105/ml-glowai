"""Sensitivity study: personal adaptation with 5/10/20/all preceding answers.

Protocol: leave-one-person-out for the global model. A new user is never in
population fit. For prediction on day D, personal calibration only sees labels
from strictly earlier calendar days. Uses fixed regularization lambda=10.

Run: PYTHONPATH=src python experiments/history_ablation.py \
 --data data/raw/mmc3.xlsx --out results
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from glowai.dataset import load_pantavou, subject_splits, BASE, RAD
from glowai.estimators import OrdinalLogistic, evaluate, CLASSES

CAPS = [5, 10, 20, None]
LAMBDA = 10.0


def fit_personal_intercept(model, history, lam=LAMBDA):
    """Regularized scalar shift fitted on labeled historical surveys only."""
    y = history.tsv.to_numpy(dtype=int) + 3
    scores = model.score_raw(history)
    thresholds = model.thresholds

    def nll(shift):
        cdf = expit(thresholds[None, :] - scores[:, None] - shift)
        p = np.diff(cdf, axis=1)
        return -np.log(np.maximum(p[np.arange(len(y)), y], 1e-14)).sum() + lam * shift**2

    res = minimize_scalar(nll, method='bounded', bounds=(-6, 6), options={'xatol': 1e-7})
    if not res.success:
        raise RuntimeError('Personal shift optimization did not converge')
    return float(res.x)


def run(data_path, out_dir):
    df = load_pantavou(data_path)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []

    for uid, train, own, test in subject_splits(df):
        print('Fold', uid, 'train', len(train), 'test', len(test), flush=True)
        for kind, features in [('base4', BASE), ('radiation_diagnostic', BASE + RAD)]:
            model = OrdinalLogistic(features).fit(train)
            for day, chunk in test.groupby('local_date', sort=True):
                chunk = chunk.sort_values(['timestamp_local', 'source_row_id'])
                prev = own.loc[own.local_date < day].sort_values(['timestamp_local', 'source_row_id'])
                assert len(prev) and prev.timestamp_local.max() < chunk.timestamp_local.min()
                n_history_days = prev.local_date.nunique()
                general_prob = model.predict_proba(chunk)
                results = [('population', general_prob, 0.0)]
                for cap in CAPS:
                    hist = prev.tail(cap) if cap is not None else prev
                    assert hist.timestamp_local.max() < chunk.timestamp_local.min()
                    shift = fit_personal_intercept(model, hist)
                    results.append((f'last_{cap}' if cap is not None else 'all_past',
                                    model.predict_shift(chunk, shift), shift))
                for method, probs, shift in results:
                    pred_median = CLASSES[(np.cumsum(probs, axis=1) >= .5).argmax(axis=1)]
                    pred_mode = CLASSES[probs.argmax(axis=1)]
                    for idx, (_, r) in enumerate(chunk.iterrows()):
                        records.append({
                            'user_id': int(uid), 'source_row_id': int(r.source_row_id),
                            'local_date': day, 'model_features': kind,
                            'history': method, 'available_previous_answers': int(len(prev)),
                            'previous_distinct_days': int(n_history_days),
                            'used_previous_answers': int(min(len(prev), int(method[5:])) if method.startswith('last_') else len(prev)) if method != 'population' else 0,
                            'shift': shift, 'tsv': int(r.tsv),
                            'median_prediction': int(pred_median[idx]), 'mode_prediction': int(pred_mode[idx]),
                            'mae': abs(int(pred_median[idx]) - int(r.tsv)),
                            'accuracy': int(pred_mode[idx] == int(r.tsv)),
                            **{f'p_{k}': float(probs[idx,j]) for j,k in enumerate(CLASSES)},
                        })

    preds = pd.DataFrame.from_records(records)
    assert len(preds) == 350 * 5 * 2
    assert preds[['model_features','history','user_id','source_row_id']].duplicated().sum() == 0
    assert preds.groupby(['model_features','history']).size().eq(350).all()
    preds.to_csv(output/'history_ablation_predictions.csv', index=False)

    summary = []
    byperson = []
    for features in ['base4','radiation_diagnostic']:
        frame = preds.loc[preds.model_features == features]
        for subset_label, frame_part in [('all_future_days',frame),
                                         ('at_least_20_prior_answers',frame.loc[frame.available_previous_answers >= 20]),
                                         ('at_least_2_prior_days',frame.loc[frame.previous_distinct_days >= 2]),
                                         ('at_least_3_prior_days',frame.loc[frame.previous_distinct_days >= 3])]:
            for method in ['population','last_5','last_10','last_20','all_past']:
                g = frame_part.loc[frame_part.history == method]
                per = g.groupby('user_id').agg(n=('mae','size'),mae=('mae','mean'),accuracy=('accuracy','mean')).reset_index()
                if len(per) != 5: raise RuntimeError('Not all subjects represented')
                per['model_features'] = features
                per['subset'] = subset_label
                per['history'] = method
                byperson.append(per)
                pmat=g[[f'p_{k}' for k in CLASSES]].to_numpy(float)
                allmetrics=evaluate(g.tsv.to_numpy(),pmat)
                summary.append(dict(model_features=features,subset=subset_label,history=method,
                                    test_rows=len(g),persons=len(per),
                                    person_mean_mae=per.mae.mean(),person_mean_accuracy=per.accuracy.mean(),
                                    pooled_mae=allmetrics['mae_median'], pooled_accuracy=allmetrics['accuracy7'],
                                    pooled_rps=allmetrics['rps']))

    summ = pd.DataFrame(summary)
    perperson = pd.concat(byperson,ignore_index=True)
    summ.to_csv(output/'history_ablation.csv',index=False)
    perperson.to_csv(output/'history_ablation_by_person.csv',index=False)
    print('\nBASE4 (all future days):')
    print(summ.query("model_features=='base4' and subset=='all_future_days'")[['history','test_rows','person_mean_mae','person_mean_accuracy']].to_string(index=False))
    print('\nBASE4 (>=20 past answers, same test rows):')
    print(summ.query("model_features=='base4' and subset=='at_least_20_prior_answers'")[['history','test_rows','person_mean_mae','person_mean_accuracy']].to_string(index=False))
    return summ,perperson,preds

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--out',default='results')
    args=p.parse_args()
    run(args.data,args.out)
