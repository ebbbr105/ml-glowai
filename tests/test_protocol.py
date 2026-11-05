"""Regression checks for label ordering, timing and subject isolation."""
import numpy as np
import pandas as pd
from glowai.dataset import load_pantavou, subject_splits
from glowai.estimators import evaluate


def test_metric_one_hot_perfect():
    y=np.arange(-3,4)
    p=np.eye(7)
    s=evaluate(y,p)
    assert s['accuracy7']==1
    assert s['mae_median']==0
    assert s['rps']==0


def test_ordinal_metric_penalizes_far_misses():
    y=np.array([0])
    near=np.eye(7)[[4]] # predict TSV +1
    far=np.eye(7)[[6]]  # predict TSV +3
    assert evaluate(y,near)['mae_median'] < evaluate(y,far)['mae_median']
    assert evaluate(y,near)['rps'] < evaluate(y,far)['rps']


def test_person_splits_and_calendar_history(tmp_path):
    """Synthetic smoke test, independent of third-party data."""
    rows=[]
    for u in range(1,4):
        for day in [1,2,3]:
            rows.append({'user_id':u,'local_date':f'2025-01-0{day}', 'tsv':day-2,
                         'timestamp_local':pd.Timestamp(f'2025-01-0{day}T10:00:00'),
                         'source_row_id':u*10+day})
    df=pd.DataFrame(rows)
    for uid,train,own,test in subject_splits(df):
        assert uid not in train.user_id.unique()
        assert len(test)==2
        assert all(test.timestamp_local > own.timestamp_local.min())
        for day in sorted(test.local_date.unique()):
            prev=own[own.local_date<day]
            assert prev.timestamp_local.max()<test[test.local_date==day].timestamp_local.min()
