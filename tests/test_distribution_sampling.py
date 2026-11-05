import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from distribution_sampling import select_history


def sample_history():
    x=pd.DataFrame({'air_temperature_c':[10.,11.,15.,20.,21.,24.,29.,30.,33.,35.,36.],
      'local_date':['a','a','b','b','b','c','c','d','d','e','e'],
      'timestamp_local':pd.date_range('2020-01-01',periods=11,freq='h'),
      'tsv':[-3,-3,-2,-1,-1,0,1,2,3,3,3]})
    return x


def test_selection_uses_only_past_weather_and_calendar():
    x=sample_history()
    for strategy in ['recent','quantiles','days']:
        a=select_history(x,5,strategy)
        assert len(a)==5 and a.index.is_unique
        assert set(a.index).issubset(set(x.index))
        shuffled=x.copy();shuffled['tsv']=np.arange(len(x))[::-1]
        b=select_history(shuffled,5,strategy)
        assert a.index.tolist()==b.index.tolist(),strategy


def test_short_history_returns_all():
    x=sample_history().iloc[:3]
    for strategy in ['recent','quantiles','days']:
        assert select_history(x,20,strategy).index.tolist()==x.index.tolist()
