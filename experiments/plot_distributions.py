"""Show the actual observed Pantavou class and weather distributions.

Run: PYTHONPATH=src python experiments/plot_distributions.py \
  --data data/raw/mmc3.xlsx --out results
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glowai.dataset import load_pantavou


def run(data_path,out_dir):
    df=load_pantavou(data_path)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    tab=pd.crosstab(df.user_id,df.tsv).reindex(columns=range(-3,4),fill_value=0)
    tab.to_csv(out/'tsv_distribution_by_person.csv')
    tab.sum().rename('count').to_csv(out/'tsv_distribution_all.csv')
    fig,ax=plt.subplots(figsize=(8.4,4.3))
    counts=df.tsv.value_counts().sort_index().reindex(range(-3,4),fill_value=0)
    ax.bar([str(i) for i in counts.index],counts.values)
    ax.set(xlabel='TSV class (-3 cold ... +3 hot)', ylabel='Number of surveys',title='Pantavou: empirical thermal sensation distribution')
    ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(out/'tsv_distribution.png',dpi=150);plt.close(fig)
    # Empirical distributions, not hypothetical Gaussian curves.
    fig,ax=plt.subplots(figsize=(8.4,4.3))
    for uid,g in df.groupby('user_id'):
        ax.hist(g.air_temperature_c,bins=np.arange(5,41,3),alpha=.3,label=f'Person {uid}')
    ax.set(xlabel='Observed air temperature (°C)',ylabel='Number of surveys',title='Pantavou: actual air temperature by person')
    ax.legend(ncol=5,loc='upper left',fontsize=8)
    fig.tight_layout();fig.savefig(out/'temperature_distributions.png',dpi=150);plt.close(fig)
    print('Participant class distribution:')
    print(tab.to_string())
    return tab

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='results')
    a=p.parse_args();run(a.data,a.out)
