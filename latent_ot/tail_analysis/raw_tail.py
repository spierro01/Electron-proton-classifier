"""Measure original classifier scores on the final selected parquet rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from selection_diagnostics import HERE, baseline

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--predictions',type=Path,required=True)
args=parser.parse_args()
data=args.data
predictions=args.predictions
provenance=json.loads((predictions/'provenance.json').read_text())
expected=pd.read_csv(HERE/'final_counts_by_energy.csv').set_index(['domain','subset','species','energy_mev'])
primary={'e':[2.5,2,1,0],'p':[3.5,3,2,1],'C':[5,4,3,2]}
fixed={'e':[2,1,0,-1],'p':[3.5,2.5,1.5,.5],'C':[5,4,3,2]}
records=[]
quantiles=[]
for domain,filename in [('MC','dumpMC_v2.parquet'),('TB','dumpTB_v2_dq.parquet')]:
    assert baseline.sha256(data/filename)==provenance['source_files'][domain]['sha256']
    keep=np.load(HERE/f'{domain}_final_selection.npy')
    if domain=='TB': keep &= ~np.load(data/'tb_is_derivation_v2_dq.npy')
    chunks=[]
    for df in pd.read_csv(predictions/f'predictions_{domain}_selected.csv',chunksize=200000):
        df=df.loc[keep[df.parquet_row.to_numpy()]]
        chunks.append(df)
    df=pd.concat(chunks,ignore_index=True)
    np.testing.assert_array_equal(df.parquet_row,np.flatnonzero(keep))
    for c,part in df.groupby('truth'):
        others=[f'p_{i}' for i in ['e','p','C'] if i!=c]
        d=np.log(2*part[f'p_{c}'].to_numpy()/part[others].sum(axis=1).to_numpy())
        assert np.isfinite(d).all()
        for energy, group in part.assign(D=d).groupby('energy_mev'):
            x=group.D.to_numpy(); n=len(x)
            assert n==expected.loc[(domain,'all' if domain=='MC' else 'validation',c,energy),'final_events']
            q=np.quantile(x,[.01,.03,.05,.1,.5,.95,.99])
            quantiles.append([domain,c,energy,n,*q])
            for cut in sorted(set(primary[c]+fixed[c])):
                k=int((x<=cut).sum()); f=k/n
                center=(f+.5/n)/(1+1/n); half=np.sqrt(f*(1-f)/n+.25/n**2)/(1+1/n)
                records.append([domain,c,energy,cut,k,n,f,center-half,center+half])
        fig,ax=plt.subplots(figsize=(8,4))
        for energy, group in part.assign(D=d).groupby('energy_mev'):
            ax.hist(group.D,bins=np.linspace(np.floor(d.min()),np.ceil(d.max()),180),histtype='step',density=True,label=f'{energy:g}')
        for cut in fixed[c]:ax.axvline(cut,color='0.4',ls=':',lw=1)
        ax.set_yscale('log');ax.set_xlabel('Signal discriminant D');ax.set_ylabel('Probability density')
        ax.set_title(f'{domain} '+{'C':'carbon','p':'protons','e':'electrons'}[c]+' before OT')
        ax.legend(title='Beam energy [MeV/n]' if c=='C' else 'Beam energy [MeV]',ncol=3,fontsize=8)
        fig.tight_layout();fig.savefig(HERE/f'{domain}_{c}_tail_diagnostic.pdf');plt.close(fig)
    del chunks,df
pd.DataFrame(quantiles,columns=['domain','species','energy_mev','n','q01','q03','q05','q10','median','q95','q99']).to_csv(HERE/'score_quantiles.csv',index=False)
result=pd.DataFrame(records,columns=['domain','species','energy_mev','candidate_cut','tail_count','n','tail_fraction','wilson_low','wilson_high'])
result.to_csv(HERE/'candidate_tail_fractions.csv',index=False)
result.loc[[r.candidate_cut in fixed[r.species] for r in result.itertuples()]].rename(columns={'candidate_cut':'cut'}).to_csv(HERE/'raw_tail_fractions.csv',index=False)
