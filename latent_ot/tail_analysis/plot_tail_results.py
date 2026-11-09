"""Compare raw MC, nominal OT and LOEO with TB at fixed thresholds."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('results/fixed_tail_campaign')
nominal=pd.read_csv(root/'nominal_tail_fractions.csv')
loeo=pd.read_csv(root/'loeo_tail_fractions.csv')
for cls,title in [('e','Electrons'),('p','Protons'),('C','Carbon')]:
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    cuts=sorted(nominal[nominal.species==cls].cut.unique(),reverse=True)
    for ax,cut in zip(axes.flat,cuts):
        for table,sample,label,color,marker in [
            (nominal,'raw_MC','Original MC','#3475a4','o'),
            (nominal,'calibrated_MC','Nominal OT','#23814a','s'),
            (loeo,'calibrated_MC','LOEO OT','#ce6719','^'),
            (nominal,'TB_validation','TB validation','black','D')]:
            q=table[(table.species==cls)&(table.cut==cut)&(table['sample']==sample)].sort_values('energy_mev')
            assert len(q)==(7 if cls=='e' else 9)
            if sample=='TB_validation':
                errors=[100*(q.fraction-q.wilson_low).clip(lower=0),100*(q.wilson_high-q.fraction).clip(lower=0)]
                ax.errorbar(q.energy_mev,100*q.fraction,yerr=errors,color=color,marker=marker,ls='none',label=label)
            else:
                ax.plot(q.energy_mev,100*q.fraction,color=color,marker=marker,lw=1,label=label)
        ax.set_title(f'D ≤ {cut:g}')
        ax.set_xlabel('Beam energy [MeV/n]' if cls=='C' else 'Beam energy [MeV]')
        ax.set_ylabel('Signal in lower tail [%]')
        ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False) if tuple(map(int,matplotlib.__version__.split('.')[:2])) >= (3,7) else axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'{title}: fixed-threshold tail fractions',fontweight='bold')
    for ext in ['pdf','png']:fig.savefig(root/f'{cls}_tail_comparison.{ext}',dpi=180)
    plt.close(fig)
