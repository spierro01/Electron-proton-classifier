"""Run five TB-derivation bootstrap replicas per species and summarize efficiencies.

Completed, verified jobs are reused. Incomplete fits restart in new directories.
Training randomness, MC, validation events and DQ selections remain fixed.
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import sys
from run_loeo_v2_dq import run, sha256, read_json, next_attempt

HERE=Path(__file__).resolve().parent
CLASSES=['e','p','C']

def summarize(tables,out,replicas):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=pd.concat([pd.read_csv(p) for p in tables],ignore_index=True)
    if data.duplicated(['cls','energy_mev','replica']).any():
        raise ValueError('Duplicate bootstrap results.')
    records=[]
    for (cls,energy),group in data.groupby(['cls','energy_mev'],sort=False):
        for key in ['n_MC','n_TB_validation','raw_MC','nominal_MC','TB_validation']:
            if group[key].nunique()!=1:
                raise ValueError(f'Inconsistent fixed baseline: {key}')
        records.append(dict(cls=cls,energy_mev=energy,replicas=len(group),raw_MC=group.raw_MC.iloc[0],nominal_MC=group.nominal_MC.iloc[0],TB_validation=group.TB_validation.iloc[0],bootstrap_mean=group.bootstrap_MC.mean(),bootstrap_sd_pp=100*group.bootstrap_MC.std(ddof=1),nominal_minus_TB_pp=100*(group.nominal_MC.iloc[0]-group.TB_validation.iloc[0]),bootstrap_mean_minus_nominal_pp=100*(group.bootstrap_MC.mean()-group.nominal_MC.iloc[0])))
    summary=pd.DataFrame(records)
    folder=out/'summary'; folder.mkdir(exist_ok=True)
    data.to_csv(folder/'replica_efficiencies.csv',index=False)
    temp=folder/'bootstrap_summary.partial.csv'; summary.to_csv(temp,index=False); temp.replace(folder/'bootstrap_summary.csv')
    complete=len(tables)==3*replicas and len(summary)==28 and summary.replicas.eq(replicas).all()
    (folder/'status.json').write_text(json.dumps(dict(completed_evaluations=len(tables),expected_evaluations=3*replicas,completed=bool(complete)),indent=2))
    if not complete:
        return
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,cls,title in zip(axes,CLASSES,['Electrons','Protons','Carbon']):
        group=summary[(summary.cls==cls)&(summary.energy_mev.astype(str)!='all_common')].copy()
        group['energy_mev']=pd.to_numeric(group.energy_mev); group=group.sort_values('energy_mev')
        for key,label,color in [('raw_MC','MC original','tab:blue'),('nominal_MC','Nominal OT','tab:green'),('TB_validation','TB validation','black')]:
            ax.plot(group.energy_mev,100*group[key],'-o',label=label,color=color)
        ax.errorbar(group.energy_mev,100*group.bootstrap_mean,yerr=group.bootstrap_sd_pp,fmt='s--',color='tab:orange',capsize=3,label='Bootstrap mean ± 1 SD')
        ax.set_title(title,fontweight='bold')
        ax.set_xlabel('Beam energy [MeV/n]' if cls=='C' else 'Beam energy [MeV]',fontweight='bold')
        ax.set_ylabel('PID efficiency [%]',fontweight='bold'); ax.grid(alpha=.25)
    fig.suptitle('Bootstrap variability of the calibrated identification efficiency',fontweight='bold')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.09,1,.94))
    for extension in ['png','pdf']:
        fig.savefig(folder/f'bootstrap_efficiency.{extension}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    (folder/'README.md').write_text('''# TB derivation bootstrap
TB derivation is resampled within each energy with replacement and unchanged
event counts. MC, the PID head, energy settings, DQ selections and their median
thresholds remain fixed. TB normalisation statistics are recomputed for each
replica. The training seed is fixed; GPU nondeterminism is not separately measured.
Efficiencies are evaluated on identical MC events and fixed TB validation events.
The reported sample SD uses ddof=1 and is in percentage points. It describes
variation across fitted maps, not the standard error of the mean or the total
uncertainty of the MC/TB comparison. Five replicas give a preliminary estimate.
The plot has independent vertical scales; only bootstrap SD bars are displayed.
''',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replicas',type=int,default=5)
    ap.add_argument('--out',type=Path,default=HERE/'results/bootstrap_v2_dq')
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=ap.parse_args()
    if args.replicas<2:
        raise ValueError('At least two replicas are required.')
    args.out=args.out.resolve()
    nominal=HERE/'results/nominal_v2_dq_20261015_115613'
    evaluation=HERE/'results/nominal_v2_dq_evaluation_20261015_171535'
    ckpt=HERE.parent/'transformer_no_veto_v2/checkpoints/pid_spectra_v2_no_veto_selected_20261013_181104/best.pt'
    files={'MC':HERE/'derived_v2_dq/latents_MC.npz','TB':HERE/'derived_v2_dq/latents_TB.npz','mask':HERE/'derived_v2/tb_is_derivation_v2_dq.npy','checkpoint':ckpt}
    script_names=['train_bootstrap_v2_dq.py','evaluate_bootstrap_v2_dq.py','run_bootstrap_v2_dq.py','run_loeo_v2_dq.py','train_conditional_ot_v2_dq.py','evaluate_nominal_v2_dq.py','conditional_ot_model.py','train_ot.py','train_domain_discriminator.py']
    configs={c:read_json(nominal/c/'config.json') for c in CLASSES}
    plan=dict(replicas=args.replicas,device=args.device,input_hashes={k:sha256(p) for k,p in files.items()},script_hashes={k:sha256(HERE/k) for k in script_names},pid_source_sha256=sha256(ckpt.parents[2]/'pid_model.py'),nominal_maps={c:sha256(nominal/c/'map.pt') for c in CLASSES},nominal_configs=configs,nominal_evaluation_provenance=read_json(evaluation/'efficiency/provenance.json'))
    if read_json(evaluation/'efficiency/completion.json').get('completed') is not True:
        raise ValueError('Nominal efficiency evaluation is incomplete.')
    args.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.out/'runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.out/'plan.json').exists():
        if read_json(args.out/'plan.json')!=plan:
            raise ValueError('Plan changed. Use a new output directory.')
    else:
        (args.out/'plan.json').write_text(json.dumps(plan,indent=2))
    tables=[]
    for cls in CLASSES:
        config=configs[cls]
        for replica in range(1,args.replicas+1):
            seed=10000+1000*CLASSES.index(cls)+replica
            job=args.out/f'{cls}_replica_{replica:03d}'
            fit=None
            for candidate in sorted(job.glob('attempt_*'),reverse=True):
                if not all((candidate/n).is_file() for n in ['map.pt','config.json','verification.json']):
                    continue
                c=read_json(candidate/'config.json'); v=read_json(candidate/'verification.json')
                if v.get('reload_invariance_passed') is True and c.get('analysis')=='bootstrap' and c.get('replica')==replica and c.get('bootstrap_seed')==seed and c.get('smoke_test') is False and c.get('class_name')==cls and c.get('outer_steps')==config['outer_steps']:
                    fit=candidate; break
            if fit is None:
                fit=next_attempt(job,'attempt')
                command=[sys.executable,'-u',HERE/'train_bootstrap_v2_dq.py','--nominal-dir',nominal/cls,'--class-name',cls,'--replica',str(replica),'--bootstrap-seed',str(seed),'--mc',files['MC'],'--tb',files['TB'],'--mask',files['mask'],'--device',args.device,'--hidden',str(config['architecture']['hidden_dim']),'--layers',str(config['architecture']['num_layers']),'--input-modulation' if config['architecture']['input_modulation'] else '--no-input-modulation','--batch-size',str(config['batch_size']),'--steps',str(config['outer_steps']),'--identity-steps',str(config['identity_steps']),'--seed',str(config['seed']),'--log-every','100','--out',fit]
                run(command,args.out/'logs'/f'{job.name}_{fit.name}_{datetime.now():%Y%m%d_%H%M%S_%f}.log')
            result=None
            for candidate in sorted(fit.glob('evaluation_*'),reverse=True):
                if (candidate/'completion.json').is_file() and (candidate/'efficiency.csv').is_file():
                    v=read_json(candidate/'completion.json')
                    if v.get('completed') is True and v.get('map_sha256')==sha256(fit/'map.pt') and v.get('table_sha256')==sha256(candidate/'efficiency.csv'):
                        result=candidate; break
            if result is None:
                result=next_attempt(fit,'evaluation')
                run([sys.executable,'-u',HERE/'evaluate_bootstrap_v2_dq.py','--fit',fit,'--nominal-root',nominal,'--nominal-evaluation',evaluation,'--out',result,'--ckpt',ckpt,'--device',args.device],args.out/'logs'/f'{job.name}_{result.name}_{datetime.now():%Y%m%d_%H%M%S_%f}.log')
            if read_json(result/'completion.json').get('completed') is not True:
                raise RuntimeError('Evaluation incomplete.')
            tables.append(result/'efficiency.csv')
            summarize(tables,args.out,args.replicas)
            print(f'Completed bootstrap evaluations: {len(tables)}/{3*args.replicas}',flush=True)
    print('Bootstrap completed. Results:',args.out/'summary',flush=True)

if __name__=='__main__':
    main()
