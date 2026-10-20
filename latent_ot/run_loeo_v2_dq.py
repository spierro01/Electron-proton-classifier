"""Run all 25 LOEO fits and held-out evaluations, preserving completed jobs.

Rerun the same command to skip verified jobs. Incomplete fits start again in
a new attempt directory; optimiser state is not resumed. No files are deleted.
"""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CLASSES = ['e', 'p', 'C']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('Running:', ' '.join(map(str, command)), flush=True)
    with log.open('x', encoding='utf-8') as stream:
        with subprocess.Popen(list(map(str, command)), cwd=HERE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f'Command failed with exit code {code}. Inspect {log}')


def find_finished_fit(job, cls, energy, nominal, smoke):
    for path in sorted(job.glob('attempt_*'), reverse=True):
        if not (path/'map.pt').is_file() or not (path/'verification.json').is_file():
            continue
        report, config = read_json(path/'verification.json'), read_json(path/'config.json')
        steps, identity = (20, 10) if smoke else (nominal['outer_steps'], nominal['identity_steps'])
        if (report.get('reload_invariance_passed') is True and report.get('excluded_energy_absent_from_fitting') is True
            and config.get('class_name') == cls and config.get('excluded_energy') == energy
            and config.get('smoke_test') == smoke and config.get('outer_steps') == steps
            and config.get('identity_steps') == identity and config.get('architecture') == nominal['architecture']):
            return path
    return None


def next_attempt(job, prefix):
    job.mkdir(parents=True, exist_ok=True)
    number = 1
    while (job/f'{prefix}_{number:03d}').exists():
        number += 1
    return job/f'{prefix}_{number:03d}'


def summarize(records, out):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = pd.DataFrame(records)
    table['class_order'] = table.cls.map({c: i for i, c in enumerate(CLASSES)})
    table = table.sort_values(['class_order','energy_mev']).drop(columns='class_order')
    folder = out/'summary'
    folder.mkdir(exist_ok=True)
    temporary = folder/'loeo_summary.partial.csv'
    table.to_csv(temporary, index=False)
    temporary.replace(folder/'loeo_summary.csv')
    (folder/'status.json').write_text(json.dumps({'completed_evaluations':len(table),'expected_evaluations':25,
                                               'completed':len(table)==25}, indent=2), encoding='utf-8')
    if len(table) != 25:
        return
    names = {'e':'Electrons','p':'Protons','C':'Carbon'}
    for metric, title in [('LOEO_minus_nominal_pp','LOEO sensitivity relative to the nominal map'),
                           ('LOEO_minus_TB_pp','LOEO efficiency differences relative to TB validation')]:
        fig, axes = plt.subplots(1,3,figsize=(16,5))
        for ax, cls in zip(axes,CLASSES):
            group = table[table.cls==cls]
            ax.plot(group.energy_mev,group[metric],color='tab:blue',alpha=.55)
            for kind, marker, color in [('interpolation','o','tab:blue'),('extrapolation','s','tab:orange')]:
                part = group[group.type==kind]
                ax.scatter(part.energy_mev,part[metric],marker=marker,color=color,label=kind.capitalize(),zorder=3)
            ax.axhline(0,color='black',linestyle='--',linewidth=1)
            ax.set_title(names[cls],fontweight='bold')
            ax.set_xlabel('Beam energy [MeV/n]' if cls=='C' else 'Beam energy [MeV]',fontweight='bold')
            ax.set_ylabel('LOEO - nominal [pp]' if metric=='LOEO_minus_nominal_pp' else 'LOEO - TB [pp]',fontweight='bold')
            ax.grid(alpha=.25)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
        fig.suptitle(title,fontweight='bold')
        fig.tight_layout(rect=(0,.09,1,.94))
        for ext in ['png','pdf']:
            fig.savefig(folder/f'{metric}.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,cls in zip(axes,CLASSES):
        group=table[table.cls==cls]
        for key,label,color,marker in [('nominal_MC','Nominal OT','tab:green','o'),('LOEO_MC','LOEO OT','tab:orange','s'),('TB_validation','TB validation','black','o')]:
            ax.plot(group.energy_mev,100*group[key],marker=marker,color=color,label=label)
        ax.set_title(names[cls],fontweight='bold')
        ax.set_xlabel('Beam energy [MeV/n]' if cls=='C' else 'Beam energy [MeV]',fontweight='bold')
        ax.set_ylabel('PID efficiency [%]',fontweight='bold')
        ax.grid(alpha=.25)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Identification efficiency: nominal and leave-one-energy-out OT',fontweight='bold')
    fig.tight_layout(rect=(0,.09,1,.94))
    for ext in ['png','pdf']:
        fig.savefig(folder/f'efficiency_vs_energy.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    (folder/'README.md').write_text('''# LOEO results
Each point uses a separate fit excluding that energy from MC and TB derivation.
Latent and energy normalisation are recomputed on retained fitting events.
DQ selections and their nominal median thresholds remain fixed.
Nominal and LOEO predictions are compared on identical held-out-energy MC rows.
TB efficiency uses validation only. Differences are percentage points (pp).
LOEO minus nominal is an energy-omission sensitivity, not a statistical standard
deviation or an automatic confidence interval. Endpoints are extrapolation;
interior points are interpolation. Curves are visual guides. Efficiency figures
show point estimates without counting or map-uncertainty bars.
''', encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nominal-root',type=Path,default=HERE/'results/nominal_v2_dq_20261015_115613')
    parser.add_argument('--nominal-evaluation',type=Path,default=HERE/'results/nominal_v2_dq_evaluation_20261015_171535')
    parser.add_argument('--out',type=Path,default=HERE/'results/loeo_v2_dq')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    args.out=args.out.resolve()
    files={'MC':HERE/'derived_v2_dq/latents_MC.npz','TB':HERE/'derived_v2_dq/latents_TB.npz',
           'mask':HERE/'derived_v2/tb_is_derivation_v2_dq.npy',
           'ckpt':HERE.parent/'transformer_no_veto_v2/checkpoints/pid_spectra_v2_no_veto_selected_20261013_181104/best.pt'}
    scripts={name:HERE/name for name in ['train_loeo_v2_dq.py','evaluate_loeo_v2_dq.py',
             'train_conditional_ot_v2_dq.py','evaluate_nominal_v2_dq.py','conditional_ot_model.py']}
    print('Verifying input fingerprints and nominal runs...',flush=True)
    configs={cls:read_json(args.nominal_root/cls/'config.json') for cls in CLASSES}
    jobs=[(cls,float(energy)) for cls in CLASSES for energy in configs[cls]['common_energies']]
    if len(jobs)!=25:
        raise ValueError(f'Expected 25 LOEO fits, found {len(jobs)}.')
    plan={'nominal_root':str(args.nominal_root.resolve()),'nominal_evaluation':str(args.nominal_evaluation.resolve()),
          'input_hashes':{k:sha256(p) for k,p in files.items()},
          'script_hashes':{k:sha256(p) for k,p in scripts.items()},
          'nominal_map_hashes':{c:sha256(args.nominal_root/c/'map.pt') for c in CLASSES},
          'nominal_configurations':configs,'jobs':[[c,e] for c,e in jobs], 'device':args.device}
    for cls in CLASSES:
        if read_json(args.nominal_root/cls/'verification.json').get('reload_invariance_passed') is not True:
            raise ValueError(f'Nominal map not verified: {cls}')
    if read_json(args.nominal_evaluation/'efficiency/completion.json').get('completed') is not True:
        raise ValueError('Nominal efficiency evaluation is incomplete.')
    args.out.mkdir(parents=True,exist_ok=True)
    plan_path=args.out/'plan.json'
    if plan_path.exists():
        if read_json(plan_path)!=plan:
            raise ValueError('Inputs or scripts differ from the saved plan. Use a new output root.')
    else:
        plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    # A nonblocking host-local lock prevents two runners using the same output.
    import fcntl
    lock=(args.out/'runner.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another runner is already using this output directory.')
    log_dir=args.out/'logs'
    log_dir.mkdir(exist_ok=True)

    def train(cls,energy,job,smoke=False):
        config=configs[cls]
        finished=find_finished_fit(job,cls,energy,config,smoke)
        if finished:
            print('Reusing verified fit:',finished,flush=True)
            return finished
        attempt=next_attempt(job,'attempt')
        command=[sys.executable,'-u',scripts['train_loeo_v2_dq.py'],
            '--class-name',cls,'--exclude-energy',str(energy),'--nominal-dir',args.nominal_root/cls,
            '--mc',files['MC'],'--tb',files['TB'],'--mask',files['mask'],'--device',args.device,
            '--hidden',str(config['architecture']['hidden_dim']),'--layers',str(config['architecture']['num_layers']),
            '--input-modulation' if config['architecture']['input_modulation'] else '--no-input-modulation',
            '--batch-size',str(config['batch_size']),'--seed',str(config['seed']),
            '--steps',str(20 if smoke else config['outer_steps']),
            '--identity-steps',str(10 if smoke else config['identity_steps']),
            '--log-every','1' if smoke else '100','--out',attempt]
        if smoke:
            command.append('--smoke-test')
        log=log_dir/f'{job.name}_{attempt.name}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.log'
        run(command,log)
        if find_finished_fit(job,cls,energy,config,smoke)!=attempt:
            raise RuntimeError(f'Training verification failed: {attempt}')
        return attempt

    smoke_energy=90.0 if 90.0 in configs['e']['common_energies'] else float(configs['e']['common_energies'][len(configs['e']['common_energies'])//2])
    train('e',smoke_energy,args.out/'smoke_e',smoke=True)
    records=[]
    for number,(cls,energy) in enumerate(jobs,1):
        tag=f'{cls}_exclude_{energy:g}'.replace('.','p')
        print(f'\nJOB {number}/25: {cls}, excluded energy {energy:g}',flush=True)
        attempt=train(cls,energy,args.out/tag)
        map_hash=sha256(attempt/'map.pt')
        evaluation=None
        for candidate in sorted(attempt.glob('evaluation_*'),reverse=True):
            if (candidate/'completion.json').is_file():
                report=read_json(candidate/'completion.json')
                if report.get('completed') is True and report.get('loeo_map_sha256')==map_hash and report.get('nominal_map_sha256')==plan['nominal_map_hashes'][cls]:
                    evaluation=candidate
                    break
        if evaluation is None:
            evaluation=next_attempt(attempt,'evaluation')
            command=[sys.executable,'-u',scripts['evaluate_loeo_v2_dq.py'],
                '--class-name',cls,'--energy',str(energy),'--loeo-dir',attempt,
                '--nominal-root',args.nominal_root,'--nominal-evaluation',args.nominal_evaluation,
                '--mc',files['MC'],'--tb',files['TB'],'--mask',files['mask'],'--ckpt',files['ckpt'],
                '--device',args.device,'--out',evaluation]
            run(command,log_dir/f'{tag}_{evaluation.name}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.log')
        report=read_json(evaluation/'completion.json')
        if report.get('completed') is not True:
            raise RuntimeError('LOEO evaluation did not complete.')
        record={**report['metrics'],'map_directory':str(attempt),'evaluation_directory':str(evaluation)}
        records.append(record)
        summarize(records,args.out)
        print(f'Completed evaluations: {len(records)}/25',flush=True)
    print('All 25 LOEO fits and evaluations completed.',flush=True)
    print('Summary:',args.out/'summary/loeo_summary.csv',flush=True)


if __name__=='__main__':
    main()
