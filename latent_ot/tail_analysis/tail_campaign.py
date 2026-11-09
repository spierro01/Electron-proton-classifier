"""Train the new LOEO/bootstrap maps and evaluate fixed tails on Spark."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tail_metrics import discriminant, tail_counts

CLASSES = ('e', 'p', 'C')
CUTS = {'e': [2, 1, 0, -1], 'p': [3.5, 2.5, 1.5, .5], 'C': [5, 4, 3, 2]}
ROOT = Path.cwd()
ARCHIVES = ROOT / 'derived_tail_charge'
NOMINAL = ROOT / 'results/nominal_tail_charge'
OUT = ROOT / 'results/fixed_tail_campaign'
SPLIT = ROOT / 'derived_v2/tb_is_derivation_v2_dq.npy'


def read_json(path):
    return json.loads(path.read_text())


def jobs(cls, kind, energies):
    if kind == 'nominal':
        yield NOMINAL / cls, {}, []
    elif kind == 'loeo':
        for energy in energies:
            yield (OUT / 'loeo' / f'{cls}_{energy:g}', {'excluded_energy': energy},
                   ['--exclude-energy', str(energy)])
    elif kind == 'bootstrap':
        for replica in range(1, 6):
            seed = 10000 + 1000 * CLASSES.index(cls) + replica
            yield (OUT / 'bootstrap' / f'{cls}_{replica:03}', {'replica': replica, 'bootstrap_seed': seed},
                   ['--replica', str(replica), '--bootstrap-seed', str(seed)])


def verify(folder, cls, hashes, details, nominal):
    cfg = read_json(folder/'config.json')
    report = read_json(folder/'verification.json')
    assert report['reload_invariance_passed'] and cfg['TB_validation_used_for_fitting'] is False
    assert cfg['class_name'] == cls and cfg['latent_archive_sha256'] == hashes
    assert all(cfg[k] == v for k, v in details.items())
    assert cfg['checkpoint_sha256'] == nominal['checkpoint_sha256']
    if details:
        assert not cfg['smoke_test']
        assert cfg['outer_steps'] == nominal['outer_steps']
    if 'excluded_energy' in details:
        assert not np.isclose(cfg['common_energies'], details['excluded_energy']).any()
    return cfg


def train(kind, device, hashes):
    for cls in CLASSES:
        cfg = read_json(NOMINAL/cls/'config.json')
        verify(NOMINAL/cls, cls, hashes, {}, cfg)
        arch = cfg['architecture']
        for folder, details, flags in jobs(cls,kind,cfg['common_energies']):
            if not (folder/'verification.json').is_file():
                if folder.exists():
                    raise RuntimeError(f'Incomplete fit: {folder}. Preserve it under another name before retrying.')
                command = [sys.executable,'-u',str(ROOT/f'train_{kind}_v2_dq.py'),
                           '--nominal-dir',str(NOMINAL/cls),'--class-name',cls,'--mc',str(ARCHIVES/'latents_MC.npz'),
                           '--tb',str(ARCHIVES/'latents_TB.npz'),'--mask',str(SPLIT),'--out',str(folder),
                           '--device',device,'--hidden',str(arch['hidden_dim']),'--layers',str(arch['num_layers']),
                           '--input-modulation' if arch['input_modulation'] else '--no-input-modulation',
                           '--batch-size',str(cfg['batch_size']),'--steps',str(cfg['outer_steps']),
                           '--identity-steps',str(cfg['identity_steps']),'--seed',str(cfg['seed']),*flags]
                subprocess.run(command,check=True)
            verify(folder,cls,hashes,details,cfg)


def evaluate(kind, device, hashes, helpers, transport):
    pid = ROOT.parent/'transformer_no_veto_v2'
    checkpoint = pid/'checkpoints/pid_spectra_v2_no_veto_selected_20261013_181104/best.pt'
    checkpoint_hash = transport.sha256_file(checkpoint)
    model = helpers.load_pid_head(checkpoint,pid/'pid_model.py',device)
    with np.load(ARCHIVES/'latents_MC.npz') as mc, np.load(ARCHIVES/'latents_TB.npz') as tb:
        np.testing.assert_array_equal(mc['classes'],CLASSES)
        np.testing.assert_array_equal(tb['classes'],CLASSES)
        np.testing.assert_array_equal(tb['is_derivation'], np.load(SPLIT)[tb['parquet_row']])
        validation = ~tb['is_derivation']
        z, energy, labels, raw = (mc[k] for k in ['z_enc','energy_mev','label','logits'])
        tb_energy, tb_labels, tb_logits = (tb[k][validation] for k in ['energy_mev','label','logits'])
    rows=[]
    working_points=[]
    for i,cls in enumerate(CLASSES):
        nominal=read_json(NOMINAL/cls/'config.json')
        assert nominal['checkpoint_sha256']==checkpoint_hash
        energies=nominal['common_energies']
        for folder,details,_ in jobs(cls,kind,energies):
            verify(folder,cls,hashes,details,nominal)
            mapping=transport.ConditionalTransport.load(folder/'map.pt',device=device)
            for e in ([details['excluded_energy']] if kind=='loeo' else energies):
                selected=(labels==i)&np.isclose(energy,e)
                target=(tb_labels==i)&np.isclose(tb_energy,e)
                mapped=mapping.transform(z[selected],energy[selected],batch_size=256)
                scores={'raw_MC':discriminant(raw[selected],i),
                        'calibrated_MC':discriminant(helpers.head_logits(model,mapped,device,256),i),
                        'TB_validation':discriminant(tb_logits[target],i)}
                observed=scores['TB_validation']
                for sample in ('raw_MC','calibrated_MC'):
                    ordered=np.sort(scores[sample])
                    for percent in [60,70,80,90]:
                        rank=(percent*len(ordered)+99)//100
                        threshold=ordered[-rank]
                        passed=int((observed>=threshold).sum())
                        efficiency=passed/len(observed)
                        working_points.append([cls,e,details.get('replica',0),sample,percent,threshold,passed,
                                               len(observed),efficiency,100*efficiency-percent])
                for cut in CUTS[cls]:
                    for sample,values in scores.items():
                        k,n,f=tail_counts(values,cut)
                        center=(f+.5/n)/(1+1/n)
                        half=np.sqrt(f*(1-f)/n+.25/n**2)/(1+1/n)
                        rows.append([cls,e,cut,details.get('replica',0),sample,k,n,f,max(0,center-half),min(1,center+half)])
    table=pd.DataFrame(rows,columns=['species','energy_mev','cut','replica','sample','tail_count','n','fraction','wilson_low','wilson_high'])
    assert len(table)==300*(5 if kind=='bootstrap' else 1)
    OUT.mkdir(parents=True,exist_ok=True)
    table.to_csv(OUT/f'{kind}_tail_fractions.csv',index=False)
    pd.DataFrame(working_points,columns=['species','energy_mev','replica','threshold_source','target_percent',
                 'threshold_D','tb_passed','tb_total','tb_efficiency','residual_pp']).to_csv(
                     OUT/f'{kind}_secondary_wp.csv',index=False)
    if kind=='bootstrap':
        group=table[table['sample']=='calibrated_MC'].groupby(['species','energy_mev','cut']).fraction
        group.agg(['count','mean','std']).to_csv(OUT/'bootstrap_tail_spread.csv')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','evaluate'])
    parser.add_argument('kind',choices=['nominal','loeo','bootstrap'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT))
    import train_conditional_ot_v2_dq as transport
    hashes={d:transport.sha256_file(ARCHIVES/f'latents_{d}.npz') for d in ['MC','TB']}
    if args.action=='train':
        if args.kind=='nominal':parser.error('Use start_nominal.py to prepare archives and train nominal maps.')
        train(args.kind,args.device,hashes)
    else:
        import evaluate_nominal_v2_dq as helpers
        evaluate(args.kind,args.device,hashes,helpers,transport)
