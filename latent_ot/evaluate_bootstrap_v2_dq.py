"""Evaluate a TB-derivation bootstrap map on fixed MC and TB validation rows."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import evaluate_nominal_v2_dq as E
import train_conditional_ot_v2_dq as T
from train_bootstrap_v2_dq import bootstrap_positions

HERE=Path(__file__).resolve().parent

def verify_bootstrap_selection(saved, mc, tb, index, common, seed):
    mc_rows=np.flatnonzero((mc['label']==index)&np.isin(mc['energy'],common))
    original=np.flatnonzero((tb['label']==index)&tb['is_derivation']&np.isin(tb['energy'],common))
    sampled=original[bootstrap_positions(tb['energy'][original],seed)]
    for key,value in [('mc_rows',mc_rows),('original_tb_derivation_rows',original),
                      ('tb_derivation_rows',sampled),('mc_parquet_rows',mc['parquet_row'][mc_rows]),
                      ('tb_derivation_parquet_rows',tb['parquet_row'][sampled]),
                      ('mc_energy',mc['energy'][mc_rows]),('tb_energy',tb['energy'][sampled])]:
        np.testing.assert_array_equal(saved[key],value)
    return mc_rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit',type=Path,required=True)
    ap.add_argument('--nominal-root',type=Path,required=True)
    ap.add_argument('--nominal-evaluation',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--ckpt',type=Path,required=True)
    ap.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(8)
    config=json.loads((args.fit/'config.json').read_text())
    report=json.loads((args.fit/'verification.json').read_text())
    if config.get('analysis')!='bootstrap' or config.get('smoke_test') is not False or report.get('reload_invariance_passed') is not True:
        raise ValueError('A completed full bootstrap fit is required.')
    cls=config['class_name']; index=E.CLASSES.index(cls)
    nominal=json.loads((args.nominal_root/cls/'config.json').read_text())
    for key in ['architecture','batch_size','outer_steps','identity_steps','seed','common_energies','checkpoint_sha256','tb_split_sha256','latent_archive_sha256','g_updates','f_updates','lr','min_lr','betas','weight_decay','grad_clip']:
        if config[key]!=nominal[key]:
            raise ValueError(f'Nominal configuration mismatch: {key}')
    nominal_hash=T.sha256_file(args.nominal_root/cls/'map.pt')
    if config['nominal_map_sha256']!=nominal_hash or T.sha256_file(args.ckpt)!=config['checkpoint_sha256']:
        raise ValueError('Nominal map or PID checkpoint changed.')
    paths={d:HERE/f'derived_v2_dq/latents_{d}.npz' for d in ['MC','TB']}
    hashes={d:T.sha256_file(p) for d,p in paths.items()}
    mask_path=HERE/'derived_v2/tb_is_derivation_v2_dq.npy'
    mask_hash=T.sha256_file(mask_path)
    if hashes!=config['latent_archive_sha256'] or mask_hash!=config['tb_split_sha256']:
        raise ValueError('Input fingerprints changed.')
    mask=np.load(mask_path,allow_pickle=False)
    mc=E.load_archive(paths['MC'],'MC',mask,mask_hash)
    tb=E.load_archive(paths['TB'],'TB',mask,mask_hash)
    with np.load(args.fit/'selection.npz',allow_pickle=False) as saved:
        rows=verify_bootstrap_selection(saved,mc,tb,index,config['common_energies'],config['bootstrap_seed'])
    validation=np.flatnonzero((tb['label']==index)&~tb['is_derivation']&np.isin(tb['energy'],config['common_energies']))
    baseline_dir=args.nominal_evaluation/'efficiency'
    provenance=json.loads((baseline_dir/'provenance.json').read_text())
    completion=json.loads((baseline_dir/'completion.json').read_text())
    if completion.get('completed') is not True or provenance['latent_archive_sha256']!=hashes or provenance['map_sha256'][cls]!=nominal_hash or provenance['mask_sha256']!=mask_hash:
        raise ValueError('Nominal evaluation provenance mismatch.')
    model=E.load_pid_head(args.ckpt,HERE.parent/'transformer_no_veto_v2/pid_model.py',args.device)
    actual=E.head_logits(model,mc['z'][rows[:1024]],args.device,256)
    np.testing.assert_allclose(actual,mc['logits'][rows[:1024]],rtol=2e-5,atol=2e-5)
    transport=T.ConditionalTransport.load(args.fit/'map.pt',device=args.device)
    predictions=np.empty(len(rows),dtype=np.int64)
    for start in range(0,len(rows),20000):
        batch=rows[start:start+20000]
        z=transport.transform(mc['z'][batch],mc['energy'][batch],batch_size=256)
        predictions[start:start+len(batch)]=E.head_logits(model,z,args.device,256).argmax(1)
        print(f'{cls}: evaluated {start+len(batch):,}/{len(rows):,}',flush=True)
    baseline=pd.concat([pd.read_csv(baseline_dir/'efficiency_integrated.csv'),pd.read_csv(baseline_dir/'efficiency_by_energy.csv')],ignore_index=True)
    records=[]
    for energy in ['all_common']+config['common_energies']:
        all_energy=energy=='all_common'
        mi=np.ones(len(rows),dtype=bool) if all_energy else mc['energy'][rows]==energy
        ti=np.ones(len(validation),dtype=bool) if all_energy else tb['energy'][validation]==energy
        match=baseline.energy_mev.astype(str).eq('all_common') if all_energy else np.isclose(pd.to_numeric(baseline.energy_mev,errors='coerce'),energy,rtol=0,atol=1e-4)
        previous=baseline[(baseline.cls==cls)&match]
        if len(previous)!=1:
            raise ValueError('Missing or ambiguous baseline row.')
        previous=previous.iloc[0]
        counts={'n_MC':int(mi.sum()),'n_TB_validation':int(ti.sum()),'correct_raw_MC':int((mc['logits'][rows[mi]].argmax(1)==index).sum()),'correct_TB_validation':int((tb['logits'][validation[ti]].argmax(1)==index).sum())}
        for key,value in counts.items():
            if value!=int(previous[key]):
                raise ValueError(f'Baseline count mismatch: {key}')
        correct=int((predictions[mi]==index).sum())
        records.append(dict(cls=cls,energy_mev=energy,replica=config['replica'],bootstrap_seed=config['bootstrap_seed'],**counts,correct_bootstrap_MC=correct,bootstrap_MC=correct/counts['n_MC'],raw_MC=float(previous.raw_MC),nominal_MC=float(previous.nominal_MC),TB_validation=float(previous.TB_validation)))
    if T.sha256_file(mask_path)!=mask_hash:
        raise RuntimeError('TB split changed.')
    args.out.mkdir(parents=True)
    pd.DataFrame(records).to_csv(args.out/'efficiency.csv',index=False)
    np.savez_compressed(args.out/'predictions.npz',mc_latent_row=rows,mc_parquet_row=mc['parquet_row'][rows],prediction=predictions)
    E.write_json(args.out/'completion.json',dict(completed=True,replica=config['replica'],class_name=cls,map_sha256=T.sha256_file(args.fit/'map.pt'),nominal_map_sha256=nominal_hash,TB_validation_only=True,bootstrap_selection_verified=True,mask_unchanged=True,table_sha256=T.sha256_file(args.out/'efficiency.csv')))
    print(pd.DataFrame(records).to_string(index=False),flush=True)

if __name__=='__main__':
    main()
