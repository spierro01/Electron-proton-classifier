"""Apply the two analysis cuts on the original fixed-energy parquet rows."""
import argparse
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from selection_diagnostics import HERE, Path, baseline
from reference_charge import charge as ref

def process(path, domain, medians, split, previous):
    signals, species, energies = ref.plane_signals(path)
    calibration = ref.calibrate(signals, species, energies)
    carbon = species == 'C'
    z = np.full(len(species), np.nan)
    for energy in ref.ENERGIES['C']:
        idx = np.flatnonzero(carbon & np.isclose(energies, energy))
        z[idx] = ref.charge(signals, calibration, idx, energy)[0]
    assert np.isfinite(z[carbon]).all()
    assert all(np.isfinite([fit['a'],fit['alpha']]).all() and fit['alpha']>0 and fit['a']>0 for fit in calibration.values())
    del signals
    match = np.load(HERE/f'{domain}_strip_match_only.npy')
    assert match.shape == z.shape
    prefix = 'file_' if domain=='MC' else ''
    columns = ['particle_type','energy_mev','theta',prefix+'run_nr',prefix+'boot_nr'] + baseline.TR1 + baseline.TR2 + baseline.VETO + baseline.EDEP
    parts=[]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns,batch_size=65536):
        frame=batch.to_pandas()
        labels=frame.particle_type.to_numpy()
        stages=baseline.base_selection_stages(frame,labels,domain,['e','p','C'])
        stages+=baseline.btf_selection_stages(frame,labels,dict(zip(['TR1','TR2'],medians)))
        parts.append(np.logical_and.reduce([mask for _,mask in stages]))
    before=np.concatenate(parts) & match
    final=before & (~carbon | (z>=2.5))
    groups={'all':np.ones(len(z),bool)} if domain=='MC' else {'derivation':split,'validation':~split}
    rows=[]
    for group,mask in groups.items():
        for cls in ('e','p','C'):
            selected=mask & before & (species==cls)
            assert selected.sum()==previous.loc[(domain,group,cls),'after_strip_match']
            for energy in sorted(np.unique(energies[selected])):
                at_energy=selected & (energies==energy)
                rows.append((domain,group,cls,float(energy),int(at_energy.sum()),int((at_energy & final).sum())))
    np.save(HERE/f'{domain}_final_selection.npy',final)
    return calibration,rows

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path.home()/'Desktop/tesi')
    data=parser.parse_args().data
    splitpath=data/'tb_is_derivation_v2_dq.npy'
    assert baseline.sha256(splitpath)=='286e8091b4e652b839eaa94a3b6099406347c587958aea23f801001fd13152c5'
    split=np.load(splitpath)
    previous=pd.read_csv(HERE/'strip_cutflow.csv').set_index(['domain','subset','species'])
    report={}
    rows=[]
    for domain,file,medians in [('MC','dumpMC_v2.parquet',(0.266472727060318,1.4173989295959473)),('TB','dumpTB_v2_dq.parquet',(0.3167046308517456,1.4290019273757935))]:
        report[domain],summary=process(data/file,domain,medians,split,previous)
        rows.extend(summary)
        print(domain,'verified',flush=True)
    (HERE/'charge_calibration.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    table=pd.DataFrame(rows,columns=['domain','subset','species','energy_mev','after_strip_match','final_events'])
    table.to_csv(HERE/'final_counts_by_energy.csv',index=False)
    integrated=table.groupby(['domain','subset','species'])[['after_strip_match','final_events']].sum()
    integrated.to_csv(HERE/'final_counts.csv')
    print(integrated)
