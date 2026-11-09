"""Check the track/strip selection on the fixed-energy spectra, before charge selection."""
from pathlib import Path
import argparse
from collections import Counter
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'transformer_no_veto_v2'))
import evaluate_selected_v2_dq as baseline


def strip_columns(plane, n):
    return ([f'{plane}_posx_{i}' for i in range(n)]
            + [f'EDEP_{plane}_HG_{i}' for i in range(n)]
            + [f'RAW_{plane}_{gain}_{i}_{s}' for gain in ('HG', 'LG')
               for i in range(n) for s in (0, 1)])


def strip_match(frame, plane, n):
    pos = frame[[f'{plane}_posx_{i}' for i in range(n)]].to_numpy()
    crossed = pos > -998
    edep = frame[[f'EDEP_{plane}_HG_{i}' for i in range(n)]].to_numpy()
    hg, lg = [frame[[f'RAW_{plane}_{gain}_{i}_{s}' for i in range(n)
                     for s in (0, 1)]].to_numpy() for gain in ('HG', 'LG')]
    saturated = hg.max(axis=1) > 3000
    lg_sum = lg.clip(min=0).reshape(len(frame), n, 2).sum(axis=2)
    fired = np.where(saturated, lg_sum.argmax(axis=1), edep.argmax(axis=1))
    return crossed.any(axis=1) & (crossed.argmax(axis=1) == fired)


def main(data):
    split = np.load(data / 'tb_is_derivation_v2_dq.npy', allow_pickle=False)
    totals = Counter()
    for domain, filename, medians in [
        ('MC', 'dumpMC_v2.parquet', (0.266472727060318, 1.4173989295959473)),
        ('TB', 'dumpTB_v2_dq.parquet', (0.3167046308517456, 1.4290019273757935))]:
        dataset = pq.ParquetFile(data / filename)
        if domain == 'TB':
            assert split.dtype == bool and len(split) == dataset.metadata.num_rows
            assert baseline.sha256(data/'tb_is_derivation_v2_dq.npy') == '286e8091b4e652b839eaa94a3b6099406347c587958aea23f801001fd13152c5'
        prefix = 'file_' if domain == 'MC' else ''
        columns = ['particle_type', 'energy_mev', 'theta', prefix+'boot_nr',prefix+'run_nr']
        columns += baseline.VETO + baseline.EDEP + strip_columns('TR1', 5) + strip_columns('TR2', 4)
        offset = 0
        masks = []
        for batch in dataset.iter_batches(columns=list(dict.fromkeys(columns)), batch_size=65536):
            frame = batch.to_pandas()
            labels = frame.particle_type.replace({'electron':'e','proton':'p','carbon':'C'}).to_numpy()
            stages = baseline.base_selection_stages(frame, labels, domain, ['e','p','C'])
            stages += baseline.btf_selection_stages(frame, labels, dict(zip(['TR1','TR2'], medians)))
            before = np.logical_and.reduce([keep for _, keep in stages])
            matches = strip_match(frame,'TR1',5) & strip_match(frame,'TR2',4)
            masks.append(matches)
            groups = {'all': np.ones(len(frame), bool)} if domain == 'MC' else {
                'derivation':split[offset:offset+len(frame)],'validation':~split[offset:offset+len(frame)]}
            for group, group_rows in groups.items():
                for cls in ('e','p','C'):
                    selected = before & group_rows & (labels==cls)
                    totals[domain,group,cls,'before'] += int(selected.sum())
                    totals[domain,group,cls,'after'] += int((selected & matches).sum())
            offset += len(frame)
        assert offset == dataset.metadata.num_rows
        for cls, expected in zip(('e','p','C'), (933387,1661511,573751) if domain=='MC' else (25499,305028,105076)):
            assert totals[domain,'all' if domain=='MC' else 'validation',cls,'before']==expected
        np.save(HERE/f'{domain}_strip_match_only.npy', np.concatenate(masks))
        print(domain, 'baseline counts verified', flush=True)
    rows=[(*key[:3],value,totals[(*key[:3],'after')]) for key,value in totals.items() if key[-1]=='before']
    table = pd.DataFrame(rows,columns=['domain','subset','species','before_strip_match','after_strip_match'])
    table.to_csv(HERE/'strip_cutflow.csv',index=False)
    print(table.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path.home()/'Desktop/tesi')
    main(parser.parse_args().data)
