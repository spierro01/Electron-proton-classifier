"""Run from latent_ot on Spark after copying this script and both final masks."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()

def main():
    root=Path.cwd()
    package=Path(__file__).resolve().parent
    target=root/'derived_tail_charge'
    target.mkdir(exist_ok=True)
    mask_path=root/'derived_v2/tb_is_derivation_v2_dq.npy'
    assert digest(mask_path)=='286e8091b4e652b839eaa94a3b6099406347c587958aea23f801001fd13152c5'
    split=np.load(mask_path)
    for domain,expected_hash in [('MC','c976267c4e36a404d553363a98ea5f601ade813224aa1c0332020eeefbea83a3'),('TB','318e239238093df35cdcfbca896ba142d6b98d2d51db3ac598636c5cab19113c')]:
        source=root/f'derived_v2_dq/latents_{domain}.npz'
        assert digest(source)==expected_hash
        old=json.loads(source.with_suffix('.verification.json').read_text())
        assert all(old[k] is True for k in ['completed','saved_arrays_verified','reference_predictions_match','mask_unchanged'])
        path=package/f'{domain}_final_selection.npy'
        keep=np.load(path)
        with np.load(source,allow_pickle=False) as data:
            rows=data['parquet_row']; take=keep[rows]; new_rows=rows[take]
            np.testing.assert_array_equal(new_rows,np.flatnonzero(keep))
            keys=['parquet_row','label','energy_mev','logits','z_enc']
            if domain=='TB':keys.append('is_derivation')
            arrays={key:data[key][take] for key in keys}
            if domain=='TB':np.testing.assert_array_equal(arrays['is_derivation'],split[new_rows])
            metadata=json.loads(data['metadata_json'].item())
            metadata.update(n_selected=len(new_rows),parent_archive_sha256=expected_hash,final_mask_sha256=digest(path))
            arrays.update(classes=data['classes'],checkpoint_sha256=data['checkpoint_sha256'],metadata_json=np.array(json.dumps(metadata)))
        dest=target/source.name
        partial=dest.with_suffix('.partial.npz')
        np.savez_compressed(partial,**arrays)
        with np.load(partial,allow_pickle=False) as check:
            for key,val in arrays.items():np.testing.assert_array_equal(check[key],val)
        partial.replace(dest)
        report={k:old[k] for k in ['completed','saved_arrays_verified','reference_predictions_match','mask_unchanged','domain','checkpoint_sha256']}
        report.update(n_selected=len(new_rows),parent_archive_sha256=expected_hash,final_mask_sha256=digest(path),note='Reference agreement inherited from the verified parent on exactly retained rows.')
        dest.with_suffix('.verification.json').write_text(json.dumps(report,indent=2))
        del arrays, val
    run=root/'results/nominal_tail_charge'
    for cls in ['e','p','C']:
        command=[sys.executable,'-u','train_conditional_ot_v2_dq.py','--mc',str(target/'latents_MC.npz'),'--tb',str(target/'latents_TB.npz'),'--mask',str(mask_path),'--class-name',cls,'--out',str(run/cls),'--device','cuda','--hidden','2048','--layers','4','--input-modulation','--batch-size','1024','--seed','42','--steps','10000','--identity-steps','1000','--log-every','100']
        subprocess.run(command,check=True)

if __name__=='__main__':
    main()
