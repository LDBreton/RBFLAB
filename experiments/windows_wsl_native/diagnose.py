"""Untimed matrix/conditioning check for the larger Float64 Poisson case."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .benchmark import replay, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();output=args.directory.resolve()
    case='poisson-float64-n10-t1'
    commands=json.loads((output/(case+'-input.json')).read_text())
    commands.insert(-1,'EXPORT')
    binary=None
    for folder in (output/'binaries/float64').iterdir():
        manifest=json.loads((output/'cache'/folder.name/'manifest.json').read_text())
        if manifest['identity']['pressure_order']==0:
            binary=folder
    assert binary is not None
    matrices={};responses={}
    for host,name in [('windows','worker.exe'),('wsl','worker-linux')]:
        _, reply=replay(host,binary/name,commands)
        responses[host]=reply
        tokens=next(row['response'].split() for row in reply if row['command']=='EXPORT')
        size=int(tokens[1])
        matrices[host]=np.array(list(map(float,tokens[2:]))).reshape(size,size)
    a,b=matrices['windows'],matrices['wsl']
    gap=float(np.max(np.abs(a-b)))
    condition=float(np.linalg.cond(a))
    result=dict(case=case,unknowns=len(a),matrix_max_abs_gap=gap,
        matrix_relative_frobenius_gap=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
        windows_matrix_condition_2_float64_estimate=condition,
        condition_times_float64_epsilon=condition*float(np.finfo(float).eps),
        note='Untimed exported unscaled augmented matrix; condition number is a Float64 SVD estimate. '
        'Condition*epsilon is sensitivity context, not a forward-error bound for the equilibrated solver.')
    save(output/'matrix-diagnostic.json',result)
    save(output/'matrix-diagnostic-outputs.json',responses)
    print(result)
    evidence={}
    for root in ('binaries','local-binaries'):
        for folder in (output/root).glob('*/*'):
            if folder.is_dir():
                evidence[str(folder.relative_to(output))]={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in folder.iterdir() if p.is_file() and p.suffix not in ('.log','.json')}
    save(output/'artifact-sha256.json',evidence)


if __name__=='__main__':main()
