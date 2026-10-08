"""Independent-cloud refinement and matched-backend accuracy measurements."""
import argparse
import json
from pathlib import Path
from examples import annulus, ellipse_boundary, ball_poisson, heat_flower, stokes_annulus


def run(backends=("python",), output=None):
    records=[]
    for seed in (17,42,73):
        for interior in (120,480):
            records.append(annulus.run(interior=interior,seed=seed)[3] | {"seed":seed,"interior":interior})
    for backend in backends:
        for example in (annulus,ellipse_boundary,ball_poisson,heat_flower):
            records.append(example.run(backend=backend)[3] | {"seed":42,"matched_backend":True})
    records.append(stokes_annulus.run()[3])
    if output:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")
    return records


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backends",nargs="+",choices=("python","cpp","torch"),default=["python"])
    p.add_argument("--output",default="outputs/curved_validation.json")
    args=p.parse_args();run(args.backends,args.output)
