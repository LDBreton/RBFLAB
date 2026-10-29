"""Standard-library MPI rank bridge for native local workers (also under WSL).

The launcher supplies rank identity; this bridge does not emulate MPI with a
process pool. It owns one native chunk and writes a validated completion record.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def _identity():
    for rank, size in (("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
                       ("PMI_RANK", "PMI_SIZE"), ("PMIX_RANK", "PMIX_UNIV_SIZE"),
                       ("MV2_COMM_WORLD_RANK", "MV2_COMM_WORLD_SIZE")):
        if rank in os.environ and size in os.environ:
            return int(os.environ[rank]), int(os.environ[size])
    raise RuntimeError("MPI launcher did not provide supported rank/size identity")


def main():
    folder = Path(sys.argv[1])
    config = json.loads((folder / "config.json").read_text())
    rank, size = _identity()
    if size != config["ranks"] or not 0 <= rank < size:
        raise RuntimeError("MPI rank identity does not match requested launch")
    start = time.perf_counter()
    if config["counts"][rank]:
        run = subprocess.run([config["executable"], config["inputs"][rank],
                              config["outputs"][rank]], capture_output=True,
                             text=True, timeout=config["timeout"])
        code, stdout, stderr = run.returncode, run.stdout, run.stderr
    else:
        code, stdout, stderr = 0, "", ""
    record = dict(rank=rank, size=size, hostname=socket.gethostname(),
                  seconds=time.perf_counter()-start, returncode=code,
                  stdout=stdout, stderr=stderr, jobs=config["counts"][rank])
    (folder / ("rank%d.json" % rank)).write_text(json.dumps(record, indent=2))
    if code:
        raise RuntimeError("Native local rank %d failed: %s" % (rank, stderr))


if __name__ == "__main__":
    main()
