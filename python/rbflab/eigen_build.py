"""Shared, content-addressed Eigen dependency for the C++ build paths."""
from pathlib import Path
from .native_paths import source_root
import hashlib


def eigen_dependency():
    root=source_root()/'cpp/deps/usr/include/eigen3'
    if not (root/'Eigen/Core').is_file():
        raise FileNotFoundError('Eigen headers missing: extract libeigen3-dev into cpp/deps (see cpp/README.md)')
    digest=hashlib.sha256()
    for path in sorted((root/'Eigen').rglob('*')):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return root,digest.hexdigest()
