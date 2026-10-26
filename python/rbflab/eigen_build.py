"""Shared, content-addressed Eigen dependency for the C++ build paths."""
from pathlib import Path
from .native_paths import source_root
import hashlib
import os


def eigen_dependency():
    root=source_root()/'cpp/deps/usr/include/eigen3'
    if not (root/'Eigen/Core').is_file() and os.name == 'nt':
        from .native_toolchain import compiler
        root=Path(compiler()).parent.parent/'include/eigen3'
    if not (root/'Eigen/Core').is_file():
        raise FileNotFoundError('Eigen headers missing: run scripts/setup_windows_cpp.py on Windows or cpp/prepare_dependencies.sh on Linux (see cpp/README.md)')
    digest=hashlib.sha256()
    for path in sorted((root/'Eigen').rglob('*')):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return root,digest.hexdigest()
