"""Explicit, content-addressed optional MPLAPACK MPFR installation."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
from .native_toolchain import native_path as _native_path


@dataclass(frozen=True)
class MplapackDependency:
    flags: tuple
    identity: dict
    files: tuple


def mplapack_dependency(directory):
    """Resolve a self-contained prefix; never install or download dependencies."""
    helper = 'scripts/build_mplapack_windows.py' if os.name == 'nt' else 'scripts/build_mplapack.sh'
    if directory is None:
        raise FileNotFoundError(f"MPLAPACK requires mplapack_dir: run {helper} with an explicit prefix (see cpp/README.md).")
    prefix = Path(directory).expanduser().resolve()
    include = prefix / 'include'
    public = include / 'mplapack' if (include / 'mplapack').is_dir() else include
    manifest = prefix / 'rbflab-mplapack.json'
    required = [public / name for name in ('mpblas_mpfr.h', 'mplapack_mpfr.h', 'mplapack_config.h', 'mpfrxx_mkII.h', 'gmpfrxx_mkII/detail/config.hpp', 'mpcxx_mkII.h', 'mplapack_mpfr_precision.h')]
    required += [include / name for name in ('mpfr.h', 'gmp.h', 'mpc.h')]
    archive = prefix / 'lib/libmplapack_mpfr_opt.a'
    required += [archive, manifest]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Incomplete MPLAPACK prefix: ' + ', '.join(missing) + f'. Build an explicit prefix with {helper} (see cpp/README.md).')
    metadata = json.loads(manifest.read_text(encoding='utf-8'))
    if metadata.get('platform', os.name) != os.name:
        raise ValueError('MPLAPACK prefix platform does not match this host; build a native prefix with ' + helper)
    if metadata.get('mplapack_version') != '3.0.1' or metadata.get('gmpfrxx_version') != '1.4.0':
        raise ValueError('MPLAPACK prefix must record MPLAPACK 3.0.1 and gmpfrxx_mkII 1.4.0.')
    if '#define GMPFRXX_MKII_VERSION "1.4.0"' not in required[4].read_text(encoding='utf-8'):
        raise ValueError('MPLAPACK prefix headers must use gmpfrxx_mkII 1.4.0.')
    libraries = []
    for name in ('mpc', 'mpfr', 'gmp'):
        pattern = 'lib' + name + ('.a' if os.name == 'nt' else '.so*')
        candidates = sorted((prefix / 'lib').glob(pattern))
        candidates = [path for path in candidates if path.is_file()]
        if not candidates:
            raise FileNotFoundError(f'MPLAPACK prefix needs native {pattern} libraries in {prefix / "lib"}; Linux .so files cannot be used on Windows. See cpp/README.md.')
        libraries.append(candidates[0])
    files = tuple(sorted(set([manifest, archive, *libraries] + [path for path in include.rglob('*') if path.is_file()] + [path for path in (prefix / 'lib').glob('*.so*') if path.is_file()])))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(prefix).as_posix().encode('utf-8') + b'\0')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    flags = ('-DRBFLAB_MPLAPACK=1', '-I' + _native_path(public), '-I' + _native_path(include), _native_path(archive))
    flags += tuple(_native_path(path) for path in libraries)
    flags += ('-static', '-fopenmp') if os.name == 'nt' else ('-Wl,-rpath,' + _native_path(prefix / 'lib'), '-fopenmp')
    directories = (prefix, include, prefix / 'lib') + tuple(path for path in include.rglob('*') if path.is_dir())
    return MplapackDependency(flags, {'prefix': str(prefix), 'sha256': digest.hexdigest(), 'manifest': metadata, 'flags': list(flags)}, files + directories)
