"""Optional dependency resolution does not require a native compiler."""
import json
import os
import pytest
from rbflab.mplapack_build import mplapack_dependency


def _prefix(tmp_path):
    include = tmp_path / 'include'
    public = include / 'mplapack'
    public.mkdir(parents=True)
    for name in ('mpblas_mpfr.h', 'mplapack_mpfr.h', 'mplapack_config.h', 'mpfrxx_mkII.h', 'mpcxx_mkII.h', 'mplapack_mpfr_precision.h'):
        (public / name).write_text('// dummy header')
    config = public / 'gmpfrxx_mkII/detail/config.hpp'
    config.parent.mkdir(parents=True)
    config.write_text('#define GMPFRXX_MKII_VERSION "1.4.0"')
    for name in ('mpfr.h', 'gmp.h', 'mpc.h'):
        (include / name).write_text('// dummy dependency header')
    lib = tmp_path / 'lib'
    lib.mkdir()
    suffix = '.a' if os.name == 'nt' else '.so'
    for name in ('libmplapack_mpfr_opt.a', 'libmpc'+suffix, 'libmpfr'+suffix, 'libgmp'+suffix):
        (lib / name).write_bytes(b'dummy library')
    (tmp_path / 'rbflab-mplapack.json').write_text(json.dumps({'mplapack_version': '3.0.1', 'gmpfrxx_version': '1.4.0'}))
    return tmp_path


def test_explicit_prefix_required():
    with pytest.raises(FileNotFoundError, match='mplapack_dir'):
        mplapack_dependency(None)


def test_missing_dependency_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match='build_mplapack'):
        mplapack_dependency(tmp_path)


def test_wrong_platform_rejected(tmp_path):
    prefix = _prefix(tmp_path)
    manifest = prefix / 'rbflab-mplapack.json'
    metadata = json.loads(manifest.read_text())
    metadata['platform'] = 'posix' if os.name == 'nt' else 'nt'
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='platform'):
        mplapack_dependency(prefix)


def test_prefix_version_rejected(tmp_path):
    prefix = _prefix(tmp_path)
    (prefix / 'rbflab-mplapack.json').write_text(json.dumps({'mplapack_version': '3.0.0', 'gmpfrxx_version': '1.4.0'}))
    with pytest.raises(ValueError, match='3.0.1'):
        mplapack_dependency(prefix)


@pytest.mark.parametrize('relative', ['include/gmp.h', 'lib/libmplapack_mpfr_opt.a', 'lib/libmpfr'+('.a' if os.name == 'nt' else '.so'), 'rbflab-mplapack.json'])
def test_content_changes_identity(tmp_path, relative):
    prefix = _prefix(tmp_path)
    before = mplapack_dependency(prefix)
    path = prefix / relative
    with path.open('ab') as stream:
        stream.write(b'\n')
    after = mplapack_dependency(prefix)
    assert before.identity['sha256'] != after.identity['sha256']
    assert path in after.files
    assert '-DRBFLAB_MPLAPACK=1' in after.flags
