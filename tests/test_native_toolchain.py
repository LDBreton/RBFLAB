"""Native host selection and Windows binary isolation without running GCC."""
from pathlib import Path
from types import SimpleNamespace
import os
import pytest
from rbflab import native_toolchain as native


def test_windows_executable_is_distinct_from_linux_cache(monkeypatch):
    path = Path('cache/kernel')
    monkeypatch.setattr(native, 'os', SimpleNamespace(name='nt'))
    assert native.executable_path(path) == path.with_name('kernel.exe')
    assert native.executable_path(path.with_name('kernel.exe')) == path.with_name('kernel.exe')


def test_compiler_override_and_target_validation(monkeypatch):
    monkeypatch.setenv('RBFLAB_CXX', 'custom-g++')
    monkeypatch.setattr(native.shutil, 'which', lambda name: name if name == 'custom-g++' else None)
    assert native.compiler() == 'custom-g++'
    monkeypatch.setattr(native, 'os', SimpleNamespace(name='nt', environ=os.environ))
    monkeypatch.setattr(native.subprocess, 'check_output',
                        lambda args, **kwargs: 'x86_64-linux-gnu' if '-dumpmachine' in args else 'GCC version')
    with pytest.raises(RuntimeError, match='MinGW'):
        native.compiler_identity()


def test_windows_mpfr_uses_matching_static_libraries(tmp_path, monkeypatch):
    include = tmp_path / 'include'
    include.mkdir()
    for name in ('mpfr.h', 'gmp.h'):
        (include / name).write_text('header')
    monkeypatch.setenv('RBFLAB_NATIVE_PREFIX', str(tmp_path))
    monkeypatch.setattr(native, 'compiler', lambda: str(tmp_path / 'bin/g++.exe'))
    monkeypatch.setattr(native, 'os', SimpleNamespace(name='nt', environ=os.environ))
    flags = native.arithmetic_flags('mpfr')
    assert '-static' in flags and '-lmpfr' in flags and '-lgmp' in flags
    assert '-I' + str(include) in flags
    assert not any('.so' in flag or '/mnt/' in flag for flag in flags)
