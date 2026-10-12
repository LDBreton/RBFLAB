"""Warm global artifact memo invalidation without requiring a C++ compiler."""
import json
from pathlib import Path

import pytest
import rbflab as r
import rbflab.global_backend as global_backend
import rbflab.kernel_compiler as compiler


@pytest.fixture
def cached_artifact(tmp_path, monkeypatch):
    root = tmp_path/'source'
    cpp = root/'cpp'
    eigen = cpp/'deps/usr/include/eigen3/Eigen'
    eigen.mkdir(parents=True)
    (eigen/'Core').write_text('eigen')
    for name in ('global_collocation.hpp', 'real.hpp', 'real_double.hpp',
                 'lhi_mpfr.cpp', 'eigen_solver.hpp'):
        (cpp/name).write_text(name)
    directory = tmp_path/'cache'/'artifact'
    directory.mkdir(parents=True)
    for name in ('kernel', 'kernel.cpp', 'real.hpp'):
        (directory/name).write_text(name)
    (directory/'manifest.json').write_text(json.dumps({'files': {
        name: 'validated by request_cache' for name in ('kernel', 'kernel.cpp', 'real.hpp')}}))
    index = directory.parent/'request.json'
    index.write_text('{}')
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return index, directory

    monkeypatch.setattr(compiler, 'ROOT', root)
    monkeypatch.setattr(compiler, 'request_cache', request)
    global_backend._compiled_memo.clear()
    yield root, directory, index, calls
    global_backend._compiled_memo.clear()


def compile_cached(cache, kernel=None, dimension=2, order=2, arithmetic='float64'):
    return global_backend._compiled(kernel or r.IMQ(), dimension, order, arithmetic,
                                   r.CppGlobalBackend(cache_dir=cache))


def test_warm_lookup_reuses_validation_and_current_parameters(cached_artifact):
    _, directory, _, calls = cached_artifact
    first, bound1 = compile_cached(directory.parent, r.IMQ(c=1))
    second, bound2 = compile_cached(directory.parent, r.IMQ(c=2))
    assert first == second == directory/'kernel'
    assert bound1.values != bound2.values
    assert len(calls) == 1


@pytest.mark.parametrize('relative', [
    'cpp/global_collocation.hpp', 'cpp/real.hpp', 'cpp/real_double.hpp',
    'cpp/lhi_mpfr.cpp', 'cpp/eigen_solver.hpp',
    'cpp/deps/usr/include/eigen3/Eigen/Core',
])
def test_dependency_edits_revalidate(cached_artifact, relative):
    root, directory, _, calls = cached_artifact
    compile_cached(directory.parent)
    path = root/relative
    path.write_text(path.read_text()+' edited')
    compile_cached(directory.parent)
    assert len(calls) == 2


@pytest.mark.parametrize('filename', ['kernel', 'kernel.cpp', 'real.hpp', 'manifest.json'])
@pytest.mark.parametrize('delete', [False, True])
def test_artifact_changes_trigger_full_validation(cached_artifact, monkeypatch, filename, delete):
    _, directory, _, calls = cached_artifact
    compile_cached(directory.parent)
    path = directory/filename
    if delete:
        path.unlink()
    else:
        path.write_text(path.read_text()+' edited')

    def rejected(*args, **kwargs):
        raise RuntimeError('Compiled cache failed integrity validation')

    monkeypatch.setattr(compiler, 'request_cache', rejected)
    with pytest.raises(RuntimeError, match='integrity'):
        compile_cached(directory.parent)


def test_new_eigen_header_and_index_change_revalidate(cached_artifact):
    root, directory, index, calls = cached_artifact
    compile_cached(directory.parent)
    (root/'cpp/deps/usr/include/eigen3/Eigen/NewHeader').write_text('new')
    compile_cached(directory.parent)
    index.write_text('{"changed": true}')
    compile_cached(directory.parent)
    assert len(calls) == 3


@pytest.mark.parametrize('kwargs', [
    {'kernel': r.Gaussian()}, {'dimension': 3}, {'order': 4}, {'arithmetic': 'mpfr'},
])
def test_request_configuration_does_not_share_memo(cached_artifact, kwargs):
    _, directory, _, calls = cached_artifact
    compile_cached(directory.parent)
    compile_cached(directory.parent, **kwargs)
    compile_cached(directory.parent/'other')
    assert len(calls) == 3

def test_deleted_eigen_header_revalidates(cached_artifact, monkeypatch):
    root, directory, _, _ = cached_artifact
    compile_cached(directory.parent)
    (root/'cpp/deps/usr/include/eigen3/Eigen/Core').unlink()

    def missing(*args, **kwargs):
        raise FileNotFoundError('Eigen headers missing')

    monkeypatch.setattr(compiler, 'request_cache', missing)
    with pytest.raises(FileNotFoundError, match='Eigen'):
        compile_cached(directory.parent)


@pytest.mark.parametrize('module', [global_backend, compiler])
def test_python_dependency_change_revalidates(cached_artifact, monkeypatch, module):
    _, directory, _, calls = cached_artifact
    compile_cached(directory.parent)
    original = global_backend._file_signature
    changed = Path(module.__file__)

    def signature(path):
        value = original(path)
        return (*value, 'edited') if path == changed else value

    monkeypatch.setattr(global_backend, '_file_signature', signature)
    compile_cached(directory.parent)
    assert len(calls) == 2
