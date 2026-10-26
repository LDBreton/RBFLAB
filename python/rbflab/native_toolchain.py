"""Native host C++ compiler and executable paths (never invokes WSL)."""
from pathlib import Path
import os
import hashlib
import shutil
import subprocess
from .native_paths import source_root


def native_path(path):
    return str(Path(path).resolve())


def executable_path(path):
    path = Path(path)
    return path.with_name(path.name + '.exe') if os.name == 'nt' and path.suffix != '.exe' else path


def compiler():
    configured = os.environ.get('RBFLAB_CXX')
    if configured:
        found = shutil.which(configured)
        if not found:
            raise FileNotFoundError('RBFLAB_CXX compiler does not exist: ' + configured)
        return found
    candidates = ([
        str(source_root() / 'cpp/deps/windows/ucrt64/bin/g++.exe'),
        'g++', 'C:/msys64/ucrt64/bin/g++.exe', 'C:/msys64/mingw64/bin/g++.exe']
        if os.name == 'nt' else ['g++'])
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError('Native g++ not found. On Windows run python scripts/setup_windows_cpp.py '
                            'or set RBFLAB_CXX to your MinGW-w64 g++.exe.')


def compiler_identity(cxx=None):
    cxx = cxx or compiler()
    version = subprocess.check_output([cxx, '--version'], text=True).splitlines()[0]
    target = subprocess.check_output([cxx, '-dumpmachine'], text=True).strip()
    if os.name == 'nt' and 'mingw' not in target:
        raise RuntimeError('Windows backend requires a native MinGW-w64 compiler: ' + target)
    return cxx + ' | ' + target + ' | ' + version


def arithmetic_flags(arithmetic):
    if arithmetic == 'float64':
        flags = ['-DRBFLAB_DOUBLE']
    elif os.name == 'nt':
        # Match MPFR/GMP headers and libraries to the selected compiler prefix.
        prefix = Path(os.environ.get('RBFLAB_NATIVE_PREFIX', Path(compiler()).parent.parent))
        if not all((prefix / 'include' / name).is_file() for name in ('mpfr.h', 'gmp.h')):
            raise FileNotFoundError('Native MPFR/GMP headers missing from ' + str(prefix))
        flags = ['-I' + str(prefix / 'include'), '-L' + str(prefix / 'lib'), '-lmpfr', '-lgmp']
    else:
        include = source_root() / 'cpp/deps/usr/include'
        flags = ['-I' + str(include), '-I' + str(include / 'x86_64-linux-gnu'),
                 '-Wl,-l:libmpfr.so.6', '-Wl,-l:libgmp.so.10']
    # Self-contained executables also work from Python without editing PATH.
    return flags + (['-static'] if os.name == 'nt' else [])


def dependency_identity(arithmetic):
    """Track native MPFR/GMP inputs as well as the compiler target/version."""
    if os.name != 'nt' or arithmetic != 'mpfr':
        return {}
    prefix = Path(os.environ.get('RBFLAB_NATIVE_PREFIX', Path(compiler()).parent.parent))
    paths = [prefix / 'include' / name for name in ('mpfr.h', 'gmp.h')]
    paths += [prefix / 'lib' / name for name in ('libmpfr.a', 'libgmp.a')]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def compiler_environment():
    env = os.environ.copy()
    if os.name == 'nt':
        env['PATH'] = str(Path(compiler()).parent) + os.pathsep + env.get('PATH', '')
    return env
