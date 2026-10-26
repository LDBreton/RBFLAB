"""Build an isolated native Windows MPLAPACK MPFR/OpenMP static prefix.

Run explicitly from a native-enabled checkout. Requires CMake, Ninja and the
MinGW-w64 toolchain prepared by setup_windows_cpp.py. No WSL or admin rights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

from rbflab.native_toolchain import compiler, compiler_environment, compiler_identity

URL = 'https://github.com/nakatamaho/mplapack/archive/refs/tags/v3.0.1.tar.gz'
SHA256 = '60b8f35bc08b74ea802f6a009c54dc97e7adb6397cd004fb0364d1048977bb94'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path)
    parser.add_argument('--deps', type=Path)
    parser.add_argument('--archive', type=Path, help='Previously downloaded official source archive')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('Use scripts/build_mplapack.sh on Linux')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    cxx = Path(compiler()).resolve()
    deps = (args.deps or cxx.parent.parent).resolve()
    prefix = args.prefix.resolve()
    work = prefix.with_name(prefix.name + '.build')
    work.mkdir(parents=True, exist_ok=True)
    cmake = shutil.which('cmake')
    ninja = shutil.which('ninja') or str(Path(sys.executable).parent/'ninja.exe')
    if not cmake or not Path(ninja).is_file():
        raise FileNotFoundError('Install CMake and Ninja in this environment: python -m pip install cmake ninja')
    dependency_flags = []
    for name in ('gmp', 'mpfr', 'mpc'):
        header, library = deps/'include'/f'{name}.h', deps/'lib'/f'lib{name}.a'
        if not header.is_file() or not library.is_file():
            raise FileNotFoundError(f'Missing native static {name} dependency under {deps}')
        dependency_flags += [f'-D{name.upper()}_INCLUDE_DIR={header.parent.as_posix()}',
                             f'-D{name.upper()}_LIBRARY={library.as_posix()}']
    archive = args.archive.resolve() if args.archive else work/'mplapack-v3.0.1.tar.gz'
    if not archive.is_file():
        print('Downloading pinned MPLAPACK source', flush=True)
        temporary = archive.with_suffix('.download')
        urllib.request.urlretrieve(URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('MPLAPACK archive checksum mismatch')
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError('MPLAPACK archive checksum mismatch')
    source = work/('source-'+SHA256)
    marker = source/'.rbflab-source-sha256'
    if not marker.is_file():
        if source.exists():
            raise RuntimeError('Unverified source directory exists; choose a fresh prefix: '+str(source))
        source.mkdir()
        subprocess.run(['tar','-xf',str(archive),'-C',str(source),'--strip-components=1'],check=True)
        marker.write_text(SHA256)
    if marker.read_text() != SHA256:
        raise RuntimeError('Source archive identity mismatch')
    build = work/'build'
    env = compiler_environment()
    options = [f'-DCMAKE_CXX_COMPILER={cxx.as_posix()}',
        f'-DCMAKE_C_COMPILER={cxx.with_name("gcc.exe").as_posix()}',
        f'-DCMAKE_MAKE_PROGRAM={Path(ninja).as_posix()}', '-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_SHARED_LIBS=OFF', f'-DCMAKE_INSTALL_PREFIX={prefix.as_posix()}',
        '-DCMAKE_INSTALL_LIBDIR=lib','-DMPLAPACK_ENABLE_MPFR=ON','-DMPLAPACK_ENABLE_OPT=ON']
    options += [f'-DMPLAPACK_ENABLE_{name}=OFF' for name in
                ('GMP','QD','DD','DOUBLE','BINARY128','BINARY80')]
    options += [f'-DMPLAPACK_BUILD_{name}=OFF' for name in ('TESTS','EXAMPLES','BENCHMARKS')]
    subprocess.run([cmake,'-S',str(source),'-B',str(build),'-G','Ninja',
                    *options,*dependency_flags],env=env,check=True)
    cache = (build/'CMakeCache.txt').read_text()
    if 'OpenMP_CXX_FLAGS:STRING=-fopenmp' not in cache:
        raise RuntimeError('MPLAPACK optimized MPFR requires detected OpenMP')
    subprocess.run([cmake,'--build',str(build),'--target','mplapack_mpfr_opt',
                    '--parallel',str(args.jobs)],env=env,check=True)
    # Stage only the selected target and its complete public header tree.
    include = prefix/'include'
    public = include/'mplapack'
    lib = prefix/'lib'
    lib.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source/'include',public,dirs_exist_ok=True)
    shutil.copy2(build/'include/mplapack_config.h',public)
    shutil.copytree(build/'external/i/GMPFRXX_MKII/include',public,dirs_exist_ok=True)
    shutil.copy2(build/'libmplapack_mpfr_opt.a',lib)
    for name in ('gmp','mpfr','mpc'):
        shutil.copy2(deps/'include'/f'{name}.h',include)
        shutil.copy2(deps/'lib'/f'lib{name}.a',lib)
    licenses = prefix/'licenses'
    licenses.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and ('license' in path.name.lower() or 'copying' in path.name.lower()):
            shutil.copy2(path,licenses)
    for name in ('gmp','mpfr','mpc'):
        for notice in (deps/'share/licenses'/name, deps/'share/doc'/name):
            if notice.is_dir():
                shutil.copytree(notice,licenses/name,dirs_exist_ok=True)
    wrapper=build/'external/i/GMPFRXX_MKII'
    for path in wrapper.iterdir():
        if path.is_file() and ('license' in path.name.lower() or 'copying' in path.name.lower()):
            target=licenses/'gmpfrxx_mkII';target.mkdir(exist_ok=True);shutil.copy2(path,target)
    metadata = dict(mplapack_version='3.0.1',gmpfrxx_version='1.4.0',platform='nt',
        compiler=compiler_identity(),source_url=URL,source_sha256=SHA256,
        dependency_prefix=str(deps),build='Release; native Windows; static MPFR optimized; OpenMP',
        library_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in lib.glob('*.a')})
    (prefix/'rbflab-mplapack.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Native Windows MPLAPACK prefix prepared:',prefix,flush=True)


if __name__ == '__main__':
    main()
