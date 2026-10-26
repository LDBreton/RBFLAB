"""Install a repository-local native UCRT64 GCC/MPFR toolchain, without WSL.

Downloads official MSYS2 packages and verifies their repository SHA256 hashes.
Run explicitly with Python on Windows; no administrator rights are needed.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import urllib.request

REPOSITORY = 'https://repo.msys2.org/mingw/ucrt64/'
PREFIX = 'mingw-w64-ucrt-x86_64-'


def main():
    root = Path(__file__).resolve().parents[1] / 'cpp/deps/windows'
    root.mkdir(parents=True, exist_ok=True)
    database = urllib.request.urlopen(REPOSITORY + 'ucrt64.db').read()
    packages = {}
    # The repository database can itself be zstd-compressed.
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        db = Path(temporary) / 'repository.db'
        db.write_bytes(database)
        subprocess.run(['tar', '-xf', str(db), '-C', temporary], check=True)
        for description in Path(temporary).glob('*/desc'):
            fields = description.read_text(encoding='utf-8').split('\n\n')
            info = {f.splitlines()[0].strip('%'): f.splitlines()[1:] for f in fields if f.strip()}
            packages[info['NAME'][0]] = info
    selected = {}

    def visit(name):
        name = name.split('>=')[0].split('=')[0].split('<')[0].split('>')[0]
        if name in selected:
            return
        if name not in packages:
            raise RuntimeError('Unavailable native dependency: ' + name)
        info = selected[name] = packages[name]
        for dependency in info.get('DEPENDS', []):
            visit(dependency)

    for name in ('gcc', 'libgomp', 'mpfr', 'gmp', 'eigen3'):
        visit(PREFIX + name)
    downloads = root / 'downloads'
    downloads.mkdir(exist_ok=True)
    for name, info in selected.items():
        filename, digest = info['FILENAME'][0], info['SHA256SUM'][0]
        target = downloads / filename
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            print('Downloading', filename, flush=True)
            urllib.request.urlretrieve(REPOSITORY + filename, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Package checksum mismatch: ' + filename)
        # Windows tar (libarchive) supports the MSYS2 .tar.zst packages.
        subprocess.run(['tar', '-xf', str(target), '-C', str(root), '--exclude=.BUILDINFO',
                        '--exclude=.MTREE', '--exclude=.PKGINFO', '--exclude=.INSTALL'], check=True)
    (root / 'packages.json').write_text(json.dumps(selected, indent=2) + '\n')
    print('Native toolchain ready:', root / 'ucrt64/bin/g++.exe')


if __name__ == '__main__':
    main()
