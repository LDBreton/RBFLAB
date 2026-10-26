#!/usr/bin/env bash
# Optional isolated dependency build. No sudo, apt install, or Python prepare hook.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 PREFIX [DEPENDENCIES_PREFIX] [JOBS]" >&2
  echo "Provide GMP/MPFR/MPC development headers and libraries beforehand." >&2
  exit 2
fi
prefix=$(realpath -m "$1")
deps=$(realpath -m "${2:-/usr}")
jobs=${3:-2}
work="${prefix}.build"
mkdir -p "$work"
# Resolve exact inputs before configuring, so CMake and staging agree.
declare -A headers libraries
dependency_flags=()
for component in gmp mpfr mpc; do
  headers[$component]=$(find "$deps/include" -name "$component.h" -print | sort | head -n 1)
  libraries[$component]=$(find "$deps/lib" -name "lib${component}.so" -print | sort | head -n 1)
  [[ -n "${headers[$component]}" && -f "${headers[$component]}" && -n "${libraries[$component]}" && -f "${libraries[$component]}" ]] || {
    echo "Missing $component development header/library beneath $deps" >&2; exit 1;
  }
  upper=${component^^}
  dependency_flags+=("-D${upper}_INCLUDE_DIR=$(dirname "${headers[$component]}")" "-D${upper}_LIBRARY=${libraries[$component]}")
done
# The official release archive is pinned by tag and content.
url=https://github.com/nakatamaho/mplapack/archive/refs/tags/v3.0.1.tar.gz
archive="$work/mplapack-v3.0.1.tar.gz"
expected=60b8f35bc08b74ea802f6a009c54dc97e7adb6397cd004fb0364d1048977bb94
if [[ ! -f "$archive" ]]; then
  download=$(mktemp "$work/download.XXXXXX")
  if ! curl --fail --location "$url" -o "$download"; then rm -f "$download"; exit 1; fi
  actual=$(sha256sum "$download" | cut -d ' ' -f 1)
  if [[ "$actual" != "$expected" ]]; then echo 'MPLAPACK archive checksum mismatch' >&2; rm -f "$download"; exit 1; fi
  mv "$download" "$archive"
fi
actual=$(sha256sum "$archive" | cut -d ' ' -f 1)
[[ "$actual" == "$expected" ]] || { echo 'MPLAPACK archive checksum mismatch' >&2; exit 1; }
# A source tree may be reused only with the verified archive marker.
source="$work/source-$expected"
if [[ ! -f "$source/.rbflab-source-sha256" ]] || [[ "$(cat "$source/.rbflab-source-sha256")" != "$expected" ]]; then
  extracted=$(mktemp -d "$work/extracted.XXXXXX")
  if ! tar -xf "$archive" -C "$extracted" --strip-components=1; then rm -r "$extracted"; exit 1; fi
  printf '%s\n' "$expected" > "$extracted/.rbflab-source-sha256"
  if [[ -e "$source" ]]; then echo "Unverified source tree exists: $source; move it aside before retrying." >&2; rm -r "$extracted"; exit 1; fi
  mv "$extracted" "$source"
fi
build="$work/build-$expected"
cmake -S "$source" -B "$build" "${dependency_flags[@]}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$deps" \
  -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib \
  -DMPLAPACK_ENABLE_MPFR=ON -DMPLAPACK_ENABLE_OPT=ON \
  -DMPLAPACK_ENABLE_GMP=OFF -DMPLAPACK_ENABLE_QD=OFF \
  -DMPLAPACK_ENABLE_DD=OFF -DMPLAPACK_ENABLE_DOUBLE=OFF \
  -DMPLAPACK_ENABLE_BINARY128=OFF -DMPLAPACK_ENABLE_BINARY80=OFF \
  -DMPLAPACK_BUILD_TESTS=OFF -DMPLAPACK_BUILD_EXAMPLES=OFF \
  -DMPLAPACK_BUILD_BENCHMARKS=OFF
grep -Eq '^OpenMP_CXX_FLAGS:STRING=.+$' "$build/CMakeCache.txt" && \
  grep -Eq '^OpenMP_CXX_LIB_NAMES:STRING=.+$' "$build/CMakeCache.txt" || {
    echo 'MPLAPACK optimized MPFR requires detected C++ OpenMP flags and runtime libraries.' >&2; exit 1;
  }
cmake --build "$build" --target mplapack_mpfr_opt --parallel "$jobs"
# Install only the selected archive and its complete header dependency tree.
mkdir -p "$prefix/include/mplapack" "$prefix/lib" "$prefix/licenses"
cp "$build/libmplapack_mpfr_opt.a" "$prefix/lib/"
cp -R "$source/include/." "$prefix/include/mplapack/"
cp "$build/include/mplapack_config.h" "$prefix/include/mplapack/"
cp -R "$build/external/i/GMPFRXX_MKII/include/." "$prefix/include/mplapack/"
for library in mpc mpfr gmp; do
  cp "${headers[$library]}" "$prefix/include/"
  found=${libraries[$library]}
  # Copy dereferenced SONAME files as well as the development linker name.
  cp -L "$found" "$prefix/lib/"
  for runtime in "$(dirname "$found")"/lib${library}.so.*; do
    [[ -e "$runtime" ]] && cp -L "$runtime" "$prefix/lib/"
  done
done
find "$source" -maxdepth 1 -type f \( -iname '*license*' -o -iname '*copying*' \) -exec cp {} "$prefix/licenses/" \;
# Preserve available dependency notices; separately packaged licenses may need
# to be supplied by the user before redistributing this prefix.
for component in gmp mpfr mpc gmpfrxx_mkII; do
  mkdir -p "$prefix/licenses/$component"
  for location in "$deps/share/doc/$component" "$deps/share/doc/lib${component}-dev" "$build/external/i/GMPFRXX_MKII/share/doc/$component"; do
    if [[ -d "$location" ]]; then cp -R "$location/." "$prefix/licenses/$component/"; fi
  done
done
python3 - "$archive" "$prefix" "$url" "$deps" <<'PY'
import hashlib,json,pathlib,sys
archive,prefix,url,deps=sys.argv[1:]
manifest={'mplapack_version':'3.0.1','gmpfrxx_version':'1.4.0',
          'source_url':url,'source_sha256':hashlib.sha256(pathlib.Path(archive).read_bytes()).hexdigest(),
          'dependency_prefix':deps,'build':'Release; MPFR optimized; OpenMP; all other arithmetic backends disabled'}
pathlib.Path(prefix,'rbflab-mplapack.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
echo "MPLAPACK prefix prepared: $prefix"
