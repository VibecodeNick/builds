#!/usr/bin/env bash
# Build the patched Radiance runtime for gfx1201 (Radeon AI PRO R9700).
# Everything compiles inside the stock ROCm container; the host needs git and docker only.
#
#   ./build.sh            -> ./runtime (bin/, radiance_home/, lib/, share/)
#
# WORK    scratch directory for the source checkout and the build tree (default ./work)
# OUT     where the finished runtime goes (default ./runtime, must not exist)
# JOBS    parallel compile jobs (default 6; the HIP kernels are memory hungry)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
COMMIT=689f554596d668c24efcf60f0ea74e47c7b01338
REPO=https://codeberg.org/StillDeadcode/radiance.git
PATCH=$here/patches/radiance-inkling2-gfx1201.patch
IMAGE=${IMAGE:-rocm/dev-ubuntu-24.04:10.1.0-full}
WORK=${WORK:-$here/work}
OUT=${OUT:-$here/runtime}
JOBS=${JOBS:-6}

if [ -e "$OUT" ]; then echo "$OUT exists; move it away first" >&2; exit 1; fi
mkdir -p "$WORK"
if [ ! -d "$WORK/radiance/.git" ]; then
    git clone --no-checkout "$REPO" "$WORK/radiance"
fi
git -C "$WORK/radiance" checkout --quiet --detach --force "$COMMIT"
git -C "$WORK/radiance" clean -qfdx
git -C "$WORK/radiance" apply --check "$PATCH"
git -C "$WORK/radiance" apply "$PATCH"
rm -rf "$WORK/build" "$WORK/stage"

docker run -i --rm --memory "${BUILD_MEMORY:-24g}" \
    -e BUILD_UID="$(id -u)" -e BUILD_GID="$(id -g)" -e JOBS="$JOBS" \
    -v "$WORK:/work" -w /work "$IMAGE" bash -se <<'IN'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# Remember what the stock image ships, so only the libraries this build adds are staged.
find /usr/lib/x86_64-linux-gnu -name '*.so*' -printf '%f\n' | sort -u > /tmp/libs.stock
apt-get -qq update
apt-get -qq install -y --no-install-recommends cmake ninja-build g++-14 pkg-config gawk libopenblas-dev \
    libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libswresample-dev
exec setpriv --reuid="$BUILD_UID" --regid="$BUILD_GID" --clear-groups \
    env HOME=/tmp CC=gcc-14 CXX=g++-14 ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm JOBS="$JOBS" bash -se <<'BUILD'
set -euo pipefail
cmake -S radiance -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DRAD_WITH_FFMPEG=ON -DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201
cmake --build build -j"$JOBS" --target radiance rad-convert rad-kbench inkling2 libref libavx libquant spechub libr4d
# The model plugin, built with ROCm clang at -O2, the same way as the measured build.
/opt/rocm/llvm/bin/clang++ -std=c++20 -O2 -fPIC -shared -I radiance/abi -I build/rad_include \
    -I radiance/arch/inkling2 radiance/arch/inkling2/inkling2.cpp -o build/inkling2.so

mkdir -p stage/bin stage/lib stage/share
cp -a build/radiance_home stage/
cp -a build/bin/radiance build/bin/rad-convert build/bin/rad-kbench stage/bin/
cp build/inkling2.so stage/radiance_home/architectures/inkling2.so
cp radiance/data/recipes/inkling2-bf16.recipe radiance/data/recipes/inkling2-experts-i4.recipe \
   radiance/tools/inkling2_prepare.py radiance/tools/inkling2_prepare_draft.py \
   radiance/tools/inkling2_prepare_vision.py radiance/libr4d/r4d_tanh_ff.h radiance/LICENSE radiance/NOTICE stage/share/
# Shared libraries the build pulled in that the stock image does not have (OpenBLAS, FFmpeg and theirs).
for f in stage/bin/* stage/radiance_home/*/*.so; do ldd "$f" 2>/dev/null || true; done |
    awk '$2 == "=>" && $3 ~ /^\// {print $3}' | sort -u | while read -r lib; do
        case "$lib" in /opt/rocm/*) continue ;; esac
        grep -qxF "$(basename "$lib")" /tmp/libs.stock || cp -L "$lib" stage/lib/
    done
BUILD
IN

mv "$WORK/stage" "$OUT"
( cd "$OUT" && sha256sum bin/radiance bin/rad-convert radiance_home/architectures/inkling2.so radiance_home/kernels/libr4d.so )
echo "runtime ready: $OUT"
