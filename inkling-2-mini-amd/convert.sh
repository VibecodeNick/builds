#!/usr/bin/env bash
# Turn the Hugging Face checkpoint into the pack the runtime serves.
#
#   CHECKPOINT=/path/to/Inkling-2-Mini PACKS=/path/to/packs GPU_PCI=0000:xx:00.0 ./convert.sh 1gpu
#   CHECKPOINT=/path/to/Inkling-2-Mini PACKS=/path/to/packs GPU_PCI=0000:xx:00.0 ./convert.sh 2gpu
#
# CHECKPOINT  local download of thinkingmachines/Inkling-2-Mini (about 20 GB)
# PACKS       output directory. Needs about 75 GB for the 1gpu pack and its intermediate files
#             (PACKS/prepared, safe to delete afterwards) and 19 GB more for the 2gpu pack
# GPU_PCI     PCI address of one R9700; the converter checks the kernels the pack will be served with
# RUNTIME     output of build.sh (default ./runtime)
#
# 1gpu  writes PACKS/inkling-2-mini-1gpu.rad: the query and gate projections of each attention
#       layer stored as one weight, which is what the one-card build reads.
# 2gpu  copies the 1gpu pack to PACKS/inkling-2-mini-2gpu.rad and rewrites those 24 weights as the
#       48 separate ones two cards split by head. Takes seconds. Run 1gpu first.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
layout=${1:?1gpu or 2gpu}
RUNTIME=${RUNTIME:-$here/runtime}
PACKS=${PACKS:?output directory for packs}
GPU_PCI=${GPU_PCI:?PCI address of one card, for example 0000:xx:00.0}
IMAGE=${IMAGE:-rocm/dev-ubuntu-24.04:10.1.0-full}
test -x "$RUNTIME/bin/rad-convert"
mkdir -p "$PACKS"
render=$(readlink -e "/dev/dri/by-path/pci-$GPU_PCI-render")
one=$PACKS/inkling-2-mini-1gpu.rad
two=$PACKS/inkling-2-mini-2gpu.rad

convert() {   # extra rad-convert arguments
    docker run --rm --memory 24g --network none --device /dev/kfd --device "$render" \
        -u "$(id -u):$(id -g)" --group-add "$(stat -c '%g' "$render")" --group-add "$(stat -c '%g' /dev/kfd)" \
        --ulimit memlock=-1:-1 -e ROCR_VISIBLE_DEVICES=0 -e RADIANCE_HOME=/runtime/radiance_home \
        -e RADIANCE_HOST_VRAM_MIB=20000 -e LD_LIBRARY_PATH=/runtime/lib:/opt/rocm/lib \
        -e OMP_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=1 \
        -v "$RUNTIME:/runtime:ro" -v "$PACKS:/packs" "$IMAGE" \
        /runtime/bin/rad-convert /packs/prepared/mm --arch inkling2 --draft-model /packs/prepared/draft --max-spec 7 \
        --recipe /runtime/share/inkling2-bf16.recipe --recipe /runtime/share/inkling2-experts-i4.recipe \
        --kernels spechub:libref:libr4d --max-tok 32 --max-ctx 16384 \
        --chat-template /packs/prepared/mm/chat_template.jinja "$@"
}

case "$layout" in
1gpu)
    CHECKPOINT=${CHECKPOINT:?path to the downloaded checkpoint}
    test -f "$CHECKPOINT/config.json"
    if [ -e "$one" ]; then echo "$one exists" >&2; exit 1; fi
    if [ ! -f "$PACKS/prepared/mm/config.json" ]; then
        # CPU only: expand the checkpoint's packed 4-bit experts to BF16, split out the draft head
        # and the vision tower. torch is used to read and write tensors, nothing runs a model.
        docker run --rm --memory 24g -u "$(id -u):$(id -g)" -e HOME=/tmp \
            -v "$RUNTIME:/runtime:ro" -v "$CHECKPOINT:/checkpoint:ro" -v "$PACKS:/packs" "$IMAGE" bash -sec '
            set -euo pipefail
            mkdir -p /packs/prepared
            python3 -m pip install --quiet --no-cache-dir --target /packs/prepared/py \
                --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
            python3 -m pip install --quiet --no-cache-dir --target /packs/prepared/py safetensors==0.8.0 numpy==2.2.6
            export PYTHONPATH=/packs/prepared/py
            python3 /runtime/share/inkling2_prepare.py /checkpoint /packs/prepared/text-bf16
            python3 /runtime/share/inkling2_prepare_draft.py /checkpoint /packs/prepared/draft
            python3 /runtime/share/inkling2_prepare_vision.py /checkpoint /packs/prepared/vision-bf16
            python3 /runtime/share/inkling2_prepare_vision.py /packs/prepared/vision-bf16 /packs/prepared/mm --text /packs/prepared/text-bf16'
    fi
    convert -o /packs/inkling-2-mini-1gpu.rad
    sha256sum "$one"
    ;;
2gpu)
    test -f "$one" || { echo "convert the 1gpu pack first" >&2; exit 1; }
    if [ -e "$two" ]; then echo "$two exists" >&2; exit 1; fi
    cp --reflink=auto "$one" "$two"
    convert --set radiance.inkling2.qgate=split --reuse /packs/inkling-2-mini-2gpu.rad --in-place
    rm -f "$two.pre-append"
    sha256sum "$two"
    ;;
*) echo "usage: $0 1gpu|2gpu" >&2; exit 2 ;;
esac
