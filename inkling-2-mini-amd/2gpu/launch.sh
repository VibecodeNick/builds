#!/usr/bin/env bash
# Serve Inkling 2 Mini across two Radeon AI PRO R9700s (tensor parallel, tp 2).
# Foreground; Ctrl-C or `docker stop inkling2-2gpu` ends it.
#
#   PACK=/path/to/packs/inkling-2-mini-2gpu.rad GPU_PCI_0=0000:xx:00.0 GPU_PCI_1=0000:yy:00.0 ./launch.sh [serve.yaml]
#
# PACK                 the pack from `convert.sh 2gpu`
# GPU_PCI_0, GPU_PCI_1 PCI addresses of the two cards (lspci -D | grep -i radeon)
# RUNTIME              output of build.sh (default ../runtime)
# NAME                 container name (default inkling2-2gpu); MEMORY host-memory limit (default 24g)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
config=$(readlink -e "${1:-$here/serve.yaml}")
RUNTIME=$(readlink -e "${RUNTIME:-$here/../runtime}")
PACK=$(readlink -e "${PACK:?path to inkling-2-mini-2gpu.rad}")
GPU_PCI_0=${GPU_PCI_0:?PCI address of the first card, for example 0000:xx:00.0}
GPU_PCI_1=${GPU_PCI_1:?PCI address of the second card, for example 0000:yy:00.0}
IMAGE=${IMAGE:-rocm/dev-ubuntu-24.04:10.1.0-full}
test -x "$RUNTIME/bin/radiance"
grep -Eq '^[[:space:]]+tp: 2$' "$config"
r0=$(readlink -e "/dev/dri/by-path/pci-$GPU_PCI_0-render")
r1=$(readlink -e "/dev/dri/by-path/pci-$GPU_PCI_1-render")
test "$r0" != "$r1"
exec docker run --rm --name "${NAME:-inkling2-2gpu}" --memory "${MEMORY:-24g}" --network host \
    --device /dev/kfd --device "$r0" --device "$r1" \
    -u "$(id -u):$(id -g)" --group-add "$(stat -c '%g' "$r0")" --group-add "$(stat -c '%g' /dev/kfd)" \
    --ulimit memlock=-1:-1 -e ROCR_VISIBLE_DEVICES=0,1 -e RADIANCE_HOME=/runtime/radiance_home \
    -e LD_LIBRARY_PATH=/runtime/lib:/opt/rocm/lib -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1 \
    -e R4D_INKLING2_WMMA_PREFILL=1 -e R4D_INKLING2_DECODE=grouped \
    -v "$RUNTIME:/runtime:ro" -v "$PACK:/model/inkling-2-mini.rad:ro" -v "$config:/config/serve.yaml:ro" \
    "$IMAGE" /runtime/bin/radiance --config /config/serve.yaml
