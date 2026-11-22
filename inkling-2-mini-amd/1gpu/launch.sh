#!/usr/bin/env bash
# Serve Inkling 2 Mini on one Radeon AI PRO R9700. Foreground; Ctrl-C or `docker stop inkling2-1gpu` ends it.
#
#   PACK=/path/to/packs/inkling-2-mini-1gpu.rad GPU_PCI=0000:xx:00.0 ./launch.sh [serve.yaml]
#
# PACK     the pack from `convert.sh 1gpu`
# GPU_PCI  PCI address of the card (lspci -D | grep -i radeon)
# RUNTIME  output of build.sh (default ../runtime)
# NAME     container name (default inkling2-1gpu); MEMORY host-memory limit (default 24g)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
config=$(readlink -e "${1:-$here/serve.yaml}")
RUNTIME=$(readlink -e "${RUNTIME:-$here/../runtime}")
PACK=$(readlink -e "${PACK:?path to inkling-2-mini-1gpu.rad}")
GPU_PCI=${GPU_PCI:?PCI address of the card, for example 0000:xx:00.0}
IMAGE=${IMAGE:-rocm/dev-ubuntu-24.04:10.1.0-full}
test -x "$RUNTIME/bin/radiance"
grep -Eq '^[[:space:]]+tp: 1$' "$config"
render=$(readlink -e "/dev/dri/by-path/pci-$GPU_PCI-render")
exec docker run --rm --name "${NAME:-inkling2-1gpu}" --memory "${MEMORY:-24g}" --network host \
    --device /dev/kfd --device "$render" \
    -u "$(id -u):$(id -g)" --group-add "$(stat -c '%g' "$render")" --group-add "$(stat -c '%g' /dev/kfd)" \
    --ulimit memlock=-1:-1 -e ROCR_VISIBLE_DEVICES=0 -e RADIANCE_HOME=/runtime/radiance_home \
    -e LD_LIBRARY_PATH=/runtime/lib:/opt/rocm/lib -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1 \
    -e R4D_INKLING2_WMMA_PREFILL=1 -e R4D_INKLING2_DECODE=grouped \
    -v "$RUNTIME:/runtime:ro" -v "$PACK:/model/inkling-2-mini.rad:ro" -v "$config:/config/serve.yaml:ro" \
    "$IMAGE" /runtime/bin/radiance --config /config/serve.yaml
