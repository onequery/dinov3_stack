#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$ROOT_DIR"

DEVICES_STR="${DEVICES_STR:-philips_integris_h alluraxper integris_allura_flat_detector}"
read -r -a DEVICES <<< "${DEVICES_STR}"

echo "[run_global_5_per_device_patient_retrieval_all] ROOT_DIR=${ROOT_DIR}"
echo "[run_global_5_per_device_patient_retrieval_all] DEVICES=${DEVICES_STR}"

for DEVICE in "${DEVICES[@]}"; do
  case "${DEVICE}" in
    philips_integris_h)
      bash scripts/analysis/global_analysis/run_global_5_single_device_patient_retrieval.sh
      ;;
    alluraxper)
      bash scripts/analysis/global_analysis/run_global_5_1_single_device_patient_retrieval.sh
      ;;
    integris_allura_flat_detector)
      bash scripts/analysis/global_analysis/run_global_5_2_single_device_patient_retrieval.sh
      ;;
    *)
      echo "Unknown device key: ${DEVICE}" >&2
      exit 1
      ;;
  esac
 done

bash scripts/analysis/global_analysis/run_global_5_per_device_patient_retrieval_summary.sh
