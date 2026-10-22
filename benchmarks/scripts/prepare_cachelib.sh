#!/usr/bin/env bash
set -euo pipefail

readonly CACHELIB_COMMIT="f4012d692d849d1ae9927dd17ea186fb45f6ef00"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly WORK_ROOT="${1:-${REPOSITORY_ROOT}/../cachelib-dependency}"
readonly SOURCE_DIR="${WORK_ROOT}/source"
readonly SCRATCH_DIR="${WORK_ROOT}/scratch"
readonly INSTALL_DIR="${WORK_ROOT}/install"
readonly ENV_FILE="${WORK_ROOT}/environment.sh"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "CacheLib's supported build path is Linux. Run this script on a Linux benchmark host." >&2
  exit 1
fi

mkdir -p "${WORK_ROOT}"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone --depth 1 --filter=blob:none --no-checkout \
    https://github.com/facebook/CacheLib.git "${SOURCE_DIR}"
fi

git -C "${SOURCE_DIR}" fetch origin "${CACHELIB_COMMIT}"
git -C "${SOURCE_DIR}" checkout --detach "${CACHELIB_COMMIT}"

readonly GETDEPS="${SOURCE_DIR}/build/fbcode_builder/getdeps.py"
readonly GETDEPS_OPTIONS=(
  --scratch-path "${SCRATCH_DIR}"
  --install-prefix "${INSTALL_DIR}"
  --allow-system-packages
)

python3 "${GETDEPS}" "${GETDEPS_OPTIONS[@]}" build cachelib \
  --build-type Release --no-tests --src-dir "${SOURCE_DIR}"
python3 "${GETDEPS}" env "${GETDEPS_OPTIONS[@]}" \
  --src-dir "${SOURCE_DIR}" cachelib > "${ENV_FILE}"

echo "CacheLib ${CACHELIB_COMMIT} is ready."
echo "Environment: ${ENV_FILE}"
