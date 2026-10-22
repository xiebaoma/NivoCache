#!/usr/bin/env bash
set -euo pipefail

readonly CACHELIB_COMMIT="f4012d692d849d1ae9927dd17ea186fb45f6ef00"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly WORK_ROOT="${1:-${REPOSITORY_ROOT}/../cachelib-dependency}"
readonly CACHELIB_SOURCE="${WORK_ROOT}/source"
readonly CACHELIB_INSTALL="${WORK_ROOT}/install/cachelib"
readonly ENV_FILE="${WORK_ROOT}/environment.sh"
readonly BUILD_DIR="${REPOSITORY_ROOT}/build/comparison"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "The CacheLib comparison must run on Linux." >&2
  exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Run benchmarks/scripts/prepare_cachelib.sh first." >&2
  exit 1
fi
if [[ ! -d "${CACHELIB_SOURCE}/.git" ]]; then
  echo "Missing CacheLib checkout at ${CACHELIB_SOURCE}." >&2
  exit 1
fi
if [[ "$(git -C "${CACHELIB_SOURCE}" rev-parse HEAD)" != "${CACHELIB_COMMIT}" ]]; then
  echo "CacheLib is not at the pinned revision. Run benchmarks/scripts/prepare_cachelib.sh again." >&2
  exit 1
fi
if [[ -n "$(git -C "${CACHELIB_SOURCE}" status --porcelain)" ]]; then
  echo "CacheLib has local changes; refusing to record a mislabeled comparison." >&2
  exit 1
fi

set +u
source "${ENV_FILE}"
set -u
export CMAKE_PREFIX_PATH="${CACHELIB_INSTALL}:${CMAKE_PREFIX_PATH:-}"

cmake -S "${REPOSITORY_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MODULE_PATH="${CACHELIB_SOURCE}/cachelib/cmake" \
  -DNIVO_BUILD_TESTS=OFF \
  -DNIVO_BUILD_EXAMPLES=OFF \
  -DNIVO_BUILD_BENCHMARKS=ON \
  -DNIVO_BENCHMARK_FETCH_GOOGLE_BENCHMARK=OFF \
  -DNIVO_BENCHMARK_WITH_CACHELIB=ON \
  -DNIVO_BENCHMARK_CACHELIB_REVISION="${CACHELIB_COMMIT}"
cmake --build "${BUILD_DIR}" --parallel

cd "${REPOSITORY_ROOT}"
python3 benchmarks/scripts/run_and_record.py \
  --binary "${BUILD_DIR}/nivo_cache_bench" \
  --label "pinned-cachelib"
