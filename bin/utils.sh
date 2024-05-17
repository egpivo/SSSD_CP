DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/../bin/color_map.sh"


check_file_exists() {
  local FILE_PATH="$1"
  if [[ ! -f "${FILE_PATH}" ]]; then
    echo "Error: File '${FILE_PATH}' not found." >&2
    exit "${ERROR_EXITCODE}"
  fi
}
