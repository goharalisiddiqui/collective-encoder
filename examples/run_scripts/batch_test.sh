#!/bin/bash
# ==============================================================================
# batch-cpu.sh
#
# Runs tests in multiple directories matching a glob pattern or regex.
# Pure Bash implementation (no Python dependencies).
#
# Usage:
#   bash batch-cpu.sh <directory_pattern_or_regex> <config_yaml>
#   ./batch-cpu.sh "/path/to/runs/*_1" path/to/config.yaml
#
# For each matching directory:
#   1. Copies the config YAML file to that directory
#   2. Updates the 'outpath' entry in the copied config to the directory's absolute path
#   3. Updates the 'load_model' (or 'load_network') entry to the directory's absolute path
#   4. Executes/submits: bash run_cpu.sh -t -c <path_to_config_file>
# ==============================================================================

set -e

# Validate arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dir_pattern_or_regex> <config_yaml_file>"
    echo "Example: $0 '/path/to/runs/*_1' test_configs/test_dmetrics.yaml"
    exit 1
fi

PATTERN="$1"
CONFIG_FILE="$2"

# Ensure config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

CONFIG_BASENAME=$(basename "$CONFIG_FILE")
ABS_CONFIG_FILE=$(realpath "$CONFIG_FILE")
ROOT_DIR=$(pwd)

echo "=================================================================="
echo "Batch CPU Runner (Pure Bash)"
echo "  Pattern / Regex : $PATTERN"
echo "  Source Config   : $ABS_CONFIG_FILE"
echo "  Working Dir     : $ROOT_DIR"
echo "=================================================================="

# Find all matching directories
matched_dirs=()

# 1. Expand glob patterns (e.g. /path/to/runs/*_1 or test_runs/VAE_*)
shopt -s nullglob globstar
eval "for d in $PATTERN; do
    if [ -d \"\$d\" ]; then
        matched_dirs+=(\"\$(realpath \"\$d\")\")
    fi
done"
shopt -u nullglob globstar

# 2. If no direct glob match, search using find with regex matching
if [ "${#matched_dirs[@]}" -eq 0 ]; then
    BASE_SEARCH_DIR="."
    if [[ "$PATTERN" == /* ]]; then
        BASE_SEARCH_DIR=$(dirname "$PATTERN" | sed -E 's/(\*|\?|\[|\\|\(|\)|\^|\$).*//')
        [ -z "$BASE_SEARCH_DIR" ] && BASE_SEARCH_DIR="/"
        [ ! -d "$BASE_SEARCH_DIR" ] && BASE_SEARCH_DIR="/"
    fi

    while IFS= read -r dir; do
        if [ -d "$dir" ]; then
            matched_dirs+=("$(realpath "$dir")")
        fi
    done < <(find "$BASE_SEARCH_DIR" -type d -regextype posix-extended -regex "$PATTERN" 2>/dev/null || true)
fi

# Remove duplicates while preserving sorted order
unique_dirs=()
if [ "${#matched_dirs[@]}" -gt 0 ]; then
    while IFS= read -r line; do
        [ -n "$line" ] && unique_dirs+=("$line")
    done < <(printf "%s\n" "${matched_dirs[@]}" | sort -u)
fi

if [ "${#unique_dirs[@]}" -eq 0 ]; then
    echo "No directories matched pattern '$PATTERN'."
    exit 0
fi

echo "Found ${#unique_dirs[@]} matching directory(ies):"
for d in "${unique_dirs[@]}"; do
    echo "  - $d"
done
echo "------------------------------------------------------------------"

# Process each matching directory
for TARGET_DIR in "${unique_dirs[@]}"; do
    echo "[Processing] $TARGET_DIR"

    TARGET_CONFIG="$TARGET_DIR/$CONFIG_BASENAME"

    # 1. Copy config file to destination directory
    cp "$ABS_CONFIG_FILE" "$TARGET_CONFIG"
    echo "  -> Copied config to: $TARGET_CONFIG"

    # 2. Update 'outpath' entry to the 'test_runs' subdirectory of TARGET_DIR
    OUTPATH_DIR="${TARGET_DIR}/test_runs"
    if grep -q "^[[:space:]]*outpath:" "$TARGET_CONFIG"; then
        sed -i -E "s|^([[:space:]]*outpath:)[[:space:]]*.*|\1 ${OUTPATH_DIR}|" "$TARGET_CONFIG"
    else
        echo -e "outpath: ${OUTPATH_DIR}\n$(cat "$TARGET_CONFIG")" > "$TARGET_CONFIG"
    fi
    echo "  -> Updated 'outpath' to: $OUTPATH_DIR"

    # 3. Update 'load_model' or 'load_network' entry to the absolute path of TARGET_DIR
    if grep -q "^[[:space:]]*load_model:" "$TARGET_CONFIG"; then
        sed -i -E "s|^([[:space:]]*load_model:)[[:space:]]*.*|\1 ${TARGET_DIR}|" "$TARGET_CONFIG"
        echo "  -> Updated 'load_model' to: $TARGET_DIR"
    elif grep -q "^[[:space:]]*load_network:" "$TARGET_CONFIG"; then
        sed -i -E "s|^([[:space:]]*load_network:)[[:space:]]*.*|\1 ${TARGET_DIR}|" "$TARGET_CONFIG"
        echo "  -> Updated 'load_network' to: $TARGET_DIR"
    else
        echo -e "load_model: ${TARGET_DIR}\n$(cat "$TARGET_CONFIG")" > "$TARGET_CONFIG"
        echo "  -> Added 'load_model: $TARGET_DIR'"
    fi

    # 4. Submit / run test script with run_cpu.sh
    echo "  -> Executing: sbatch run_cpu.sh -t -c $TARGET_CONFIG"
    sbatch run_cpu.sh -t -c "$TARGET_CONFIG"

    echo "  -> Finished submitting for $TARGET_DIR"
    echo ""
done

echo "=================================================================="
echo "Batch execution completed for all matching directories."
echo "=================================================================="
