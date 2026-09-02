#!/usr/bin/bash

BATCH_SIZE=200

file_list="$1"
short_title="$2"
year="$3"
work_dir="$4"

if [ -z "$file_list" ] || [ ! -f "$file_list" ]; then
    echo "Usage: $0 <file_list> <short_title> <year> <work_dir>"
    exit 1
fi

# ---------------------------------------------------------
# 0. CUDA / cuDNN ENVIRONMENT
# ---------------------------------------------------------
# CONDA_PREFIX is set by `mamba run -n eynollah`; if running standalone,
# fall back to whatever env is active.
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: CONDA_PREFIX not set. Run via 'mamba run -n eynollah ...'"
    exit 1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

CUDNN_PATH=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")
if [ -n "$CUDNN_PATH" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDNN_PATH/lib"
fi

if [ -n "$MAMBA_ROOT_PREFIX" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/lib"
fi

# ---------------------------------------------------------
# 1. PER-INSTANCE TEMP SPACE
# ---------------------------------------------------------
tmp_dir=$(mktemp -d -t "eynollah_${short_title}_${year}_XXXXXX")
batch_file="${tmp_dir}/batch.txt"        # tab-sep slice of remaining work
local_list="${tmp_dir}/local_paths.txt"  # what eynollah reads
remaining="${tmp_dir}/remaining.txt"

cp "$file_list" "$remaining"

# Wipe the whole tmp_dir on exit (JPGs, batch files, everything)
trap "rm -rf $tmp_dir" EXIT

# ---------------------------------------------------------
# 2. OUTPUT DIR
# ---------------------------------------------------------
output_dir="${work_dir}/Layout/${short_title}/${year}"
mkdir -p "$output_dir"
echo "Output: $output_dir"
echo "Tmp:    $tmp_dir"

# ---------------------------------------------------------
# 3. BATCH LOOP
# ---------------------------------------------------------
while [ -s "$remaining" ]; do

    head -n "$BATCH_SIZE" "$remaining" > "$batch_file"
    count=$(wc -l < "$batch_file")
    now=$(date +"%R")
    echo "$now [$short_title/$year]: downloading $count pages..."

    # Download each page as anno_id.jpg
    > "$local_list"
    while IFS=$'\t' read -r anno_id url; do
        local_path="${tmp_dir}/${anno_id}.jpg"
        if curl -sS -f -o "$local_path" "$url"; then
            echo "$local_path" >> "$local_list"
        else
            echo "  WARN: download failed for $anno_id ($url)"
        fi
    done < "$batch_file"

    # Run eynollah on the downloaded JPGs
    echo "$now [$short_title/$year]: running eynollah on $(wc -l < "$local_list") files..."
    eynollah -m model_dir layout -i "$local_list" -o "$output_dir" -cl -fl

    if [ $? -ne 0 ]; then
        echo "CRITICAL ERROR: eynollah crashed for $short_title/$year."
        echo "Batch preserved in: $tmp_dir"
        trap - EXIT
        exit 1
    fi

    # Drop the JPGs for this batch (keep the tmp_dir for the next round)
    find "$tmp_dir" -maxdepth 1 -name '*.jpg' -delete

    # Advance the remaining list
    sed -i "1,${BATCH_SIZE}d" "$remaining"
done

echo "Done: $short_title/$year"
