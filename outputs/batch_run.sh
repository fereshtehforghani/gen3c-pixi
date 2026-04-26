#!/usr/bin/env bash
set -u
cd /project/6101824/ferfer/GEN3C

LOG=/project/6101824/ferfer/GEN3C/outputs/batch_run.log
: > "$LOG"

echo "[$(date -Is)] batch start, 16 images" >> "$LOG"
for img in assets/diffusion/000000.png assets/diffusion/000001.png \
           assets/diffusion/000002.png assets/diffusion/000003.png \
           assets/diffusion/000004.png assets/diffusion/000005.png \
           assets/diffusion/000006.png assets/diffusion/000007.png \
           assets/diffusion/000008.png assets/diffusion/000009.png \
           assets/diffusion/000010.png assets/diffusion/000011.png \
           assets/diffusion/000012.png assets/diffusion/000013.png \
           assets/diffusion/000014.png assets/diffusion/000015.png; do
    stem=$(basename "$img" .png)
    name="batch_${stem}"
    echo "[$(date -Is)] >>> $img -> outputs/${name}.mp4" >> "$LOG"
    pixi run gen3c-single-image -- \
        --input_image_path "$img" \
        --video_save_name "$name" \
        --disable_guardrail --disable_prompt_encoder --disable_prompt_upsampler \
        >> "$LOG" 2>&1
    rc=$?
    echo "[$(date -Is)] <<< $img rc=$rc" >> "$LOG"
done
echo "[$(date -Is)] batch done" >> "$LOG"
