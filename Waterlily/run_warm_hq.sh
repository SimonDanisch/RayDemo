#!/usr/bin/env bash
# Auto-restart wrapper: runs smoke_warm_hq.jl until all 450 frames exist OR we
# stop making progress on a clean-exit attempt. record_longrunning skips
# already-saved frames so each restart resumes cheaply.
set -u
cd "$(dirname "$0")"
mkdir -p logs

FRAME_DIR="smoke_warm_hq_frames"
MAX_ATTEMPTS=30
attempt=0
while [[ $attempt -lt $MAX_ATTEMPTS ]]; do
    done_count=$(ls "$FRAME_DIR" 2>/dev/null | wc -l)
    if [[ $done_count -ge 450 ]]; then
        echo "[wrapper] all 450 frames done — running julia once more to assemble mp4"
        julia --project=/sim/Programmieren/VulkanDev --threads=auto smoke_warm_hq.jl \
            > "logs/warm_assemble.log" 2>&1
        break
    fi
    attempt=$((attempt+1))
    ts=$(date +%Y%m%d_%H%M%S)
    echo "[wrapper] attempt $attempt/$MAX_ATTEMPTS — $done_count/450 frames done — starting julia"
    julia --project=/sim/Programmieren/VulkanDev --threads=auto smoke_warm_hq.jl \
        > "logs/warm_run_${ts}.log" 2>&1
    ec=$?
    done_after=$(ls "$FRAME_DIR" 2>/dev/null | wc -l)
    echo "[wrapper] attempt $attempt exit=$ec  frames now: $done_after"
    if [[ $ec -eq 0 && $done_after -ge 450 ]]; then
        break
    fi
    # Only stop if a CLEAN exit made zero progress (signal exits may be
    # external kills — those should still retry).
    if [[ $done_after -le $done_count && $ec -lt 128 ]]; then
        echo "[wrapper] no progress on a clean attempt — stopping"
        break
    fi
    sleep 2
done

final=$(ls "$FRAME_DIR" 2>/dev/null | wc -l)
echo "[wrapper] final frame count: $final/450"
