#!/usr/bin/env bash
# Auto-restart the HQ render until all 450 frames are in dolphin_hq_frames/.
# Makie.record_longrunning auto-skips existing frames, so each restart
# resumes from where the prior attempt crashed.
set -u
cd "$(dirname "$0")"
mkdir -p logs

MAX_ATTEMPTS=40
ATTEMPT=0
while [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; do
    DONE=$(ls dolphin_hq_frames/ 2>/dev/null | wc -l)
    if [[ $DONE -ge 450 ]]; then
        echo "[wrapper] all 450 frames done — assembling final mp4"
        break
    fi
    ATTEMPT=$((ATTEMPT+1))
    echo "[wrapper] attempt $ATTEMPT/$MAX_ATTEMPTS — $DONE/450 frames done — starting julia"
    timestamp=$(date +%Y%m%d_%H%M%S)
    julia --project=/sim/Programmieren/VulkanDev --threads=auto hq_render.jl \
        > "logs/hq_run_${timestamp}.log" 2>&1
    ec=$?
    DONE_AFTER=$(ls dolphin_hq_frames/ 2>/dev/null | wc -l)
    echo "[wrapper] attempt $ATTEMPT exit=$ec  frames now: $DONE_AFTER"
    if [[ $ec -eq 0 && $DONE_AFTER -ge 450 ]]; then
        break
    fi
    # If no progress was made AND the run exited cleanly (not killed),
    # bail to avoid an infinite restart loop on a permanently-broken state.
    # Signal exits (128+N, e.g. SIGKILL=137, SIGTERM=143) shouldn't count —
    # I may be killing + restarting the julia process to pick up a code fix.
    if [[ $DONE_AFTER -le $DONE && $ec -lt 128 ]]; then
        echo "[wrapper] no progress this attempt (clean exit) — stopping"
        break
    fi
    sleep 2
done

DONE=$(ls dolphin_hq_frames/ 2>/dev/null | wc -l)
echo "[wrapper] final frame count: $DONE/450"
if [[ $DONE -ge 450 ]]; then
    # Final attempt: have julia assemble the mp4 (writes to dolphin_hq.mp4)
    echo "[wrapper] re-running julia to assemble final mp4"
    julia --project=/sim/Programmieren/VulkanDev --threads=auto hq_render.jl \
        > logs/hq_assemble.log 2>&1
fi
