#!/bin/bash
# regression_suite.sh — ggml-axcl end-to-end regression suite with card-state
# management. RUN ON THE PI.
#
# CARD-STATE MODEL (why: the card wedges when a host process dies with engine
# work in flight — SIGKILL timeouts / segfaults. Its firmware only resets when
# PCIe power is removed; a driver reload resets the host side only. So the
# suite never blind-kills a run: timeouts SIGINT first, escalate to SIGKILL
# only after a grace period, and verify the card between every case.)
#
#   GREEN  execute-canary passes on both engine sets -> run cases
#   TAINT  case output shows corruption signatures (garbage runs, driver
#          faults) while canary still passes -> noted, not blamed on code
#   YELLOW canary fails after a case -> zombie sweep, retry canary
#   RED    canary still fails -> ordered driver reload; on recovery ALL
#          results of this session are discarded and the FULL suite reruns
#          in the now-clean state (max SUITE_MAX_RESTARTS=2 — results from a
#          corrupted card are noise, not signal)
#   BLACK  canary fails after reload -> self-reboot the Pi, the journal
#          records state, a cron @reboot hook resumes automatically; if the
#          canary still fails after reboot -> wall-power instruction
#
# The execute-canary is load+EXECUTE of one engine per family (a wedged card
# loads fine but cannot execute — a load-only canary passes while every run
# hangs, which cost us a day once).
#
# usage:
#   ./regression_suite.sh [quick|full] [--resume] [--fresh]
#   ./regression_suite.sh --install-resume-hook   # cron @reboot auto-resume
#   ./regression_suite.sh --remove-resume-hook
# Env: GOLDEN_DIR, UPDATE_GOLDEN=1, NO_REBOOT=1 (no self-reboot, just abort),
#      SUITE_MAX_RESTARTS (default 2), SUITE_NO_RERUN=1 (skip clean rerun)

set -u
MODE=full; RESUME=0; FRESH=0

RUNNER_BIN="${RUNNER_BIN:-$HOME/build-axcl/bin/llama-simple}"
GOLDEN_DIR="${GOLDEN_DIR:-$HOME/.axcl-regression-golden}"
UPDATE_GOLDEN="${UPDATE_GOLDEN:-0}"
JOURNAL="${JOURNAL:-$HOME/.axcl-regression-journal}"
MAX_RESTARTS="${SUITE_MAX_RESTARTS:-2}"
RUNROOT="${RUNROOT:-$HOME/.axcl-regression-runs}"   # persists reboots (/tmp is tmpfs)
AXCL_SMI="sudo /usr/bin/axcl/axcl-smi"
EXECPROBE="${EXECPROBE:-$HOME/matmul/chunk_probe}"
RESUME_HOOK='# axcl-regression-resume (installed by regression_suite.sh — remove with ./regression_suite.sh --remove-resume-hook)'
RESUME_CMD="$HOME/regression_suite.sh --resume full"

S4_DIR="${S4_DIR:-$HOME/s4-gptq}"
VENDOR06_DIR="${VENDOR06_DIR:-$HOME/Qwen3-0.6B}"
Q35_DIR="${Q35_DIR:-$HOME/Qwen3.5-0.8B-int4}"
GGUF_Q3_Q8="${GGUF_Q3_Q8:-$HOME/models/qwen3-q8.gguf}"
GGUF_Q35_Q4="${GGUF_Q35_Q4:-$HOME/models/qwen35-0.8b-q4km.gguf}"
GGUF_Q35_Q8="${GGUF_Q35_Q8:-$HOME/models/qwen35-0.8b-q8.gguf}"

FLOOR_Q3_S4_DEC=17; FLOOR_Q3_S4_PF=350
FLOOR_Q35_DEC=19;  FLOOR_Q35_PF=110

export LD_LIBRARY_PATH="/usr/lib/axcl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

log()     { printf '%s | %s\n' "$(date +%H:%M:%S)" "$*"; }
journal() { echo "$(date +%s) $*" >> "$JOURNAL"; }        # persists reboots
jget()    { grep -oE "(^| )$1=[^ ]+" "$JOURNAL" 2>/dev/null | tail -1 | cut -d= -f2; }

install_resume_hook() {
    ( crontab -l 2>/dev/null | grep -vF "$RESUME_HOOK" | grep -vF "regression_suite.sh --resume"
      echo "$RESUME_HOOK"
      echo "@reboot sleep 45 && $RESUME_CMD >> $HOME/.axcl-regression-resume.log 2>&1" ) | crontab -
    log "resume hook installed (cron @reboot -> $RESUME_CMD)"
}
remove_resume_hook() {
    crontab -l 2>/dev/null | grep -vF "$RESUME_HOOK" | grep -vF "regression_suite.sh --resume" | crontab -
    log "resume hook removed"
}

for a in "$@"; do
    case "$a" in
        quick|full) MODE="$a" ;;
        --resume) RESUME=1 ;;
        --fresh)  FRESH=1 ;;
        --install-resume-hook) touch "$JOURNAL"; install_resume_hook; exit 0 ;;
        --remove-resume-hook)  remove_resume_hook;  exit 0 ;;
    esac
done

# ------------------------------------------------------------ card state ----
cmm_now() { $AXCL_SMI 2>/dev/null | grep -A1 AX650N | tail -1 | grep -oE '[0-9]+ MiB /' | head -1 | grep -oE '^[0-9]+'; }

zombie_sweep() {
    if pgrep -x llama-simple >/dev/null 2>&1; then
        log "state: stale llama-simple holding the card — SIGTERM then SIGKILL"
        pkill -x llama-simple 2>/dev/null; sleep 3
        pkill -9 -x llama-simple 2>/dev/null; sleep 2
    fi
}

canary() { # EXECUTE canary on both engine families. 0 = GREEN
    [ -x "$EXECPROBE" ] || { log "state: exec-probe missing at $EXECPROBE"; return 2; }
    [ -f "$Q35_DIR/qwen3_5_text_p128_l4_together.axmodel" ] || return 2
    [ -f "$S4_DIR/qwen3_p128_l0_together.axmodel" ] || return 2
    timeout 40 "$EXECPROBE" "$Q35_DIR/qwen3_5_text_p128_l4_together.axmodel" 0 1 2>/dev/null | grep -q "execute rc=0" || return 1
    timeout 40 "$EXECPROBE" "$S4_DIR/qwen3_p128_l0_together.axmodel" 0 0 2>/dev/null | grep -q "execute rc=0" || return 1
    return 0
}

driver_reload() {
    log "state: ordered driver reload (resets the HOST side only)"
    for m in axcl_host ax_pcie_host_dev ax_pcie_p2p_rc ax_pcie_mmb ax_pcie_msg; do
        sudo modprobe -r "$m" 2>/dev/null
    done
    sleep 5
    sudo modprobe ax_pcie_msg; sudo modprobe ax_pcie_mmb; sudo modprobe ax_pcie_p2p_rc
    sudo modprobe ax_pcie_host_dev; sudo modprobe axcl_host
    sleep 8
}

power_cycle() { # optional automated wall-power recovery (smart plug/relay)
    local pc="${POWER_CMD:-}"
    if [ -n "$pc" ] && [ "${NO_POWER_CMD:-0}" != 1 ]; then
        log "state: invoking POWER_CMD for card wall-power cycle: $pc"
        journal session=power-cycling
        sh -c "$pc"
        sleep "${POWER_SETTLE:-20}"
        return 0
    fi
    return 1
}

self_reboot() {
    if [ "${NO_REBOOT:-0}" = 1 ]; then
        log "state: reboot required but NO_REBOOT=1 — aborting. Reboot the Pi, then: $RESUME_CMD"
        journal session=aborted reason=no-reboot mode=$MODE
        exit 2
    fi
    log "state: BLACK — rebooting the Pi; the @reboot hook (or $RESUME_CMD) resumes from the journal"
    journal session=rebooting mode=$MODE restarts=$(jget restarts)
    sync
    sudo reboot; sleep 60; exit 3
}

# host-side recovery only (zombies -> ordered driver reload -> settle).
# NEVER reboots from in here: the callers own the restart counter and the
# reboot decision — an internal self_reboot bypassed the counter once and
# reboot-looped the Pi.
recover() { # 0 = card GREEN
    zombie_sweep
    if canary; then log "state: GREEN after zombie sweep"; return 0; fi
    driver_reload; sleep 2
    if canary; then log "state: GREEN after driver reload"; return 0; fi
    sleep 10   # post-reload enumeration race settle
    if canary; then log "state: GREEN after settle"; return 0; fi
    return 1
}

# --------------------------------------------------------------- runner -----
# SIGINT-first timeout: llama.cpp's abort callback stops cleanly between
# engine calls; a SIGKILL landing mid-execute is what wedges the card in the
# first place. SIGKILL only after a 30s grace period.
run_case() { # name tmo ntok prompt enginedir gguf [extra env...]
    local name="$1" tmo="$2" nt="$3" prompt="$4" edir="$5" gguf="$6"; shift 6
    CASE_NAME="$name"; OUT_TXT="$RUNDIR/$name.out"; OUT_DBG="$RUNDIR/$name.dbg"
    DECODE_TPS=""; PREFILL_TPS=""
    local envs="GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 GGML_AXCL_STREAM=1 GGML_AXCL_LAYER_DIR=$edir"
    [ $# -gt 0 ] && envs="$envs $*"
    cat "$edir"/*.axmodel >/dev/null 2>&1 || true   # warm the page cache
    timeout --signal=INT --kill-after=30 "$tmo" env $envs "$RUNNER_BIN" \
        -m "$gguf" -n "$nt" "$prompt" > "$OUT_TXT" 2> "$OUT_DBG"
    RUN_RC=$?
    zombie_sweep   # even a SIGINT death can leave the device held
    if grep -qE "nil pointer|recv dma size 0|Segmentation|ggml-axcl.*not supported" "$OUT_DBG" 2>/dev/null; then
        return 3   # fault signature
    fi
    if grep -q "device 0 is not connected" "$OUT_DBG" 2>/dev/null; then
        return 4   # transient post-reload enumeration race
    fi
    if awk '!/^\[2026/ && !/: 0x/ && !/^allocated device/' "$OUT_TXT" 2>/dev/null | grep -qE '!{6}'; then
        return 5   # "!!!!!!" garbage runs = card computing wrong, not a code bug
    fi
    DECODE_TPS=$(grep -E "^\s*llama_perf_context_print:\s+eval time" "$OUT_DBG" 2>/dev/null | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+" | head -1)
    PREFILL_TPS=$(grep -E "prompt eval time" "$OUT_DBG" 2>/dev/null | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+" | head -1)
    return $RUN_RC
}

clean_text() { awk '!/^\[2026/ && !/: 0x/ && !/^allocated device/' "$1" 2>/dev/null; }
expect_contains() { clean_text "$OUT_TXT" | grep -qF "$1"; }
expect_tps() {
    local v=""
    [ "$1" = decode ] && v="$DECODE_TPS"
    [ "$1" = prefill ] && v="$PREFILL_TPS"
    [ -n "$v" ] || return 1
    awk -v v="$v" -v f="$2" 'BEGIN{exit !(v>=f)}'
}
backend_md5() { md5sum "$HOME"/build-axcl/bin/libggml-axcl.so* 2>/dev/null | head -1 | cut -d' ' -f1; }

golden_check() { # case-name min_prefix_fraction
    local g="$GOLDEN_DIR/$1.txt"
    if [ ! -s "$g" ] || [ "$UPDATE_GOLDEN" = 1 ]; then
        { echo "# binary_md5 $(backend_md5)"; clean_text "$OUT_TXT"; } > "$g"
        log "      golden created/updated for $1 (md5 $(backend_md5))"
        return 0
    fi
    local frac
    frac=$(python3 - "$g" "$OUT_TXT" <<'EOF'
import sys
def gen(p):
    ls = open(p, encoding="utf-8", errors="replace").read().splitlines(True)
    ls = [l for l in ls if not l.startswith("[2026") and ": 0x" not in l and not l.startswith("allocated device")]
    return "".join(ls[1:] if ls and ls[0].startswith("# binary_md5") else ls)
a, b = gen(sys.argv[1]), gen(sys.argv[2])
i = 0
while i < min(len(a), len(b)) and a[i] == b[i]: i += 1
print(f"{i / len(b):.3f}" if len(b) else "1.000")
EOF
)
    if awk -v f="$frac" -v w="$2" 'BEGIN{exit !(f>=w)}'; then return 0; fi
    log "      golden drift: prefix fraction $frac < $2"
    return 1
}

# result bookkeeping — verdicts journaled so a post-reboot resume (and you)
# can see exactly how far the last clean run got
pass_result()  { PASS=$((PASS+1)); log "PASS  $CASE_NAME ($*)"; journal case=$CASE_NAME verdict=pass; }
fail_result()  { FAIL=$((FAIL+1)); log "FAIL  $CASE_NAME ($*)"; journal case=$CASE_NAME verdict=fail; }
taint_result() { FAIL=$((FAIL+1)); log "TAINT $CASE_NAME ($*) — card corruption signature, NOT counted as code failure"; journal case=$CASE_NAME verdict=tainted; }
skip_result()  { SKIP=$((SKIP+1)); log "SKIP  $CASE_NAME ($*)"; journal case=$CASE_NAME verdict=skip; }

# ---------------------------------------------------------------- startup ---
touch "$JOURNAL"
if [ "$RESUME" = 1 ] && [ "$FRESH" = 0 ] && [ "$(jget session)" = rebooting ]; then
    RESTARTS=$(jget restarts); RESTARTS=${RESTARTS:-0}
    MODE=$(jget mode); MODE=${MODE:-full}
    log "=== resuming session after reboot (restart #$RESTARTS; prior journal kept) ==="
    log "resume: last recorded events:"; tail -5 "$JOURNAL" | sed 's/^/resume:   /'
    log "resume: previous clean verdicts: $(grep -c 'verdict=pass' "$JOURNAL") passes, $(grep -c 'verdict=fail' "$JOURNAL") fails"
else
    RESTARTS=$(jget restarts)   # carried across restart-clean execs (loop guard)
    case "$RESTARTS" in ''|*[!0-9]*) RESTARTS=0 ;; esac
    : > "$JOURNAL"
    journal session=start mode=$MODE binary=$(backend_md5) restarts=$RESTARTS
fi

RUNDIR="$RUNROOT/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$RUNDIR" "$GOLDEN_DIR"
PASS=0; FAIL=0; SKIP=0; POLICY_RESTART=0

log "=== ggml-axcl regression suite ($MODE, restart #$RESTARTS) -> $RUNDIR ==="
log "backend md5: $(backend_md5)"
git -C "$HOME/llama.cpp" log --oneline -1 2>/dev/null | sed 's/^/fork HEAD: /' || log "fork HEAD: not a git checkout"
$AXCL_SMI 2>/dev/null | sed -n '2,3p' | sed 's/^/smi: /'
echo performance | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null 2>&1 \
    && log "cpu governor: performance" || log "cpu governor: NOT SET — perf floors still enforced"

if ! canary; then
    if ! recover; then
        RESTARTS=$((RESTARTS+1)); journal restarts=$RESTARTS
        if [ "$RESTARTS" -le "$MAX_RESTARTS" ]; then self_reboot; fi
        if power_cycle && canary; then
            log "state: GREEN after wall-power cycle"
            RESTARTS=0; journal restarts=0
        else
            log "FAIL  card canary unrecoverable after $RESTARTS restarts — WALL-POWER CYCLE the Pi (README 'Card stability rules' #3), then: $RESUME_CMD"
            journal session=aborted reason=wall-power
            exit 1
        fi
    fi
    POLICY_RESTART=1   # card needed help before case 1: any prior plan is moot
fi
log "cmm baseline: $(cmm_now) MiB — card GREEN"

health_gate() { # after every case: verify the card, escalate if degraded
    if canary; then return 0; fi
    log "state: canary degraded after $CASE_NAME — recovery ladder"
    if recover; then
        POLICY_RESTART=1
        return 0
    fi
    RESTARTS=$((RESTARTS+1)); journal restarts=$RESTARTS
    if [ "$RESTARTS" -le "$MAX_RESTARTS" ]; then self_reboot; fi
    log "FAIL  unrecoverable card state after $RESTARTS restarts — WALL-POWER CYCLE REQUIRED, then: $RESUME_CMD"
    journal session=aborted reason=wall-power
    exit 1
}

maybe_restart_clean() { # a mid-suite recovery invalidates this session's results
    [ "$POLICY_RESTART" = 1 ] || return 0
    POLICY_RESTART=0
    [ "${SUITE_NO_RERUN:-0}" = 1 ] && return 0
    RESTARTS=$((RESTARTS+1)); journal restarts=$RESTARTS
    if [ "$RESTARTS" -le "$MAX_RESTARTS" ]; then
        log "policy: card recovered mid-suite — discarding this session's results, rerunning FULL suite clean (restart #$RESTARTS)"
        journal session=restart-clean mode=full
        exec "$0" full --resume
    fi
    log "policy: restart budget exhausted — reporting current results as-is"
}

# ================================================================ cases =====
P06="The capital of France is"

t3_s4_batch() {
    if run_case q3_s4_batch 240 12 "$P06" "$S4_DIR" "$GGUF_Q3_Q8" GGML_AXCL_BATCH=1 \
          GGML_AXCL_POST_MODEL="$S4_DIR/qwen3_post.axmodel"; then
        ok=1
        expect_contains "Paris" || { fail_result "no Paris"; ok=0; }
        expect_tps decode "$FLOOR_Q3_S4_DEC" || { fail_result "decode ${DECODE_TPS:-?} < $FLOOR_Q3_S4_DEC"; ok=0; }
        golden_check q3_s4_batch 0.95 || { fail_result "golden drift"; ok=0; }
        [ $ok = 1 ] && pass_result "decode ${DECODE_TPS} t/s"
    elif [ $RUN_RC = 4 ]; then
        log "      enumeration race — one retry"
        if run_case q3_s4_batch 240 12 "$P06" "$S4_DIR" "$GGUF_Q3_Q8" GGML_AXCL_BATCH=1 \
               GGML_AXCL_POST_MODEL="$S4_DIR/qwen3_post.axmodel" \
           && expect_contains "Paris" && expect_tps decode "$FLOOR_Q3_S4_DEC"; then
            pass_result "after race retry, ${DECODE_TPS} t/s"
        else
            [ $RUN_RC = 5 ] && taint_result "garbage (retry)" || fail_result "retry rc=$RUN_RC"
        fi
    elif [ $RUN_RC = 5 ]; then
        taint_result "garbage output — card state, not code"
    else
        fail_result "exit $RUN_RC"
    fi
}

t3_s4_prefill() {
    local long06; long06=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 30)")
    if run_case q3_s4_prefill 240 8 "$long06 Summarize in one sentence." "$S4_DIR" "$GGUF_Q3_Q8" \
          GGML_AXCL_BATCH=1 GGML_AXCL_POST_MODEL="$S4_DIR/qwen3_post.axmodel"; then
        expect_tps prefill "$FLOOR_Q3_S4_PF" && pass_result "${PREFILL_TPS} t/s prefill" \
            || fail_result "prefill ${PREFILL_TPS:-no} < $FLOOR_Q3_S4_PF (ladder broken/fell back?)"
    elif [ $RUN_RC = 5 ]; then taint_result "garbage output"; else fail_result "exit $RUN_RC"; fi
}

t3_vendor() {
    if ! ls "$VENDOR06_DIR"/qwen3_p128_l0_together.axmodel >/dev/null 2>&1; then
        skip_result "no engine set in $VENDOR06_DIR"; return
    fi
    if run_case q3_vendor 240 12 "$P06" "$VENDOR06_DIR" "$GGUF_Q3_Q8"; then
        expect_contains "Paris" && pass_result "decode ${DECODE_TPS:-?} t/s" || fail_result "no Paris"
    elif [ $RUN_RC = 5 ]; then taint_result "garbage output"; else fail_result "exit $RUN_RC"; fi
}

t35_q4_decode() {
    if run_case q35_q4_decode 240 40 "What are the first five prime numbers?" "$Q35_DIR" "$GGUF_Q35_Q4"; then
        ok=1
        expect_contains "2, 3, 5, 7, and 11" || { fail_result "wrong primes"; ok=0; }
        expect_tps decode "$FLOOR_Q35_DEC" || { fail_result "decode ${DECODE_TPS:-?} < $FLOOR_Q35_DEC"; ok=0; }
        golden_check q35_q4_decode 0.90 || { fail_result "golden drift"; ok=0; }
        [ $ok = 1 ] && pass_result "decode ${DECODE_TPS} t/s"
    elif [ $RUN_RC = 5 ]; then taint_result "garbage output"; else fail_result "exit $RUN_RC"; fi
}

t35_q8_decode() {
    if run_case q35_q8_decode 240 40 "What are the first five prime numbers?" "$Q35_DIR" "$GGUF_Q35_Q8"; then
        expect_contains "2, 3, 5, 7, and 11" && pass_result "decode ${DECODE_TPS:-?} t/s" || fail_result "wrong primes"
    elif [ $RUN_RC = 5 ]; then taint_result "garbage output"; else fail_result "exit $RUN_RC"; fi
}

t35_agreement() {
    local agp="Alyssa, Ben, and Carmen are splitting 12 apples evenly. How many does each get? Explain step by step."
    if ! run_case q35_agree_ref 240 40 "$agp" "$Q35_DIR" "$GGUF_Q35_Q4"; then
        [ $RUN_RC = 5 ] && taint_result "garbage (ref leg)" || fail_result "ref leg exit $RUN_RC"; return
    fi
    local ref="$OUT_TXT"
    if ! run_case q35_agree_lad 240 40 "$agp" "$Q35_DIR" "$GGUF_Q35_Q4" GGML_AXCL_BATCH=1; then
        [ $RUN_RC = 5 ] && taint_result "garbage (ladder leg)" || fail_result "ladder leg exit $RUN_RC"; return
    fi
    local frac
    frac=$(python3 - "$ref" "$OUT_TXT" <<'EOF'
import sys
def gen(p):
    ls = open(p, encoding="utf-8", errors="replace").read().splitlines(True)
    return "".join(l for l in ls if not l.startswith("[2026") and ": 0x" not in l and not l.startswith("allocated device"))
a, b = gen(sys.argv[1]), gen(sys.argv[2])
i = 0
while i < min(len(a), len(b)) and a[i] == b[i]: i += 1
print(f"{i / len(b):.3f}" if len(b) else "0.000")
EOF
)
    awk -v f="$frac" 'BEGIN{exit !(f>=0.60)}' && pass_result "ladder vs per-token prefix $frac" \
        || fail_result "prefix $frac — chunked state diverges"
}

t35_prefill() {
    local long35; long35=$(python3 -c "print('The quick brown fox jumps over the lazy dog while the cat sleeps. ' * 30)")
    if run_case q35_prefill 240 8 "$long35 Summarize in one sentence." "$Q35_DIR" "$GGUF_Q35_Q4" GGML_AXCL_BATCH=1; then
        expect_tps prefill "$FLOOR_Q35_PF" && pass_result "${PREFILL_TPS} t/s prefill" \
            || fail_result "prefill ${PREFILL_TPS:-no} < $FLOOR_Q35_PF"
    elif [ $RUN_RC = 5 ]; then taint_result "garbage output"; else fail_result "exit $RUN_RC (segfault regression?)"; fi
}

t35_deep() {
    local deep; deep=$(python3 -c "print('Pack my box with five dozen liquor jugs quickly please. ' * 95)")
    if run_case q35_deep 300 8 "$deep Summarize in one sentence." "$Q35_DIR" "$GGUF_Q35_Q4" GGML_AXCL_BATCH=1; then
        pass_result "deep prompt, prefill ${PREFILL_TPS:-?} t/s, no crash"
    elif [ $RUN_RC = 5 ]; then taint_result "garbage output"; else fail_result "exit $RUN_RC — deep-prompt crash is back"; fi
}

t35_soak() {
    local a b
    run_case q35_soak_a 300 500 "Write a very long story about a mountain expedition." "$Q35_DIR" "$GGUF_Q35_Q4" >/dev/null 2>&1
    sleep 5; a=$(cmm_now)
    run_case q35_soak_b 300 500 "Write another very long story, this time at sea." "$Q35_DIR" "$GGUF_Q35_Q4" >/dev/null 2>&1
    sleep 5; b=$(cmm_now)
    if [ -n "$a" ] && [ "$b" -le $((a + 64)) ]; then
        pass_result "cmm $a -> $b MiB across two 500-tok runs (no leak)"
    else
        fail_result "cmm $a -> $b MiB (grew >64 MiB — leak?)"
    fi
    [ -s "$RUNDIR/q35_soak_b.out" ] || fail_result "second soak run produced no output"
}

t35_unicode() {
    if run_case q35_unicode 240 16 "Translate to French: café naïve résumé 🚀" "$Q35_DIR" "$GGUF_Q35_Q4"; then
        [ -s "$OUT_TXT" ] && pass_result "no crash" || fail_result "empty output"
    elif [ $RUN_RC = 5 ]; then taint_result "garbage output"; else fail_result "exit $RUN_RC"; fi
}

run_managed() { # fn-name — case + health gate + clean-rerun policy
    local fn="$1"
    log "-- $fn"
    "$fn"
    # cheap per-case check: fault signatures in the log (no device work).
    # The EXECUTE canary runs only on suspicion (rc 3/5) or explicit request —
    # per-case canaries leak device contexts and race the firmware's
    # create/destroy path, which itself destabilizes the card.
    case "$RUN_RC" in
        3|5|"") health_gate ;;
        *) [ "${SUITE_CANARY_EVERY:-0}" = 1 ] && health_gate || true ;;
    esac
    maybe_restart_clean
}

# ------------------------------------------------------------------ run ----
for case in t3_s4_batch t3_vendor t35_q4_decode t35_q8_decode t35_agreement t35_unicode; do
    run_managed "$case"
done
if [ "$MODE" = full ]; then
    for case in t3_s4_prefill t35_prefill t35_deep t35_soak; do
        run_managed "$case"
    done
fi

journal session=complete pass=$PASS fail=$FAIL skip=$SKIP
log "=== SUMMARY: $PASS pass, $FAIL fail, $SKIP skip — logs in $RUNDIR ==="
log "journal: $JOURNAL ($(grep -c 'verdict=pass' "$JOURNAL") pass verdicts recorded)"
[ "$FAIL" = 0 ]
