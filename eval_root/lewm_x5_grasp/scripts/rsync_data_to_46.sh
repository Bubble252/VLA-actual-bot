#!/usr/bin/env bash
# Parallel rsync with per-file retries + final verification pass.
# Usage: bash scripts/rsync_data_to_46.sh
set -u

DATA="${DATA:-/home/casia/project/ARX_X5_clean_server46(7.12get-data)/lewm_x5_grasp/data}"
REMOTE="${REMOTE:-zhangjh@172.18.127.46}"
DEST="${DEST:-/9950backfile/zhangjh/wh/ARX_X5_data}"
JOBS="${JOBS:-2}"
MAX_RETRY="${MAX_RETRY:-5}"
DIRS=(pick_and_place_can pour_foam_from_cup can_stacking)

LOGDIR="${LOGDIR:-/tmp/arx_rsync_retry_$$}"
mkdir -p "$LOGDIR"
echo "log=$LOGDIR jobs=$JOBS max_retry=$MAX_RETRY" | tee "$LOGDIR/master.log"

ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" \
  "mkdir -p ${DEST}/pick_and_place_can ${DEST}/pour_foam_from_cup ${DEST}/can_stacking"

: > "$LOGDIR/files.txt"
for d in "${DIRS[@]}"; do
  find "$DATA/$d" -maxdepth 1 -type f -printf "${d}/%f\n" | sort >> "$LOGDIR/files.txt"
done
echo "files=$(wc -l < "$LOGDIR/files.txt")" | tee -a "$LOGDIR/master.log"
echo "=== start $(date) ===" | tee -a "$LOGDIR/master.log"

rsync_one() {
  local rel="$1"
  local dir="${rel%%/*}"
  local name="${rel##*/}"
  local src="${DATA}/${rel}"
  local dst="${REMOTE}:${DEST}/${dir}/"
  local attempt st=1
  for attempt in $(seq 1 "$MAX_RETRY"); do
    rsync -a --partial --append-verify --timeout=300 \
      -e "ssh -o BatchMode=yes -o Compression=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -c aes128-gcm@openssh.com" \
      "$src" "$dst" >"${LOGDIR}/${dir}__${name}.try${attempt}.log" 2>&1
    st=$?
    if [ "$st" -eq 0 ]; then
      printf '%s\t%d\t%s\tattempt=%d\n' "$(date +%H:%M:%S)" "$st" "$rel" "$attempt" >> "$LOGDIR/done.tsv"
      echo "[ok] $rel (attempt $attempt)" >> "$LOGDIR/master.log"
      return 0
    fi
    echo "[retry $attempt/$MAX_RETRY exit=$st] $rel" >> "$LOGDIR/master.log"
    sleep $((attempt * 3))
  done
  printf '%s\t%d\t%s\tattempt=%d\n' "$(date +%H:%M:%S)" "$st" "$rel" "$MAX_RETRY" >> "$LOGDIR/done.tsv"
  echo "[FAIL $st] $rel after $MAX_RETRY attempts" >> "$LOGDIR/master.log"
  echo "$rel" >> "$LOGDIR/failed.txt"
  return 0
}
export -f rsync_one
export DATA REMOTE DEST LOGDIR MAX_RETRY

xargs -P "$JOBS" -a "$LOGDIR/files.txt" -I{} bash -c 'rsync_one "$@"' _ {}

# Final pass: retry anything still missing/size-mismatched on remote
echo "=== verify/repair $(date) ===" | tee -a "$LOGDIR/master.log"
: > "$LOGDIR/repair.txt"
while IFS= read -r rel; do
  dir="${rel%%/*}"
  name="${rel##*/}"
  local_sz=$(stat -c%s "${DATA}/${rel}")
  remote_sz=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" \
    "stat -c%s ${DEST}/${dir}/${name} 2>/dev/null || echo 0")
  if [ "$remote_sz" != "$local_sz" ]; then
    echo "$rel local=$local_sz remote=$remote_sz" >> "$LOGDIR/repair.txt"
  fi
done < "$LOGDIR/files.txt"

if [ -s "$LOGDIR/repair.txt" ]; then
  echo "repair_count=$(wc -l < "$LOGDIR/repair.txt")" | tee -a "$LOGDIR/master.log"
  cut -d' ' -f1 "$LOGDIR/repair.txt" > "$LOGDIR/repair_files.txt"
  xargs -P "$JOBS" -a "$LOGDIR/repair_files.txt" -I{} bash -c 'rsync_one "$@"' _ {}
else
  echo "repair_count=0" | tee -a "$LOGDIR/master.log"
fi

# Final verify summary
bad=0
while IFS= read -r rel; do
  dir="${rel%%/*}"
  name="${rel##*/}"
  local_sz=$(stat -c%s "${DATA}/${rel}")
  remote_sz=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" \
    "stat -c%s ${DEST}/${dir}/${name} 2>/dev/null || echo 0")
  if [ "$remote_sz" != "$local_sz" ]; then
    echo "[MISSING/MISMATCH] $rel local=$local_sz remote=$remote_sz" | tee -a "$LOGDIR/master.log"
    bad=$((bad + 1))
  fi
done < "$LOGDIR/files.txt"

echo "=== done $(date) bad=$bad ===" | tee -a "$LOGDIR/master.log"
ssh -o BatchMode=yes "$REMOTE" \
  "du -sh ${DEST}/pick_and_place_can ${DEST}/pour_foam_from_cup ${DEST}/can_stacking"
exit "$bad"
