#!/usr/bin/env bash
# Run every notebook, headless, in order. Creates the venv from requirements.txt if it is missing.
#
#   ./run_all.sh                         all notebooks: neural_nets, then image_nets, then qwen
#   ./run_all.sh image_nets              only that domain (any number of domains)
#   ./run_all.sh neural_nets/notebooks/00_environment_and_gpu.ipynb   a single notebook
#   ./run_all.sh --dry-run               list what would run, run nothing
#
# Executed copies (with fresh outputs) and per-notebook logs go to executed/<domain>/.
# The committed notebooks are NOT modified, but the notebooks themselves still write
# their usual files (checkpoints, datasets, plots) into <domain>/artifacts/, so a full
# run re-trains the models and overwrites those. Within a domain, notebooks depend on
# the earlier ones, so a domain stops at its first failure; the next domain still runs.
set -uo pipefail
cd "$(dirname "$0")"

DRY=0; ARGS=()
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1 || ARGS+=("$a"); done
[ ${#ARGS[@]} -eq 0 ] && ARGS=(neural_nets image_nets qwen)

NBS=()                                   # build the ordered notebook list
for a in "${ARGS[@]}"; do
  if [ -f "$a" ]; then NBS+=("$a")
  elif [ -d "$a/notebooks" ]; then for nb in "$a"/notebooks/*.ipynb; do NBS+=("$nb"); done
  else echo "skip: '$a' is neither a notebook nor a domain with notebooks" >&2; fi
done
[ ${#NBS[@]} -eq 0 ] && { echo "nothing to run" >&2; exit 2; }

echo "Notebooks to run (${#NBS[@]}):"; printf '  %s\n' "${NBS[@]}"
[ $DRY -eq 1 ] && exit 0

if [ ! -x venv/bin/python ]; then
  echo ">> venv missing: creating it and installing requirements.txt (large GPU download)"
  python3 -m venv venv && venv/bin/pip install -r requirements.txt || exit 1
fi
source venv/bin/activate

OUT=executed; RESULTS=(); FAILED=0; SKIP_DOMAIN=""
for nb in "${NBS[@]}"; do
  dom=$(dirname "$(dirname "$nb")"); name=$(basename "$nb" .ipynb)
  if [ "$dom" = "$SKIP_DOMAIN" ]; then RESULTS+=("SKIPPED  $nb (earlier notebook in $dom failed)"); continue; fi
  mkdir -p "$OUT/$dom"; t0=$SECONDS; echo ">> $nb"
  if jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
       --ExecutePreprocessor.kernel_name=python3 --output-dir "$OUT/$dom" "$nb" > "$OUT/$dom/$name.log" 2>&1; then
    RESULTS+=("OK       $nb ($((SECONDS-t0))s)")
  else
    RESULTS+=("FAILED   $nb ($((SECONDS-t0))s)  log: $OUT/$dom/$name.log"); FAILED=1; SKIP_DOMAIN="$dom"
  fi
done
echo; echo "===== summary ====="; printf '%s\n' "${RESULTS[@]}"
exit $FAILED
