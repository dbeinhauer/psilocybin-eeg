#!/bin/bash
# Show running containers + last 5 lines of each log
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$SCRIPT_DIR/logs"

for container in $(docker ps --filter "name=psilocybin-issue-" --format "{{.Names}}"); do
  issue=$(echo "$container" | sed 's/psilocybin-issue-//')
  echo "=== Issue #$issue (running) ==="
  tail -5 "$LOGS_DIR/issue-$issue.log" 2>/dev/null || echo "(no log yet)"
  echo ""
done
# Also show recently finished logs
for log in "$LOGS_DIR"/issue-*.log; do
  [[ -e "$log" ]] || continue
  issue=$(basename "$log" .log | sed 's/issue-//')
  if [[ -z "$(docker ps --filter "name=psilocybin-issue-$issue" --format "{{.Names}}")" ]]; then
    echo "=== Issue #$issue (finished) ==="
    tail -5 "$log" 2>/dev/null
    echo ""
  fi
done
