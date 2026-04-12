#!/bin/bash
# claude-orchestrator.sh — spin up one Docker container per open GitHub issue,
# run Claude Code headlessly inside each, and let it open a PR unattended.
set -euo pipefail

###############################################################################
# Defaults
###############################################################################
ISSUES=()
LABEL=""
MODEL=""
DRY_RUN=false
MAX_PARALLEL=3
MAX_TURNS=50
TIMEOUT=1800
REBUILD=false
IMAGE="psilocybin-eeg-sandbox:latest"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"

###############################################################################
# Colour helpers
###############################################################################
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[orchestrator]${NC} $*"; }
warn()    { echo -e "${YELLOW}[orchestrator]${NC} $*"; }
error()   { echo -e "${RED}[orchestrator]${NC} $*" >&2; }

###############################################################################
# CLI argument parsing
###############################################################################
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --issues N [N ...]   Process only these issue numbers
  --label LABEL        Filter issues by GitHub label
  --model MODEL        Claude model to use (e.g. sonnet, opus, claude-sonnet-4-6)
  --dry-run            Print plan without running anything
  --max-parallel N     Max containers to run in parallel (default: $MAX_PARALLEL)
  --max-turns N        Max Claude conversation turns per issue (default: $MAX_TURNS)
  --timeout SECS       Kill container after this many seconds (default: $TIMEOUT)
  --rebuild            Force rebuild of the Docker image
  -h, --help           Show this help message
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --issues)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        ISSUES+=("$1"); shift
      done ;;
    --label)        LABEL="$2";        shift 2 ;;
    --model)        MODEL="$2";        shift 2 ;;
    --dry-run)      DRY_RUN=true;      shift   ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --max-turns)    MAX_TURNS="$2";    shift 2 ;;
    --timeout)      TIMEOUT="$2";      shift 2 ;;
    --rebuild)      REBUILD=true;      shift   ;;
    -h|--help)      usage ;;
    *) error "Unknown flag: $1"; exit 1 ;;
  esac
done

###############################################################################
# Prerequisite checks
###############################################################################
for cmd in docker gh git; do
  if ! command -v "$cmd" &>/dev/null; then
    error "Required command not found: $cmd"
    exit 1
  fi
done

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  info "ANTHROPIC_API_KEY not set — will use interactive login from Docker volume."
  info "Run the one-time auth setup if you haven't already (see CLAUDE.md)."
fi

if [[ -z "${GH_TOKEN:-}" ]]; then
  error "GH_TOKEN is not set."
  exit 1
fi

###############################################################################
# Discover issues
###############################################################################
if [[ ${#ISSUES[@]} -eq 0 ]]; then
  info "Discovering open issues..."
  GH_ARGS=(issue list --state open --limit 200 --json number)
  if [[ -n "$LABEL" ]]; then
    GH_ARGS+=(--label "$LABEL")
  fi
  mapfile -t ISSUES < <(gh "${GH_ARGS[@]}" --jq '.[].number')
fi

if [[ ${#ISSUES[@]} -eq 0 ]]; then
  info "No open issues found. Nothing to do."
  exit 0
fi

###############################################################################
# Filter issues that already have an open claude/issue-N PR
###############################################################################
info "Checking for existing Claude PRs..."
OPEN_CLAUDE_PRS=$(gh pr list --state open --json headRefName --jq '.[].headRefName' 2>/dev/null || true)

FILTERED_ISSUES=()
for n in "${ISSUES[@]}"; do
  BRANCH="claude/issue-$n"
  if echo "$OPEN_CLAUDE_PRS" | grep -qx "$BRANCH"; then
    warn "Issue #$n already has an open PR on branch $BRANCH -- skipping."
  else
    FILTERED_ISSUES+=("$n")
  fi
done
ISSUES=("${FILTERED_ISSUES[@]}")

if [[ ${#ISSUES[@]} -eq 0 ]]; then
  info "All issues already have open PRs. Nothing to do."
  exit 0
fi

info "Issues to process: ${ISSUES[*]}"
info "Config: model=${MODEL:-default}, max-parallel=$MAX_PARALLEL, max-turns=$MAX_TURNS, timeout=${TIMEOUT}s"

###############################################################################
# Build Docker image
###############################################################################
build_image() {
  if $REBUILD || ! docker image inspect "$IMAGE" &>/dev/null; then
    info "Building Docker image $IMAGE..."
    docker build \
      -f "$REPO_ROOT/.devcontainer/Dockerfile" \
      -t "$IMAGE" \
      "$REPO_ROOT"
  else
    info "Docker image $IMAGE already exists (use --rebuild to force rebuild)."
  fi
}

if ! $DRY_RUN; then
  build_image
fi

###############################################################################
# Helpers
###############################################################################
mkdir -p "$LOGS_DIR"

create_worktree() {
  local n="$1"
  local wt="$REPO_ROOT/.worktrees/issue-$n"
  local branch="claude/issue-$n"

  if [[ -d "$wt" ]]; then
    warn "Worktree $wt already exists -- reusing."
    return
  fi

  info "Creating worktree for issue #$n..."
  cd "$REPO_ROOT"
  git fetch origin develop --quiet
  git worktree add "$wt" -b "$branch" origin/develop
}

# Start a container and return immediately (detached).
start_container() {
  local n="$1"
  local container="psilocybin-issue-$n"
  local wt="$REPO_ROOT/.worktrees/issue-$n"
  local log="$LOGS_DIR/issue-$n.log"

  # Remove any leftover container with the same name
  docker rm -f "$container" &>/dev/null || true

  info "[issue-$n] starting container (max-turns=$MAX_TURNS, timeout=${TIMEOUT}s)..."

  # Build env var flags — only pass ANTHROPIC_API_KEY if set
  local -a env_flags=(
    -e GH_TOKEN="$GH_TOKEN"
    -e CLAUDE_CONFIG_DIR=/home/researcher/.claude
    -e MPLBACKEND=Agg
    -e PSILOCYBIN_DATA_DIR=/data
    -e PSILOCYBIN_RESULTS_DIR=/results
  )
  if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    env_flags+=(-e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY")
  fi

  # Capture the container ID, not logs — logs are streamed via `docker logs` later
  docker run \
    --detach \
    --name "$container" \
    --stop-timeout 10 \
    --cap-add=NET_ADMIN \
    --cap-add=NET_RAW \
    -v "$wt":/workspace \
    -v "$REPO_ROOT/data":/data:ro \
    -v "$REPO_ROOT/plots":/plots:ro \
    -v "$REPO_ROOT/results":/results:ro \
    -v "$REPO_ROOT/results_db":/results_db:ro \
    -v psilocybin-claude-config:/home/researcher/.claude \
    "${env_flags[@]}" \
    "$IMAGE" \
    bash -c "sudo /usr/local/bin/init-firewall.sh && pip install -e /workspace --quiet && claude --dangerously-skip-permissions --print --max-turns $MAX_TURNS ${MODEL:+--model $MODEL} -p '/work-on $n'" \
    >/dev/null 2>&1

  # Stream container logs to the log file in background
  docker logs -f "$container" >"$log" 2>&1 &
}

# Block until the named container exits (or times out) and return its exit code.
wait_container() {
  local n="$1"
  local container="psilocybin-issue-$n"
  local log="$LOGS_DIR/issue-$n.log"

  local code

  # Wait with timeout — kill the container if it exceeds the limit
  if ! code=$(timeout "$TIMEOUT" docker wait "$container" 2>/dev/null); then
    warn "[issue-$n] timed out after ${TIMEOUT}s — stopping container..."
    docker stop -t 10 "$container" &>/dev/null || true
    code="timeout"
    echo "[TIMEOUT] Container killed after ${TIMEOUT}s" >>"$log"
  fi

  # Clean up: kill the background `docker logs` process for this container
  local log_pid
  log_pid=$(jobs -p 2>/dev/null | tail -1)
  kill "$log_pid" 2>/dev/null || true

  docker rm -f "$container" &>/dev/null || true
  echo "$code"
}

###############################################################################
# Main loop — process issues with bounded parallelism
###############################################################################
declare -A STATUSES  # issue_number -> exit code

run_batch() {
  local -a batch=("$@")
  local -a started=()

  for n in "${batch[@]}"; do
    if $DRY_RUN; then
      info "[dry-run] Would process issue #$n"
      continue
    fi
    create_worktree "$n"
    start_container "$n"
    started+=("$n")
  done

  # Wait for all containers in this batch
  for n in "${started[@]}"; do
    code=$(wait_container "$n")
    STATUSES[$n]=$code
  done
}

# Process issues in batches of MAX_PARALLEL
total=${#ISSUES[@]}
for (( i=0; i<total; i+=MAX_PARALLEL )); do
  batch=("${ISSUES[@]:$i:$MAX_PARALLEL}")
  run_batch "${batch[@]}"
done

###############################################################################
# Summary
###############################################################################
if $DRY_RUN; then
  info "Dry run complete. No containers were started."
  exit 0
fi

echo ""
echo "======================================"
echo " Results"
echo "======================================"
for n in "${ISSUES[@]}"; do
  code=${STATUSES[$n]:-"skipped"}
  if [[ "$code" == "0" ]]; then
    pr=$(gh pr list --head "claude/issue-$n" --state open --json number,url --jq '.[0].url' 2>/dev/null || true)
    if [[ -n "$pr" ]]; then
      echo -e "[issue-$n] ${GREEN}OK — PR opened: $pr${NC}"
    else
      echo -e "[issue-$n] ${GREEN}OK — done (no new PR, may already exist)${NC}"
    fi
  elif [[ "$code" == "timeout" ]]; then
    echo -e "[issue-$n] ${YELLOW}TIMEOUT — killed after ${TIMEOUT}s, see $LOGS_DIR/issue-$n.log${NC}"
  elif [[ "$code" == "skipped" ]]; then
    echo -e "[issue-$n] ${YELLOW}SKIPPED (dry-run)${NC}"
  else
    echo -e "[issue-$n] ${RED}FAILED (exit $code) — see $LOGS_DIR/issue-$n.log${NC}"
  fi
done
echo "======================================"
