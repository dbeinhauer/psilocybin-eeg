#!/bin/bash
# Stop all running psilocybin Claude containers
docker ps --filter "name=psilocybin-issue-" --format "{{.Names}}" | xargs -r docker stop
echo "All Claude issue containers stopped."
