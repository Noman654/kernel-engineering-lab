#!/usr/bin/env bash
# Daily logger. Usage: ./log.sh        (full 60-minute session)
#                      ./log.sh min    (15-minute minimum session)
set -euo pipefail
cd "$(dirname "$0")"

kind="${1:-full}"
today=$(date +%F)
count=$(grep -c '^## Session ' worklog.md || true)
n=$((count + 1))

ask() {  # ask "Prompt" varname required(1/0)
  local answer=""
  while :; do
    read -rp "$1: " answer
    if [[ -n "$answer" || "$3" == "0" ]]; then break; fi
    echo "  (required)"
  done
  printf -v "$2" '%s' "$answer"
}

echo "Session $n — $today ($kind)"
ask "Did (one topic)"                          did 1
ask "Result (prediction vs reality)"           result 1
ask "Evidence (numbers / test / file)"         evidence 0
ask "Lesson (in my control? attention leak?)"  lesson 0
ask "Next (tomorrow's FIRST action)"           next 1

cat >> worklog.md <<EOF

## Session $n — $today ($kind)
- **Did:** $did
- **Result:** $result
- **Evidence:** ${evidence:--}
- **Lesson:** ${lesson:--}
- **Next:** $next
EOF

cat > NEXT.md <<EOF
# NEXT

> Open this. Do it. No decisions.

**Session $((n + 1)) — first action:**
$next

Plan reference: [PLAN.md](PLAN.md). When done: \`./log.sh\`
EOF

sed -i -E "s|^(Sessions logged: \*\*)[0-9]+|\1$n|" README.md

git add -A
git commit -q -m "Session $n ($today): $did"
echo "Committed session $n."
if git remote get-url origin >/dev/null 2>&1; then
  git push -q || echo "Push failed — run 'git push' later."
fi
