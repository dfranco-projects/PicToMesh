#!/bin/sh
# commit-msg hook (via pre-commit): reject AI co-author trailers.
# A "Co-Authored-By: Claude <noreply@anthropic.com>" line makes GitHub list "claude" as a
# contributor, and the only way back is a history rewrite plus a support ticket.
if grep -qiE '^(Co-Authored-By:.*(claude|anthropic)|Claude-Session:)' "$1"; then
    echo "commit message carries a Claude trailer, remove it:" >&2
    grep -iE '^(Co-Authored-By:.*(claude|anthropic)|Claude-Session:)' "$1" >&2
    exit 1
fi
