#!/bin/bash
# Page Deliver - Workspace Preparation Script
# Usage: ./prepare-workspace.sh <app-name> [base-dir]
# Creates a subdirectory under the current workspace for AI code generation.
# If the directory already contains code (existing project), it is reused as-is.
#
# Arguments:
#   app-name  - Application name (will be normalized)
#   base-dir  - Optional: base directory (defaults to current working directory,
#               which is typically the CodeBuddy workspace root)

APP_NAME="$1"
BASE_DIR="${2:-.}"

if [ -z "$APP_NAME" ]; then
    echo '{"status": "error", "message": "Usage: ./prepare-workspace.sh <app-name> [base-dir]"}' >&2
    exit 1
fi

# Normalize app name (lowercase, alphanumeric + hyphens only)
APP_NAME=$(echo "$APP_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g' | sed 's/^-\+\|-\+$//g' | cut -c1-50)

# Resolve base directory to absolute path
BASE_DIR=$(cd "$BASE_DIR" 2>/dev/null && pwd)
if [ $? -ne 0 ]; then
    echo '{"status": "error", "message": "Base directory does not exist: '"$2"'"}' >&2
    exit 1
fi

WORKSPACE_DIR="${BASE_DIR}/${APP_NAME}"

# If directory already exists and contains files → existing project, reuse it
if [ -d "$WORKSPACE_DIR" ] && [ "$(ls -A "$WORKSPACE_DIR" 2>/dev/null)" ]; then
    echo "ℹ️  Existing project detected at ${WORKSPACE_DIR}, reusing as-is." >&2
    echo "$WORKSPACE_DIR"
    exit 0
fi

# New project: create workspace root (subdirectories are created by AI during code generation)
mkdir -p "$WORKSPACE_DIR"

# Output workspace path (for use by AI)
echo "$WORKSPACE_DIR"
