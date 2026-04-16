#!/bin/bash
# Page Deliver - Local Verification Script
# Usage: ./local-verify.sh <src-dir> [--port PORT] [--timeout SECONDS]
#
# Verifies AI-generated code BEFORE deploying to AnyDev:
#   1. Check entry file exists
#   2. Install dependencies (npm install / pip install)
#   3. Syntax check (node --check / python -m py_compile)
#   4. Start server & health check (curl localhost)
#   5. Auto-cleanup (kill test server)
#
# Exit code 0 = all checks passed, safe to deploy.
# Exit code 1 = verification failed, fix before deploying.

set -e

# ========================================
# Parse arguments
# ========================================
SRC_DIR=""
PORT="9876"        # Use a high port for local test to avoid conflicts
TIMEOUT=10         # Max seconds to wait for server startup

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)    PORT="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        -*)        echo "{\"status\": \"error\", \"message\": \"Unknown option: $1\"}" >&2; exit 1 ;;
        *)         SRC_DIR="$1"; shift ;;
    esac
done

if [ -z "$SRC_DIR" ]; then
    echo '{"status": "error", "message": "Usage: ./local-verify.sh <src-dir> [--port PORT] [--timeout SECONDS]"}' >&2
    exit 1
fi

if [ ! -d "$SRC_DIR" ]; then
    echo '{"status": "error", "message": "Source directory does not exist: '"$SRC_DIR"'"}' >&2
    exit 1
fi

cd "$SRC_DIR"

ERRORS=0
WARNINGS=0

echo "🔍 Local Verification: $SRC_DIR" >&2
echo "════════════════════════════════════════════════" >&2

# ========================================
# Step 1: Detect project type & entry file
# ========================================
echo "" >&2
echo "📄 Step 1: Checking entry file..." >&2

ENTRY_FILE=""
PROJECT_TYPE=""

if [ -f "server.js" ]; then
    ENTRY_FILE="server.js"; PROJECT_TYPE="node"
elif [ -f "app.js" ]; then
    ENTRY_FILE="app.js"; PROJECT_TYPE="node"
elif [ -f "app.py" ]; then
    ENTRY_FILE="app.py"; PROJECT_TYPE="python"
elif [ -f "main.py" ]; then
    ENTRY_FILE="main.py"; PROJECT_TYPE="python"
elif [ -f "index.html" ]; then
    ENTRY_FILE="index.html"; PROJECT_TYPE="static"
else
    echo "   ❌ No entry file found (server.js/app.js/app.py/main.py/index.html)" >&2
    ERRORS=$((ERRORS + 1))
fi

if [ -n "$ENTRY_FILE" ]; then
    echo "   ✅ Entry file: $ENTRY_FILE (type: $PROJECT_TYPE)" >&2
fi

# ========================================
# Step 2: Install dependencies
# ========================================
echo "" >&2
echo "📦 Step 2: Installing dependencies..." >&2

if [ "$PROJECT_TYPE" = "node" ]; then
    if [ -f "package.json" ]; then
        if npm install --production 2>&1 | tail -5 >&2; then
            echo "   ✅ npm install succeeded" >&2
        else
            echo "   ❌ npm install failed" >&2
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo "   ⚠️  No package.json found (entry file is Node.js but no dependencies declared)" >&2
        WARNINGS=$((WARNINGS + 1))
    fi
elif [ "$PROJECT_TYPE" = "python" ]; then
    if [ -f "requirements.txt" ]; then
        if pip3 install -r requirements.txt 2>&1 | tail -5 >&2; then
            echo "   ✅ pip install succeeded" >&2
        else
            echo "   ❌ pip install failed" >&2
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo "   ⚠️  No requirements.txt found" >&2
        WARNINGS=$((WARNINGS + 1))
    fi
elif [ "$PROJECT_TYPE" = "static" ]; then
    echo "   ℹ️  Static project, no dependencies to install" >&2
fi

# ========================================
# Step 3: Syntax check
# ========================================
echo "" >&2
echo "🧪 Step 3: Syntax check..." >&2

if [ "$PROJECT_TYPE" = "node" ]; then
    # Check all .js files
    JS_ERRORS=0
    for js_file in $(find . -name "*.js" -not -path "./node_modules/*" -not -path "./.git/*" 2>/dev/null); do
        if ! node --check "$js_file" 2>/dev/null; then
            echo "   ❌ Syntax error in: $js_file" >&2
            node --check "$js_file" 2>&1 | head -5 >&2
            JS_ERRORS=$((JS_ERRORS + 1))
        fi
    done
    if [ "$JS_ERRORS" -eq 0 ]; then
        echo "   ✅ All .js files passed syntax check" >&2
    else
        echo "   ❌ $JS_ERRORS file(s) have syntax errors" >&2
        ERRORS=$((ERRORS + JS_ERRORS))
    fi

elif [ "$PROJECT_TYPE" = "python" ]; then
    # Check all .py files
    PY_ERRORS=0
    for py_file in $(find . -name "*.py" -not -path "./.venv/*" -not -path "./.git/*" 2>/dev/null); do
        if ! python3 -m py_compile "$py_file" 2>/dev/null; then
            echo "   ❌ Syntax error in: $py_file" >&2
            python3 -m py_compile "$py_file" 2>&1 | head -5 >&2
            PY_ERRORS=$((PY_ERRORS + 1))
        fi
    done
    if [ "$PY_ERRORS" -eq 0 ]; then
        echo "   ✅ All .py files passed syntax check" >&2
    else
        echo "   ❌ $PY_ERRORS file(s) have syntax errors" >&2
        ERRORS=$((ERRORS + PY_ERRORS))
    fi

elif [ "$PROJECT_TYPE" = "static" ]; then
    echo "   ℹ️  Static project, skipping syntax check" >&2
fi

# ========================================
# Step 4: Start server & health check
# ========================================
echo "" >&2
echo "🚀 Step 4: Start server & health check (port $PORT)..." >&2

# Only do startup test if no errors so far and we have a server entry
if [ "$ERRORS" -gt 0 ]; then
    echo "   ⏭️  Skipping startup test (previous errors detected)" >&2
elif [ "$PROJECT_TYPE" = "static" ]; then
    echo "   ℹ️  Static project, skipping server startup test" >&2
else
    # Kill any existing process on test port
    lsof -ti:${PORT} 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 1

    # Start server in background
    SERVER_PID=""
    if [ "$PROJECT_TYPE" = "node" ]; then
        PORT=$PORT node "$ENTRY_FILE" > /tmp/local-verify-server.log 2>&1 &
        SERVER_PID=$!
    elif [ "$PROJECT_TYPE" = "python" ]; then
        PORT=$PORT python3 "$ENTRY_FILE" > /tmp/local-verify-server.log 2>&1 &
        SERVER_PID=$!
    fi

    if [ -n "$SERVER_PID" ]; then
        echo "   ⏳ Waiting for server to start (PID: $SERVER_PID, timeout: ${TIMEOUT}s)..." >&2

        # Wait for server to be ready
        STARTED=false
        for i in $(seq 1 $TIMEOUT); do
            # Check if process is still alive
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "   ❌ Server process crashed during startup" >&2
                echo "   📋 Last 10 lines of log:" >&2
                tail -10 /tmp/local-verify-server.log 2>/dev/null >&2
                ERRORS=$((ERRORS + 1))
                break
            fi

            # Try health check
            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:${PORT}/" 2>/dev/null || echo "000")
            if [ "$HTTP_CODE" != "000" ]; then
                STARTED=true
                echo "   ✅ Server started successfully (HTTP $HTTP_CODE in ${i}s)" >&2
                break
            fi
            sleep 1
        done

        if [ "$STARTED" = false ] && kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "   ❌ Server did not respond within ${TIMEOUT}s" >&2
            echo "   📋 Last 10 lines of log:" >&2
            tail -10 /tmp/local-verify-server.log 2>/dev/null >&2
            ERRORS=$((ERRORS + 1))
        fi

        # Cleanup: kill test server
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        # Also clean up any child processes on the port
        lsof -ti:${PORT} 2>/dev/null | xargs kill -9 2>/dev/null || true
        rm -f /tmp/local-verify-server.log
        echo "   🧹 Test server cleaned up" >&2
    fi
fi

# ========================================
# Summary
# ========================================
echo "" >&2
echo "════════════════════════════════════════════════" >&2

if [ "$ERRORS" -eq 0 ] && [ "$WARNINGS" -eq 0 ]; then
    echo "✅ All checks passed — ready to deploy!" >&2
elif [ "$ERRORS" -eq 0 ]; then
    echo "⚠️  Passed with $WARNINGS warning(s) — can proceed to deploy" >&2
else
    echo "❌ Verification failed: $ERRORS error(s), $WARNINGS warning(s)" >&2
    echo "   Fix the errors above before deploying." >&2
fi

echo "════════════════════════════════════════════════" >&2

# Output JSON result to stdout
cat << JSON
{
  "status": "$([ "$ERRORS" -eq 0 ] && echo 'passed' || echo 'failed')",
  "errors": $ERRORS,
  "warnings": $WARNINGS,
  "project_type": "$PROJECT_TYPE",
  "entry_file": "$ENTRY_FILE",
  "src_dir": "$SRC_DIR"
}
JSON

# Exit with error if checks failed
if [ "$ERRORS" -gt 0 ]; then
    exit 1
fi
