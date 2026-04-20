#!/bin/bash
# Page Deliver - Main Deployment Script
# Usage: ./deploy.sh --name app-name --src-dir /path/to/code [--port PORT] [--mode full|quick] [--db mongodb|mysql|none]
# Port: if omitted, auto-allocated from 3000-3999 via allocate-port.sh
#
# This script handles AnyDev deployment:
# 0. Environment check & auto-install (node/npm/pm2/python/pip/curl)
# 1. Validate source code
# 2. Package and upload to AnyDev
# 3. Install dependencies
# 4. Set up database (if full mode)
# 5. Start application
# 6. Health check
# 7. Output deployment info as JSON
#
# NOTE: This script is meant to be executed inside AnyDev environment via webshell.
#       The AI assistant should use call_integration("anydev", "webshell", ...) to run commands.

set -e

# ========================================
# Parse arguments
# ========================================
APP_NAME=""
SRC_DIR=""
PORT=""            # empty = auto-allocate from 3000-3999
MODE="full"       # full (with DB) or quick (no DB)
DB_TYPE="auto"     # mongodb, mysql, none, or auto (detect)
DEPLOY_ROOT="/data/anydev_upload"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)    APP_NAME="$2"; shift 2 ;;
        --src-dir) SRC_DIR="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --mode)    MODE="$2"; shift 2 ;;
        --db)      DB_TYPE="$2"; shift 2 ;;
        *) echo "{\"status\": \"error\", \"message\": \"Unknown option: $1\"}" >&2; exit 1 ;;
    esac
done

# ========================================
# Validate required parameters
# ========================================
if [ -z "$APP_NAME" ]; then
    echo '{"status": "error", "message": "--name is required"}' >&2
    exit 1
fi

if [ -z "$SRC_DIR" ]; then
    echo '{"status": "error", "message": "--src-dir is required (path to generated code)"}' >&2
    exit 1
fi

if [ ! -d "$SRC_DIR" ]; then
    echo '{"status": "error", "message": "Source directory does not exist: '"$SRC_DIR"'"}' >&2
    exit 1
fi

# Check for entry file
ENTRY_FILE=""
if [ -f "$SRC_DIR/server.js" ]; then
    ENTRY_FILE="server.js"
elif [ -f "$SRC_DIR/app.js" ]; then
    ENTRY_FILE="app.js"
elif [ -f "$SRC_DIR/app.py" ]; then
    ENTRY_FILE="app.py"
elif [ -f "$SRC_DIR/main.py" ]; then
    ENTRY_FILE="main.py"
else
    echo '{"status": "error", "message": "No entry file found (server.js/app.js/app.py/main.py)"}' >&2
    exit 1
fi

# Detect project type
PROJECT_TYPE="node"
if [[ "$ENTRY_FILE" == *.py ]]; then
    PROJECT_TYPE="python"
fi
if [ -f "$SRC_DIR/index.html" ] && [ ! -f "$SRC_DIR/package.json" ] && [ ! -f "$SRC_DIR/requirements.txt" ]; then
    PROJECT_TYPE="static"
fi

# ========================================
# Auto-detect database if mode=full and db=auto
# ========================================
if [ "$MODE" = "full" ] && [ "$DB_TYPE" = "auto" ]; then
    if [ -f "$SRC_DIR/package.json" ]; then
        if grep -qiE "mongoose|mongodb" "$SRC_DIR/package.json" 2>/dev/null; then
            DB_TYPE="mongodb"
        elif grep -qiE "mysql|mysql2|sequelize" "$SRC_DIR/package.json" 2>/dev/null; then
            DB_TYPE="mysql"
        else
            DB_TYPE="none"
        fi
    elif [ -f "$SRC_DIR/requirements.txt" ]; then
        if grep -qiE "pymongo|mongoengine" "$SRC_DIR/requirements.txt" 2>/dev/null; then
            DB_TYPE="mongodb"
        elif grep -qiE "pymysql|mysqlclient|sqlalchemy" "$SRC_DIR/requirements.txt" 2>/dev/null; then
            DB_TYPE="mysql"
        else
            DB_TYPE="none"
        fi
    else
        DB_TYPE="none"
    fi
fi

if [ "$MODE" = "quick" ]; then
    DB_TYPE="none"
fi

# ========================================
# Auto-allocate port if not specified
# ========================================
if [ -z "$PORT" ]; then
    echo "🔌 Auto-allocating port (range 3000-3999)..." >&2
    PORT=$(bash "$SCRIPT_DIR/allocate-port.sh" 2>/dev/null)
    if [ $? -ne 0 ] || [ -z "$PORT" ]; then
        echo '{"status": "error", "message": "Failed to allocate port: no available port in range 3000-3999"}' >&2
        exit 1
    fi
    echo "   ✅ Allocated port: $PORT" >&2
fi

# ========================================
# Environment check & auto-install
# ========================================
echo "🔍 Environment check..." >&2

ENV_ERRORS=0

# Source bashrc/nvm for node path
source ~/.bashrc 2>/dev/null || true
source ~/.nvm/nvm.sh 2>/dev/null || true

# --- Node.js ---
if [ "$PROJECT_TYPE" = "node" ] || [ "$PROJECT_TYPE" = "static" ]; then
    if ! command -v node > /dev/null 2>&1; then
        echo "   📥 Node.js not found, installing via nvm..." >&2
        if [ -s "$HOME/.nvm/nvm.sh" ]; then
            source "$HOME/.nvm/nvm.sh"
            nvm install --lts > /dev/null 2>&1 && nvm use --lts > /dev/null 2>&1
        else
            # Install nvm first, then node
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh 2>/dev/null | bash > /dev/null 2>&1
            export NVM_DIR="$HOME/.nvm"
            source "$NVM_DIR/nvm.sh" 2>/dev/null
            nvm install --lts > /dev/null 2>&1 && nvm use --lts > /dev/null 2>&1
        fi
        if command -v node > /dev/null 2>&1; then
            echo "   ✅ Node.js installed: $(node --version)" >&2
        else
            echo "   ❌ Node.js installation failed" >&2
            ENV_ERRORS=$((ENV_ERRORS + 1))
        fi
    else
        echo "   ✅ Node.js: $(node --version)" >&2
    fi

    # --- npm ---
    if ! command -v npm > /dev/null 2>&1; then
        echo "   ❌ npm not found (should come with Node.js)" >&2
        ENV_ERRORS=$((ENV_ERRORS + 1))
    else
        echo "   ✅ npm: $(npm --version)" >&2
    fi

    # --- PM2 (best-effort, not blocking) ---
    if ! command -v pm2 > /dev/null 2>&1; then
        echo "   📥 PM2 not found, installing globally..." >&2
        npm install -g pm2 > /dev/null 2>&1 || true
        if command -v pm2 > /dev/null 2>&1; then
            echo "   ✅ PM2 installed: $(pm2 --version)" >&2
        else
            echo "   ⚠️ PM2 install failed, will use nohup fallback" >&2
        fi
    else
        echo "   ✅ PM2: $(pm2 --version)" >&2
    fi
fi

# --- Python ---
if [ "$PROJECT_TYPE" = "python" ]; then
    if ! command -v python3 > /dev/null 2>&1; then
        echo "   📥 Python3 not found, installing..." >&2
        if command -v yum > /dev/null 2>&1; then
            yum install -y python3 python3-pip > /dev/null 2>&1
        elif command -v apt-get > /dev/null 2>&1; then
            apt-get update > /dev/null 2>&1 && apt-get install -y python3 python3-pip > /dev/null 2>&1
        fi
        if command -v python3 > /dev/null 2>&1; then
            echo "   ✅ Python3 installed: $(python3 --version)" >&2
        else
            echo "   ❌ Python3 installation failed" >&2
            ENV_ERRORS=$((ENV_ERRORS + 1))
        fi
    else
        echo "   ✅ Python3: $(python3 --version)" >&2
    fi

    if ! command -v pip3 > /dev/null 2>&1; then
        echo "   ⚠️ pip3 not found, trying python3 -m pip..." >&2
        if python3 -m pip --version > /dev/null 2>&1; then
            echo "   ✅ pip: $(python3 -m pip --version)" >&2
        else
            echo "   ❌ pip not available" >&2
            ENV_ERRORS=$((ENV_ERRORS + 1))
        fi
    else
        echo "   ✅ pip3: $(pip3 --version)" >&2
    fi
fi

# --- curl (needed for health check) ---
if ! command -v curl > /dev/null 2>&1; then
    echo "   📥 curl not found, installing..." >&2
    if command -v yum > /dev/null 2>&1; then
        yum install -y curl > /dev/null 2>&1
    elif command -v apt-get > /dev/null 2>&1; then
        apt-get install -y curl > /dev/null 2>&1
    fi
    if command -v curl > /dev/null 2>&1; then
        echo "   ✅ curl installed" >&2
    else
        echo "   ⚠️ curl not available, health check will be skipped" >&2
    fi
else
    echo "   ✅ curl: available" >&2
fi

# Abort if critical dependencies are missing
if [ "$ENV_ERRORS" -gt 0 ]; then
    echo "{\"status\": \"error\", \"message\": \"Environment check failed: ${ENV_ERRORS} critical dependency(ies) missing. Check logs above.\"}"
    exit 1
fi

echo "   ✅ Environment check passed" >&2
echo "" >&2

echo "🚀 Page Deliver - Deploying ${APP_NAME}" >&2
echo "   Mode: ${MODE} | DB: ${DB_TYPE} | Type: ${PROJECT_TYPE}" >&2
echo "" >&2

# ========================================
# Step 1: Prepare deployment directory
# ========================================
echo "📂 Step 1: Preparing deployment directory..." >&2
mkdir -p "$DEPLOY_ROOT"
mkdir -p "$DEPLOY_ROOT/logs"

# Copy source code to deployment root
cp -r "$SRC_DIR"/* "$DEPLOY_ROOT/" 2>/dev/null || true
cp -r "$SRC_DIR"/.* "$DEPLOY_ROOT/" 2>/dev/null || true

echo "   ✅ Code copied to $DEPLOY_ROOT" >&2

# ========================================
# Step 2: Database setup (full mode only)
# ========================================
DB_URI=""
if [ "$DB_TYPE" = "mongodb" ]; then
    echo "🗄️ Step 2: Setting up MongoDB..." >&2

    MONGO_BIN="/usr/local/mongodb/bin/mongod"
    DATA_DIR="/data/mongodb"

    # Install MongoDB if not present
    if [ ! -f "$MONGO_BIN" ]; then
        echo "   📥 Installing MongoDB 7.0.5..." >&2
        cd /tmp
        wget -q "https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-rhel80-7.0.5.tgz" -O mongodb.tgz
        tar -xzf mongodb.tgz
        mv mongodb-linux-x86_64-* /usr/local/mongodb
        rm mongodb.tgz
        echo "   ✅ MongoDB installed" >&2
    fi

    # Create data directories
    mkdir -p "$DATA_DIR"/{db,log}

    # Start MongoDB if not running
    if ! pgrep -x mongod > /dev/null; then
        nohup "$MONGO_BIN" \
            --dbpath "$DATA_DIR/db" \
            --logpath "$DATA_DIR/log/mongod.log" \
            --port 27017 \
            --bind_ip 0.0.0.0 \
            --fork > /tmp/mongod-start.log 2>&1 &
        sleep 3
    fi

    if pgrep -x mongod > /dev/null; then
        DB_URI="mongodb://127.0.0.1:27017/${APP_NAME}"
        echo "   ✅ MongoDB running on port 27017" >&2
    else
        echo "   ⚠️ MongoDB failed to start, continuing without DB" >&2
        DB_TYPE="none"
    fi

elif [ "$DB_TYPE" = "mysql" ]; then
    echo "🗄️ Step 2: Setting up MySQL..." >&2

    # Install MySQL if not present
    if ! command -v mysql > /dev/null 2>&1; then
        if command -v yum > /dev/null 2>&1; then
            yum install -y mariadb-server mariadb > /dev/null 2>&1
        elif command -v apt-get > /dev/null 2>&1; then
            DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server mysql-client > /dev/null 2>&1
        fi
    fi

    # Initialize and start if not running
    MYSQL_DATA="/data/mysql"
    mkdir -p "$MYSQL_DATA"/{data,log}

    if ! pgrep -x mysqld > /dev/null; then
        if [ ! -d "$MYSQL_DATA/data/mysql" ]; then
            mysqld --initialize-insecure --user=root --datadir="$MYSQL_DATA/data" 2>/dev/null || \
            mysql_install_db --user=root --datadir="$MYSQL_DATA/data" 2>/dev/null || true
        fi
        nohup mysqld \
            --datadir="$MYSQL_DATA/data" \
            --socket=/tmp/mysql.sock \
            --port=3306 \
            --bind-address=0.0.0.0 \
            --log-error="$MYSQL_DATA/log/error.log" \
            > /tmp/mysql-start.log 2>&1 &
        sleep 5
    fi

    if pgrep -x mysqld > /dev/null; then
        # Create database
        mysql -h 127.0.0.1 -P 3306 -e "CREATE DATABASE IF NOT EXISTS \`${APP_NAME//-/_}\`;" 2>/dev/null || true
        DB_URI="mysql://root@127.0.0.1:3306/${APP_NAME//-/_}"
        echo "   ✅ MySQL running on port 3306" >&2
    else
        echo "   ⚠️ MySQL failed to start, continuing without DB" >&2
        DB_TYPE="none"
    fi
else
    echo "📂 Step 2: No database needed (quick mode)" >&2
fi

# ========================================
# Step 3: Create environment config
# ========================================
echo "⚙️ Step 3: Creating environment config..." >&2

cat > "$DEPLOY_ROOT/.env" << ENVEOF
NODE_ENV=production
PORT=${PORT}
APP_NAME=${APP_NAME}
ENVEOF

if [ -n "$DB_URI" ]; then
    if [ "$DB_TYPE" = "mongodb" ]; then
        echo "MONGODB_URI=${DB_URI}" >> "$DEPLOY_ROOT/.env"
    elif [ "$DB_TYPE" = "mysql" ]; then
        echo "MYSQL_URI=${DB_URI}" >> "$DEPLOY_ROOT/.env"
    fi
fi

echo "   ✅ .env created" >&2

# ========================================
# Step 4: Install dependencies
# ========================================
echo "📦 Step 4: Installing dependencies..." >&2

cd "$DEPLOY_ROOT"

if [ "$PROJECT_TYPE" = "node" ] || [ "$PROJECT_TYPE" = "static" ]; then
    if [ -f "package.json" ]; then
        npm install --production > /dev/null 2>&1 || {
            echo '{"status": "error", "message": "Failed to install npm dependencies"}' >&2
            exit 1
        }
        echo "   ✅ npm dependencies installed" >&2
    fi
elif [ "$PROJECT_TYPE" = "python" ]; then
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt > /dev/null 2>&1 || {
            echo '{"status": "error", "message": "Failed to install Python dependencies"}' >&2
            exit 1
        }
        echo "   ✅ Python dependencies installed" >&2
    fi
fi

# ========================================
# Step 5: Stop existing app and start new
# ========================================
echo "🚀 Step 5: Starting application..." >&2

# Kill existing process on the port
lsof -ti:${PORT} 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

cd "$DEPLOY_ROOT"

if [ "$PROJECT_TYPE" = "node" ]; then
    # Try PM2 first (recommended)
    if command -v pm2 > /dev/null 2>&1; then
        pm2 delete "$APP_NAME" 2>/dev/null || true
        pm2 start "$ENTRY_FILE" \
            --name "$APP_NAME" \
            --time \
            --output "$DEPLOY_ROOT/logs/server.log" \
            --error "$DEPLOY_ROOT/logs/error.log" \
            > /dev/null 2>&1
        pm2 save > /dev/null 2>&1
        echo "   ✅ Started with PM2" >&2
    else
        # Fallback to nohup
        nohup node "$ENTRY_FILE" > "$DEPLOY_ROOT/logs/server.log" 2>&1 &
        echo "   ✅ Started with nohup" >&2
    fi

elif [ "$PROJECT_TYPE" = "python" ]; then
    if grep -qiE "fastapi|uvicorn" "$DEPLOY_ROOT/requirements.txt" 2>/dev/null; then
        nohup uvicorn main:app --host 0.0.0.0 --port "$PORT" > "$DEPLOY_ROOT/logs/server.log" 2>&1 &
    elif grep -qiE "flask" "$DEPLOY_ROOT/requirements.txt" 2>/dev/null; then
        nohup python3 "$ENTRY_FILE" > "$DEPLOY_ROOT/logs/server.log" 2>&1 &
    else
        nohup python3 "$ENTRY_FILE" > "$DEPLOY_ROOT/logs/server.log" 2>&1 &
    fi
    echo "   ✅ Started with nohup" >&2

elif [ "$PROJECT_TYPE" = "static" ]; then
    if command -v pm2 > /dev/null 2>&1 && [ -f "$ENTRY_FILE" ]; then
        pm2 delete "$APP_NAME" 2>/dev/null || true
        pm2 start "$ENTRY_FILE" --name "$APP_NAME" --time > /dev/null 2>&1
        pm2 save > /dev/null 2>&1
    elif [ -f "$ENTRY_FILE" ]; then
        nohup node "$ENTRY_FILE" > "$DEPLOY_ROOT/logs/server.log" 2>&1 &
    else
        nohup python3 -m http.server "$PORT" --directory "$DEPLOY_ROOT/public" > "$DEPLOY_ROOT/logs/server.log" 2>&1 &
    fi
    echo "   ✅ Static server started" >&2
fi

sleep 3

# ========================================
# Step 6: Health check
# ========================================
echo "🔍 Step 6: Health check..." >&2

HEALTH_STATUS="unknown"
HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://127.0.0.1:${PORT}/" 2>/dev/null || echo "000")

if [ "$HEALTH_CODE" = "200" ] || [ "$HEALTH_CODE" = "301" ] || [ "$HEALTH_CODE" = "302" ]; then
    HEALTH_STATUS="ok"
    echo "   ✅ Health check passed (HTTP ${HEALTH_CODE})" >&2
else
    HEALTH_STATUS="warning"
    echo "   ⚠️ Health check: HTTP ${HEALTH_CODE}" >&2
fi

# ========================================
# Get server IP
# ========================================
SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "unknown")

# ========================================
# Output result as JSON (stdout)
# ========================================
DB_JSON="null"
if [ "$DB_TYPE" = "mongodb" ] && [ -n "$DB_URI" ]; then
    DB_JSON="{\"type\": \"mongodb\", \"uri\": \"${DB_URI}\", \"port\": 27017}"
elif [ "$DB_TYPE" = "mysql" ] && [ -n "$DB_URI" ]; then
    DB_JSON="{\"type\": \"mysql\", \"uri\": \"${DB_URI}\", \"port\": 3306}"
fi

cat << JSON
{
  "status": "success",
  "app_name": "${APP_NAME}",
  "port": ${PORT},
  "project_type": "${PROJECT_TYPE}",
  "entry_file": "${ENTRY_FILE}",
  "mode": "${MODE}",
  "anydev_host": "${SERVER_IP}",
  "internal_url": "http://${SERVER_IP}:${PORT}",
  "deployment_path": "${DEPLOY_ROOT}",
  "log_file": "${DEPLOY_ROOT}/logs/server.log",
  "health_status": "${HEALTH_STATUS}",
  "database": ${DB_JSON},
  "deployment_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo "" >&2
echo "════════════════════════════════════════════════" >&2
echo "✅ Deployment complete: ${APP_NAME}" >&2
echo "   Internal URL: http://${SERVER_IP}:${PORT}" >&2
echo "════════════════════════════════════════════════" >&2
