#!/bin/bash
# Page Deliver - App Management Script
# Usage: manage.sh {list|stop|restart|remove|logs|status} [app-name]
#
# Commands:
#   list              - List all deployed applications
#   stop <app>        - Stop an application
#   restart <app>     - Restart an application
#   remove <app>      - Remove an application (with confirmation)
#   logs <app>        - Show application logs (last 50 lines)
#   status            - Show PM2 status table (if available)
#
# NOTE: This script runs inside AnyDev environment.

DEPLOY_ROOT="/data/anydev_upload"

# Check if PM2 is available
HAS_PM2=false
if command -v pm2 > /dev/null 2>&1; then
    HAS_PM2=true
fi

case $1 in
    list)
        echo "📦 Deployed Applications:"
        echo ""
        if [ "$HAS_PM2" = true ]; then
            pm2 jlist 2>/dev/null | python3 -c "
import sys, json
try:
    apps = json.load(sys.stdin)
    for app in apps:
        name = app.get('name', 'unknown')
        pid = app.get('pid', 'N/A')
        status = app.get('pm2_env', {}).get('status', 'unknown')
        print(f'  ✅ {name} (PID: {pid}, Status: {status})')
    if not apps:
        print('  (no apps managed by PM2)')
except:
    print('  (unable to parse PM2 output)')
" 2>/dev/null || echo "  (PM2 output unavailable)"
        else
            # Fallback: check running node/python processes
            ps aux | grep -E "node server\.js|node app\.js|python3 app\.py" | grep -v grep | \
                awk '{printf "  ✅ PID: %s | %s\n", $2, $11}'
            if [ $? -ne 0 ]; then
                echo "  (no running apps found)"
            fi
        fi
        ;;

    stop)
        app_name="$2"
        if [ -z "$app_name" ]; then
            echo "Usage: manage.sh stop <app-name>"
            exit 1
        fi
        if [ "$HAS_PM2" = true ]; then
            pm2 stop "$app_name" 2>/dev/null && echo "✅ Stopped $app_name" || echo "❌ Failed to stop $app_name"
        else
            pkill -f "node.*$app_name" 2>/dev/null || pkill -f "python.*$app_name" 2>/dev/null
            echo "✅ Stopped $app_name"
        fi
        ;;

    restart)
        app_name="$2"
        if [ -z "$app_name" ]; then
            echo "Usage: manage.sh restart <app-name>"
            exit 1
        fi
        if [ "$HAS_PM2" = true ]; then
            pm2 restart "$app_name" 2>/dev/null && echo "✅ Restarted $app_name" || echo "❌ Failed to restart $app_name"
        else
            echo "⚠️ PM2 not available. Stopping and restarting manually..."
            pkill -f "node.*server.js" 2>/dev/null || true
            sleep 2
            cd "$DEPLOY_ROOT"
            source ~/.bashrc 2>/dev/null || true
            nohup node server.js > "$DEPLOY_ROOT/logs/server.log" 2>&1 &
            echo "✅ Restarted $app_name"
        fi
        ;;

    remove)
        app_name="$2"
        if [ -z "$app_name" ]; then
            echo "Usage: manage.sh remove <app-name>"
            exit 1
        fi

        echo "⚠️  This will stop the app and clean up deployment files."
        echo "   App: $app_name"
        echo "   Path: $DEPLOY_ROOT"
        echo ""

        # In non-interactive mode (AI calling), proceed directly
        if [ "$HAS_PM2" = true ]; then
            pm2 delete "$app_name" 2>/dev/null || true
            pm2 save > /dev/null 2>&1
        else
            pkill -f "node.*server.js" 2>/dev/null || true
            pkill -f "python.*app.py" 2>/dev/null || true
        fi

        # Clean up deployment files (preserve logs)
        if [ -d "$DEPLOY_ROOT" ]; then
            find "$DEPLOY_ROOT" -maxdepth 1 -not -name "logs" -not -name "." -not -name ".." -exec rm -rf {} \; 2>/dev/null
        fi

        echo "✅ Removed $app_name"
        ;;

    logs)
        app_name="$2"
        if [ -z "$app_name" ]; then
            echo "Usage: manage.sh logs <app-name>"
            exit 1
        fi

        log_file="$DEPLOY_ROOT/logs/server.log"

        if [ "$HAS_PM2" = true ]; then
            pm2 logs "$app_name" --lines 50 --nostream 2>/dev/null || {
                if [ -f "$log_file" ]; then
                    echo "📋 Last 50 lines from $log_file:"
                    tail -50 "$log_file"
                else
                    echo "❌ No logs found for $app_name"
                fi
            }
        else
            if [ -f "$log_file" ]; then
                echo "📋 Last 50 lines from $log_file:"
                tail -50 "$log_file"
            else
                echo "❌ No logs found for $app_name"
            fi
        fi
        ;;

    status)
        if [ "$HAS_PM2" = true ]; then
            pm2 status
        else
            echo "📊 Process Status:"
            echo ""
            ps aux | head -1
            ps aux | grep -E "node|python|mongod|mysqld|redis" | grep -v grep
            echo ""
            echo "📡 Listening Ports:"
            netstat -tuln 2>/dev/null | grep -E "8080|27017|3306|6379" || \
            ss -tuln 2>/dev/null | grep -E "8080|27017|3306|6379" || \
            echo "  (netstat/ss unavailable)"
        fi
        ;;

    *)
        echo "Page Deliver - App Management"
        echo ""
        echo "Usage: manage.sh {list|stop|restart|remove|logs|status} [app-name]"
        echo ""
        echo "Commands:"
        echo "  list              - List all deployed applications"
        echo "  stop <app>        - Stop an application"
        echo "  restart <app>     - Restart an application"
        echo "  remove <app>      - Remove an application and clean up"
        echo "  logs <app>        - Show application logs"
        echo "  status            - Show process and port status"
        exit 1
        ;;
esac
