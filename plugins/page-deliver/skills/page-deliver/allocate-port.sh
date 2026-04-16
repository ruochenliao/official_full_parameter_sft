#!/bin/bash
# Page Deliver - Port Allocation Script
# Usage: ./allocate-port.sh
# Scans port range 3000-3999 on the AnyDev environment and returns the first available port.
#
# NOTE: This script is meant to be executed inside AnyDev environment via webshell.
#
# Output (stdout): available port number (e.g. "3000")
# Exit code: 0 = found, 1 = no available port

PORT_MIN=3000
PORT_MAX=3999

# Collect all ports currently in use (LISTEN state)
# Works with both old-style and new-style netstat, plus ss fallback
if command -v ss > /dev/null 2>&1; then
    USED_PORTS=$(ss -tlnH 2>/dev/null | awk '{print $4}' | grep -oE '[0-9]+$' | sort -un)
elif command -v netstat > /dev/null 2>&1; then
    USED_PORTS=$(netstat -tln 2>/dev/null | awk 'NR>2 {print $4}' | grep -oE '[0-9]+$' | sort -un)
else
    # Last resort: try /proc/net/tcp (Linux)
    if [ -f /proc/net/tcp ]; then
        USED_PORTS=$(awk 'NR>1 {print $2}' /proc/net/tcp 2>/dev/null \
            | cut -d: -f2 \
            | while read hex; do printf "%d\n" "0x$hex" 2>/dev/null; done \
            | sort -un)
    else
        USED_PORTS=""
        echo "⚠️  Cannot detect used ports (no ss/netstat/proc), will try ports blindly" >&2
    fi
fi

# Build a set for O(1) lookup
declare -A USED_SET
for p in $USED_PORTS; do
    USED_SET[$p]=1
done

# Scan range and return first available
for (( port=PORT_MIN; port<=PORT_MAX; port++ )); do
    if [ -z "${USED_SET[$port]}" ]; then
        echo "$port"
        exit 0
    fi
done

echo '{"status": "error", "message": "No available port in range '"$PORT_MIN"'-'"$PORT_MAX"'"}' >&2
exit 1
