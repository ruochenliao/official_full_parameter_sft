#!/bin/bash
set -e

# ========================================
# Page Deliver - Gateway Query 查询路由脚本
# ========================================
# 功能: 查询项目在 Deliver Gateway 的路由信息
# 执行环境: ⚠️ 必须在 AnyDev 环境内通过 webshell 执行！
# ========================================

DELIVER_API="http://21.91.240.52:8080"

# ========================================
# 参数校验
# ========================================
if [ -z "$project_id" ]; then
  echo "❌ 错误: 缺少必需参数 project_id"
  echo ""
  echo "使用方法:"
  echo "  export project_id='your-project-id'"
  echo "  bash query.sh"
  exit 1
fi

# ========================================
# 查询路由信息
# ========================================
echo "🔍 正在查询项目路由信息..."
echo "   - 项目 ID: ${project_id}"
echo ""

response=$(curl -s -w "\n%{http_code}" --max-time 10 \
  "${DELIVER_API}/internal/projects/${project_id}")

http_code=$(echo "$response" | tail -1)
body=$(echo "$response" | head -1)

# ========================================
# 处理查询结果
# ========================================
if [ "$http_code" = "200" ]; then
  echo "✅ 查询成功"
  echo ""
  echo "📋 路由信息："
  echo "${body}" | jq '.' 2>/dev/null || echo "${body}"

  # 尝试解析并美化输出
  if command -v jq &> /dev/null; then
    host=$(echo "${body}" | jq -r '.host' 2>/dev/null)
    port=$(echo "${body}" | jq -r '.port' 2>/dev/null)

    if [ -n "$host" ] && [ "$host" != "null" ]; then
      echo ""
      echo "📎 访问地址："
      echo "   https://hrai.app.hrainative.woa.com/codebuddy-app-detail/${project_id}"
      echo ""
      echo "📍 后端地址："
      echo "   http://${host}:${port}"
    fi
  fi

  exit 0

else
  echo "❌ 查询失败 (HTTP ${http_code})"
  echo ""

  if [ "$http_code" = "404" ]; then
    echo "🔍 项目未找到: ${project_id}"
    echo ""
    echo "可能的原因："
    echo "  - 项目尚未注册"
    echo "  - 项目 ID 输入错误"
    echo "  - 项目已被删除"
  else
    echo "错误详情: ${body}"
  fi

  exit 1
fi
