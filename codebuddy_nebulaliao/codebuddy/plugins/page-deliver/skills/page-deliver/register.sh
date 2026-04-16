#!/bin/bash
set -e

# ========================================
# Page Deliver - Gateway Register 注册路由脚本
# ========================================
# 功能: 向 Deliver Gateway 注册项目路由，获取 OA 认证的公开 URL
# 调用方: page-deliver Skill Step 5
# 执行环境: ⚠️ 必须在 AnyDev 环境内通过 webshell 执行！
# ========================================

DELIVER_API="http://21.91.240.52:8080"

# ========================================
# 参数校验
# ========================================
if [ -z "$project_id" ]; then
  echo "❌ 错误: 缺少必需参数 project_id"
  exit 1
fi

if [ -z "$staff_id" ]; then
  echo "❌ 错误: 缺少必需参数 staff_id"
  exit 1
fi

if [ -z "$staff_name" ]; then
  echo "⚠️  警告: 缺少 staff_name，将使用 staff_id 作为 staff_name"
  staff_name="$staff_id"
fi

if [ -z "$host" ]; then
  echo "❌ 错误: 缺少必需参数 host"
  exit 1
fi

if [ -z "$port" ]; then
  echo "❌ 错误: 缺少必需参数 port"
  exit 1
fi

if [ -z "$project_type" ]; then
  echo "❌ 错误: 缺少必需参数 project_type"
  exit 1
fi

# ========================================
# 打印注册信息
# ========================================
echo "🚀 开始注册到 Deliver Gateway..."
echo ""
echo "📋 注册信息："
echo "   - 项目 ID:   ${project_id}"
echo "   - 操作者:    ${staff_name} (工号: ${staff_id})"
echo "   - 后端地址:  http://${host}:${port}"
echo "   - 项目类型:  ${project_type}"
echo ""

# ========================================
# 健康检查（可选，但推荐）
# ========================================
echo "🔍 验证后端服务是否可达..."
health_check=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://${host}:${port}/" || echo "000")

if [ "$health_check" = "000" ]; then
  echo "⚠️  警告: 无法连接到后端服务 http://${host}:${port}"
  echo "   - 请确认服务已启动"
  echo "   - 检查端口是否正确"
  echo "   - 查看日志: tail -f /tmp/${project_id}.log"
  echo ""
  read -p "是否继续注册？(y/n) " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消注册"
    exit 1
  fi
else
  echo "✅ 后端服务可达 (HTTP ${health_check})"
  echo ""
fi

# ========================================
# 调用注册 API
# ========================================
echo "📡 正在调用 Deliver API..."

response=$(curl -s -w "\n%{http_code}" --max-time 10 \
  -X POST "${DELIVER_API}/api/projects/register" \
  -H "Content-Type: application/json" \
  -d "{
    \"project_id\":   \"${project_id}\",
    \"staff_id\":     \"${staff_id}\",
    \"staff_name\":   \"${staff_name}\",
    \"host\":         \"${host}\",
    \"port\":         ${port},
    \"project_type\": \"${project_type}\"
  }")

http_code=$(echo "$response" | tail -1)
body=$(echo "$response" | head -1)

# ========================================
# 处理注册结果
# ========================================
if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
  # API 返回格式：{"code":0,"data":{"url":"https://hrai.app.hrainative.woa.com/codebuddy-app-detail/xxx"},"message":null}
  # 尝试从响应中提取 URL
  access_url=$(echo "$body" | grep -o '"url":"[^"]*"' | cut -d'"' -f4)

  # 如果无法提取，使用默认格式
  if [ -z "$access_url" ]; then
    access_url="https://hrai.app.hrainative.woa.com/codebuddy-app-detail/${project_id}"
  fi

  echo ""
  echo "════════════════════════════════════════════════"
  echo "✅ Gateway 注册成功！"
  echo "════════════════════════════════════════════════"
  echo ""
  echo "📎 访问地址："
  echo "   ${access_url}"
  echo ""
  echo "🔐 访问方式："
  echo "   使用企业微信扫码或 OA 账号登录"
  echo ""
  echo "📍 内网直连（调试用）："
  echo "   http://${host}:${port}"
  echo ""
  echo "💡 提示："
  echo "   域名解析可能需要几分钟生效，请稍后重试"
  echo ""
  echo "════════════════════════════════════════════════"

  # 输出 JSON 格式结果（便于 AI 解析）
  echo ""
  echo "<!-- REGISTER_RESULT_JSON"
  echo "{
    \"success\": true,
    \"access_url\": \"${access_url}\",
    \"direct_url\": \"http://${host}:${port}\",
    \"project_id\": \"${project_id}\",
    \"http_code\": ${http_code}
  }"
  echo "REGISTER_RESULT_JSON -->"

  exit 0

else
  echo ""
  echo "════════════════════════════════════════════════"
  echo "❌ Gateway 注册失败"
  echo "════════════════════════════════════════════════"
  echo ""
  echo "🔍 失败原因："
  echo "   HTTP ${http_code}"
  echo ""
  echo "📋 错误详情："
  echo "   ${body}"
  echo ""
  echo "💡 可能的原因："
  echo "   - Gateway 服务不可达 (${DELIVER_API})"
  echo "   - 项目 ID 已存在（尝试使用不同的 project_id）"
  echo "   - 后端服务未启动或不可达"
  echo "   - 网络连接问题"
  echo ""
  echo "🛠️ 解决方案："
  echo "   1. 检查 Gateway 服务状态："
  echo "      curl -I ${DELIVER_API}/health"
  echo ""
  echo "   2. 验证后端服务可达："
  echo "      curl -I http://${host}:${port}"
  echo ""
  echo "   3. 稍后重试注册"
  echo ""
  echo "⚠️ 您仍然可以通过内网 IP 访问应用："
  echo "   http://${host}:${port}"
  echo ""
  echo "════════════════════════════════════════════════"

  # 输出 JSON 格式错误结果
  echo ""
  echo "<!-- REGISTER_RESULT_JSON"
  echo "{
    \"success\": false,
    \"error\": \"HTTP ${http_code}: ${body}\",
    \"http_code\": ${http_code},
    \"direct_url\": \"http://${host}:${port}\"
  }"
  echo "REGISTER_RESULT_JSON -->"

  exit 1
fi
