---
name: version-update-checker
description: 自动检查并更新 hr-ai-data 插件版本。每天最多执行一次。当 data-permission-checker、data-warehouse-api-codegen、hr-data-sql-builder 被调用时，应先执行本 Skill。
---

## 前置约定

在执行任何步骤之前，必须先确定以下两项：

### 1. IDE 配置目录

根据**当前 SKILL 文件自身所在的路径**判断：

- 路径包含 `.workbuddy` → IDE 配置目录为 `~/.workbuddy`（macOS/Linux）或 `$env:USERPROFILE\.workbuddy`（Windows）
- 路径包含 `.codebuddy` → IDE 配置目录为 `~/.codebuddy`（macOS/Linux）或 `$env:USERPROFILE\.codebuddy`（Windows）

> ⚠️ **禁止默认使用 `.codebuddy`**，必须根据实际路径判断。

以下所有步骤中的 **`<IDE_DIR>`** 均指此处判断出的目录。

### 2. 操作系统

- **macOS / Linux**：使用 bash/zsh，用户目录为 `~`
- **Windows**：使用 PowerShell，用户目录为 `$env:USERPROFILE`

> 通过 `<user_info>` 中的 OS 信息判断。以下步骤均提供两套脚本，按操作系统选择执行。

## 执行流程

### Step 1：检查是否需要执行版本检查

1. 读取记忆信息，看今天是否已经执行过版本更新。如果已执行过，则跳过后续所有步骤。
2. 读取 `<IDE_DIR>/hr-ai-data.ver` 文件，获取当前版本号（日期格式，如 `2026-04-16`）
3. 判断逻辑：
   - 文件不存在或内容为空 → 将当前版本号设为空字符串，继续执行 Step 2
   - 文件存在且有内容 → 读取内容作为当前版本号，继续执行 Step 2

### Step 2：调用 MCP 工具检查版本

调用 `hr_data_service` MCP 的 `check_version` 工具，传入参数：
- `version`：Step 1 中读取到的当前版本号（可能为空字符串）

**返回结果格式**：
```json
{
  "hasNewVersion": true/false,
  "latestVersion": "2026-04-16"
}
```

**异常处理**：工具不存在、返回错误、网络超时 → **静默终止本 Skill**，不影响调用方 Skill。

### Step 3：根据检查结果决定是否更新

- `hasNewVersion` 为 `false` → 跳转到 Step 5
- `hasNewVersion` 为 `true` → 继续执行 Step 4

### Step 4：执行版本更新

在 `<IDE_DIR>/plugins/marketplaces/` 下查找插件目录并执行 `git pull`。由于仓库迁移，目录名不固定，通过 `git remote` 匹配查找。

**macOS / Linux：**

```bash
PLUGIN_DIR=""; for dir in <IDE_DIR>/plugins/marketplaces/*/; do if [ -d "${dir}.git" ]; then remote=$(git -C "$dir" remote get-url origin 2>/dev/null); if echo "$remote" | grep -qE "(cnb\.cool/tencent-hrssc/hrit-codebuddy|cnb\.woa\.com/hrssc/codebuddy)"; then PLUGIN_DIR="$dir"; break; fi; fi; done; if [ -n "$PLUGIN_DIR" ]; then cd "$PLUGIN_DIR" && git remote set-url origin https://cnb.woa.com/hrssc/codebuddy && git pull; else echo "未找到插件目录"; fi
```

**Windows（PowerShell）：**

```powershell
$pluginDir = $null; $base = "<IDE_DIR>\plugins\marketplaces"; if (Test-Path $base) { Get-ChildItem -Path $base -Directory | ForEach-Object { if (Test-Path "$($_.FullName)\.git") { $remote = git -C $_.FullName remote get-url origin 2>$null; if ($remote -match "(cnb\.cool/tencent-hrssc/hrit-codebuddy|cnb\.woa\.com/hrssc/codebuddy)") { $pluginDir = $_.FullName } } } }; if ($pluginDir) { Set-Location $pluginDir; git remote set-url origin https://cnb.woa.com/hrssc/codebuddy; git pull } else { Write-Host "未找到插件目录" }
```

> ⚠️ 执行前将 `<IDE_DIR>` 替换为前置约定中判断出的实际路径。

- 成功 → 简要提示用户插件已更新
- 失败 → 提示更新失败但不影响当前操作
- 未找到目录 → 静默跳过

### Step 5：记录版本号

> 仅当 Step 2 成功完成（未报错）时才执行本步骤。如果 Step 2 异常终止，则不记录。

1. 将 Step 2 返回的 `latestVersion` 写入 `<IDE_DIR>/hr-ai-data.ver`。

**macOS / Linux：**

```bash
mkdir -p <IDE_DIR> && echo "<LATEST_VERSION>" > <IDE_DIR>/hr-ai-data.ver
```

**Windows（PowerShell）：**

```powershell
New-Item -ItemType Directory -Force -Path "<IDE_DIR>" | Out-Null; "<LATEST_VERSION>" | Out-File -FilePath "<IDE_DIR>\hr-ai-data.ver" -Encoding utf8 -Force
```

> ⚠️ 执行前将 `<IDE_DIR>` 替换为前置约定中判断出的实际路径，将 `<LATEST_VERSION>` 替换为 Step 2 返回的 `latestVersion` 值。

2. 在用户记忆中记录今日已执行数仓插件版本更新检查，记录当前版本号。

## 注意事项

1. **容错优先**：任何环节出错都应优雅降级，绝不能因为版本检查失败而阻断调用方 Skill 的正常工作流
2. **静默执行**：仅在成功更新时简要通知用户，其余情况不打扰
3. **每日一次**：通过记忆信息控制每天只执行一次版本检查；`hr-ai-data.ver` 存储当前版本号用于与服务端比对
4. **禁止硬编码目录名**：插件目录名因仓库 URL 不同而不同，必须通过 `git remote` 动态查找
5. **IDE 目录跟随实际路径**：`hr-ai-data.ver` 写入当前 IDE 对应的目录，WorkBuddy 写 `.workbuddy`，CodeBuddy 写 `.codebuddy`