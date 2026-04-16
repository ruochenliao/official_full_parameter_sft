---
name: page-deliver
description: "Deploy AI-generated web applications to AnyDev with automatic Gateway registration. AI generates code, this skill handles deployment infrastructure. Automatically provides OA-authenticated public URLs."
---

# Page Deliver — 端到端部署 Skill (v1.3)

> **核心理念**：AI 负责生成代码，Skill 脚本负责部署基础设施。两者职责分离。

---

## ⚠️ 全局约束（必读）

| # | 约束 | 说明 |
|---|------|------|
| 1 | **AnyDev 环境内执行** | `deploy.sh`、`register.sh`、`query.sh`、`manage.sh` 必须在 AnyDev 环境内通过 webshell 执行，**不能在本地执行** |
| 2 | **禁止废弃 API** | 不得调用 `anydev-deploy-full` 或 `anydev-deploy-quick`（已废弃），统一使用 `deploy.sh` |
| 3 | **禁止硬编码身份** | `staff_id` / `staff_name` 必须运行时动态获取（见 Step 6） |
| 4 | **默认数据库 MongoDB** | 需要持久化时默认使用 MongoDB（`mongoose`），除非用户明确指定其他数据库 |
| 5 | **代码生成前读规范** | 写任何前端代码前必须先读 [`user-identity.md`](./user-identity.md) |
| 6 | **HR 数据集成** | 涉及 HR 数据需求时必须参阅 [`references/data-warehouse-integration.md`](./references/data-warehouse-integration.md) |

---

## 📖 部署工作流（6 步）

> ⛔ **关键约束：6 步全部完成才算部署成功。Step 4 部署完成后，必须继续执行 Step 5（Gateway 注册）和 Step 6（输出结果），不得跳过或遗漏。**

> 开始前，使用 `todo_write` 创建 Step 1–6 的进度跟踪。

### Step 1: 准备工作区

```bash
WORKSPACE=$(bash .codebuddy/skills/page-deliver/prepare-workspace.sh "app-name")
```

脚本在当前工作目录下创建 `<app-name>/` 子目录。

| 场景 | 行为 |
|------|------|
| 新项目（目录不存在或为空） | 创建空目录 → 继续 Step 2 |
| 已有项目（目录已有文件，输出 `ℹ️ Existing project detected`） | 复用目录 → **跳过 Step 2**，直接到 Step 3 |

### Step 2: AI 生成应用代码

> ⚡ 若 Step 1 检测到已有项目，**跳过此步**。

**前置动作（必须按顺序完成）**：
1. 读取 [`user-identity.md`](./user-identity.md)，实现顶部导航栏右侧用户徽章（头像 + 英文名）
2. 涉及 HR 数据时，读取 [`references/data-warehouse-integration.md`](./references/data-warehouse-integration.md)
3. 涉及 HR 数据取数时，**必须使用 `data-warehouse-api-codegen` skill 生成调用代码**，禁止自行编写数仓接口调用逻辑

> ⚠️ **生成代码前强制自检（缺一不可）**：
>
> | # | 检查项 | 通过标准 |
> |---|--------|----------|
> | 1 | 用户头像 | 页面包含 Header 组件，右侧展示圆形头像(32px) + 英文名；头像 URL 使用 `https://r.hrc.woa.com/photo/150/{staffName}.png?default_when_absent=true` |
> | 2 | Header 布局 | 使用 `display: flex`，头像在右侧；**禁止** `position: fixed/absolute` 浮层 |
> | 3 | HR 数仓调用方式 | 使用 `data-warehouse-api-codegen` skill 生成的代码，**不可自行编写** |
> | 4 | 数仓仅限前端调用 | 数仓接口只能在前端/浏览器端（`public/` 下的 JS）调用，**严禁在 server.js 等后端代码中调用**（后端没有用户 SSO Cookie，必定 401） |
> | 5 | 跨域凭证 | 所有数仓 fetch 包含 `credentials: 'include'`；axios 包含 `withCredentials: true` |

**代码规范**：

| 规则 | 要求 |
|------|------|
| 入口文件 | `server.js`（Node.js）或 `app.py` / `main.py`（Python） |
| 监听地址 | `0.0.0.0`（不要用 `localhost`） |
| 端口 | `process.env.PORT \|\| 8080` |
| 依赖声明 | 所有 npm 包必须在 `package.json` 中声明 |
| 数据库 | 默认 MongoDB：`process.env.MONGO_URI \|\| 'mongodb://127.0.0.1:27017/<app-name>'` |

**必须生成的文件**：

```
<app-name>/
├── server.js        # 入口（必须）
├── package.json     # 依赖声明（必须）
├── public/          # 静态资源（按需）
│   ├── index.html
│   ├── style.css
│   └── app.js
└── ...              # 其他文件（按需）
```

### Step 3: 本地验证 + 代码合规检查

部署前必须通过本地验证和代码合规检查，**任一不通过不得继续部署**。

#### 3a. 运行 local-verify 脚本

```bash
bash .codebuddy/skills/page-deliver/local-verify.sh "$WORKSPACE"
```

验证项：✅ 入口文件存在 → 📦 依赖安装成功 → 🧪 语法检查通过 → 🚀 服务能启动并响应 HTTP

输出 JSON（`status: "passed"` 才可继续）。失败时修复代码后重跑验证。

#### 3b. 代码合规检查 ⚠️（不可跳过）

> local-verify 通过后，逐项检查生成的代码，确认以下约束项全部满足。如有不满足项，**立即修复代码**后重新确认。

| # | 检查项 | 通过标准 | 不通过时修复方式 |
|---|--------|----------|------------------|
| 1 | 用户头像 | 页面包含 Header 组件，右侧展示圆形头像(32px) + 英文名；头像 URL 使用 `https://r.hrc.woa.com/photo/150/{staffName}.png?default_when_absent=true` | 按 [`user-identity.md`](./user-identity.md) 补充头像实现 |
| 2 | Header 布局 | 使用 `display: flex`，头像在右侧；**没有**使用 `position: fixed/absolute` 浮层 | 改为 flex 布局，移除浮层定位 |
| 3 | HR 数仓调用方式 | 使用 `data-warehouse-api-codegen` skill 生成的代码，**不是自行编写的** | 删除自行编写的数仓调用代码，调用 `data-warehouse-api-codegen` skill 重新生成 |
| 4 | 数仓仅限前端调用 | 数仓接口调用代码**只存在于前端文件**（`public/` 下的 JS/HTML），`server.js`、`app.js`、`routes/` 等后端文件中**不包含**数仓 API 调用 | 将数仓调用从后端移至前端（后端没有用户 SSO Cookie，必定 401） |
| 5 | 跨域凭证 | 所有数仓 `fetch` 包含 `credentials: 'include'`；所有 `axios` 包含 `withCredentials: true` | 补充凭证配置（缺少 = 匿名访问 = 401） |

> ⚠️ 第 3~5 项仅在涉及 HR 数据需求时检查。无数仓调用的项目跳过 3~5。

### Step 4: 部署到 AnyDev

#### 4a. 路由选择

根据用户需求关键词决定部署模式：

| 关键词 | 模式 | 含义 |
|--------|------|------|
| 管理系统 / 后台 / CRM / 数据库… | `full` | 带数据库部署 |
| Demo / 测试 / 静态 / 计算器… | `quick` | 不带数据库 |
| 留言板 / 博客… | `ask` | 询问用户选择 |

#### 4b. AnyDev 环境准备

**新项目检测**（满足任一即为新项目）：
- 用户意图为"生成/创建/做一个"
- 当前工作区无 `package.json` 且无 `requirements.txt`
- 无 `.anydev-deployed` 标记文件

```javascript
// 新项目 → 弹出环境选择（一次性）
if (isNewProject) {
  await call_integration("anydev", "select_environment", "{}");
}
// 已有项目 → 直接部署
```

#### 4c. 执行部署脚本

```bash
bash .codebuddy/skills/page-deliver/deploy.sh \
  --name "app-name" \
  --src-dir "$WORKSPACE" \
  --mode "full"          # 或 "quick"
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--name` | 应用名（必须） | — |
| `--src-dir` | 代码目录路径（必须） | — |
| `--mode` | `full`（带 DB）/ `quick`（无 DB） | `full` |
| `--port` | 端口号（可选） | 自动分配 3000-3999 |
| `--db` | `mongodb` / `mysql` / `none` | 自动检测，默认 `mongodb` |

脚本自动完成：代码上传 → 数据库配置 → 依赖安装 → 启动应用 → 健康检查，输出 JSON 结果。

> ⚠️ **deploy.sh 完成 ≠ 部署流程结束！`deploy.sh` 不包含 Gateway 注册。必须立即继续 Step 5 执行注册，否则用户将无法通过公网 URL 访问应用。不要在此处停下来总结或输出结果。**

### Step 5: Gateway 注册 ⚠️（必须执行，不可跳过）

> ⚠️ **必须在 AnyDev 环境内执行，不能在本地执行！**

#### 5a. 获取用户身份

**按优先级依次尝试**：

| 优先级 | 方式 | 字段映射 |
|--------|------|----------|
| 1️⃣ | `hr_data_service.get_current_user({})` | `staffId` → `staff_id`，`loginName` → `staff_name` |
| 2️⃣ | `gongfeng.get_current_user({})` | `id` → `staff_id`（转 string），`username` → `staff_name` |
| 3️⃣ | `ask_followup_question` 手动输入 | 用户输入英文名 → `staff_name`，`staff_id = "0"` |

> ⛔ **禁止**使用 `whoami` 降级 — 系统用户名可能与企业账号不一致。

#### 5b. 获取 / 生成 project_id

**优先复用已有 project_id**：
```javascript
// 1. 检查项目目录下的 .anydev-deployed 标记文件
const markerFile = path.join(WORKSPACE, '.anydev-deployed');
if (fs.existsSync(markerFile)) {
  project_id = fs.readFileSync(markerFile, 'utf-8').trim();
  // ✅ 复用已有 project_id，保持域名不变
} else {
  // 2. 首次部署才生成新 project_id
  project_id = `${name}-${yyyyMMdd}-${HHmmss}`;
}
```

| 场景 | 行为 |
|------|------|
| `.anydev-deployed` 存在且非空 | **复用**其中的 project_id（保持域名不变） |
| `.anydev-deployed` 不存在或为空 | **生成新的** project_id，格式: `{name}-{yyyyMMdd}-{HHmmss}` |

```
生成规则: 小写 + 连字符，≤63 字符（DNS 限制）
示例: todo-app-20260305-143022
```

> ⚠️ **重要**：注册成功后，必须将 project_id 写入 `.anydev-deployed` 标记文件（本地 + AnyDev 两端），确保后续重新部署时能复用。

#### 5c. 执行注册

**方式一：register.sh（推荐）**
```bash
export project_id="app-name-20260309-143022"
export staff_id="${staff_id}"
export staff_name="${staff_name}"
export host="21.6.57.8"      # deploy.sh 输出的 anydev_host
export port="8080"            # deploy.sh 输出的 port
export project_type="node"
bash .codebuddy/skills/page-deliver/register.sh
```

**方式二：直接 curl**
```bash
curl -X POST http://21.91.240.52:8080/api/projects/register \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"${project_id}","staff_id":"${staff_id}","staff_name":"${staff_name}","host":"${host}","port":${port},"project_type":"${project_type}"}'
```

**查询已注册路由**：
```bash
export project_id="app-name-20260309-143022"
bash .codebuddy/skills/page-deliver/query.sh
```

### Step 6: 输出结果 ⚠️（必须执行，不可跳过）

> ⚠️ **此步为最终步骤。完成 Step 5 后必须立即执行 Step 6，向用户展示访问地址和部署信息。没有 Step 6 的输出，用户无法知道如何访问应用。**

根据部署和注册结果，向用户展示：

```markdown
✅ **部署成功！**
════════════════════════════════════════════════

📎 **访问地址**
https://hrai.app.hrainative.woa.com/codebuddy-app-detail/{project_id}

🔐 **访问方式**
使用企业微信扫码或 OA 账号登录即可访问

📍 **内网直连**（开发调试用）
http://{anydev_host}:{port}

⚙️ **部署信息**
- 项目 ID：{project_id}
- 项目类型：{project_type}
- 数据库：{database_type}（如有）

════════════════════════════════════════════════
```

| 状态 | 输出内容 |
|------|----------|
| ✅ 全部成功 | 访问地址 + 内网直连 + 管理命令 |
| ⚠️ 部分成功（注册失败） | 内网直连地址 + 手动注册提示 |
| ❌ 失败 | 错误信息 + 排查建议 |

---

## 🚨 完成检查清单（强制执行）

> **在结束任务前，必须逐项确认以下检查全部通过。任何一项未通过 = 任务未完成。**

| # | 检查项 | 通过标准 |
|---|--------|----------|
| 1 | 用户头像已实现 | `public/` 中包含 `r.hrc.woa.com/photo` 头像 URL 引用，Header 右侧展示圆形头像 + 英文名 |
| 2 | HR 数仓调用合规 | 使用 `data-warehouse-api-codegen` skill 生成的代码，前端调用且携带 `credentials: 'include'` |
| 3 | Step 3b 代码合规检查已通过 | Step 3b 的 5 项检查全部 ✅（无数仓调用时 3~5 跳过） |
| 4 | Step 5 Gateway 注册已执行 | `register.sh` 或 curl 返回成功 |
| 5 | `.anydev-deployed` 标记已写入 | 本地 + AnyDev 两端都有 project_id |
| 6 | Step 6 结果已输出给用户 | 用户可以看到 `https://hrai.app.hrainative.woa.com/codebuddy-app-detail/{project_id}` 访问地址 |
| 7 | todo 中 Step 5 和 Step 6 已标记 completed | `todo_write` 状态更新完毕 |


---

## 🔗 相关资源

| 类型 | 资源 | 说明 |
|------|------|------|
| 规范 | [`user-identity.md`](./user-identity.md) | 前端用户身份展示 + 后端用户身份读取规范 |
| 规范 | [`references/data-warehouse-integration.md`](./references/data-warehouse-integration.md) | HR 数仓数据集成指南 |
| Skill | `hr-data-sql-builder` | 生成 HR 数仓查询 SQL |
| Skill | `data-warehouse-api-codegen` | 生成前端数仓接口调用代码（⚠️ 仅限浏览器端） |
| API | `http://21.91.240.52:8080` | Gateway — `POST /api/projects/register`、`GET /internal/projects/{project_id}` |

---

**Skill Version**: v1.3.0 | **Updated**: 2026-03-24
