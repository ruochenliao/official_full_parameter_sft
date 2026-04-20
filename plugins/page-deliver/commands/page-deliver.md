---
description: 一键部署项目到 Page Deliver 平台，获得 OA 认证的访问 URL
argument-hint: 可选：项目描述或部署参数（如 --name my-app --db mysql）
---

# Page Deliver 部署

你正在帮助用户将项目部署到 Page Deliver 平台。

## 前置要求

**必须按顺序执行以下操作**：

1. **加载 Skill**：使用 **page-deliver**  获取完整部署工作流
2. **加载身份规范**：如涉及前端代码生成，读取 page-deliver中的 `user-identity.md`
3. **加载数仓集成**：如涉及 HR 数据，读取 page-deliver 中 `references/data-warehouse-integration.md`

## 用户输入

$ARGUMENTS

## 执行

**严格按照 SKILL.md 定义的 6 个 Step 执行**，使用 `todo_write` 创建 Step 1–6 的进度跟踪。

详细流程见 SKILL.md，此处不再重复。

## ⛔ 完成条件（强制检查）

> **部署（Step 4）≠ 完成。必须完成全部 6 步才算完成。**

在你认为任务"完成"之前，**必须逐项自检**：

| ✅ 检查项 | 说明 |
|-----------|------|
| Step 5 Gateway 注册是否已执行？ | `deploy.sh` **不包含**注册逻辑，必须单独执行 `register.sh` 或 curl |
| Step 6 结果是否已输出给用户？ | 必须展示访问地址 `https://hrai.app.hrainative.woa.com/codebuddy-app-detail/{project_id}` |
| todo 列表中 Step 5、Step 6 是否标记为 completed？ | 未标记 = 未完成，必须继续执行 |

**如果任何一项检查未通过，禁止结束任务，必须继续执行缺失的步骤。**

## 关键提醒

- 使用 `deploy.sh` 脚本部署，**不要**调用已废弃的 `anydev-deploy-full` 或 `anydev-deploy-quick`
- `deploy.sh`、`register.sh` 等脚本**必须在 AnyDev 环境内执行**
- **`deploy.sh` 不包含 Gateway 注册逻辑** — Step 5 必须单独执行 `register.sh`
- Step 4 部署成功后**不要停下来总结**，必须立即继续 Step 5 和 Step 6
- 整个流程应自动化执行，尽量减少用户交互
- 如果某个 Step 失败，提供清晰的错误信息和回退方案
