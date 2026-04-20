# HR 数仓数据集成指南

当用户需求涉及 HR 数据（员工信息、组织架构、人员变动、绩效、招聘等）时，**必须**使用 `data-warehouse-api-codegen` Skill 生成数仓接口调用代码，禁止自行编写接口调用逻辑。

---

## 前置依赖

| 插件 | 用途 | 检查方式 |
|:---|:---|:---|
| **`HRIT Marketplace/hr-ai-data/hr_data_service`** | 数仓 SQL 查询、数据权限校验、用户身份获取、HR 术语查询 | CodeBuddy MCP 设置中确认已安装且状态为"已启用" |

> ⚠️ 若 `hr-ai-data` 插件未安装或未启用，Skill 调用将失败。安装指南：[HR AI Data MCP 插件安装教程](https://km.woa.com/articles/show/654881?ts=1773723655)

插件正常安装后，`hr-data-sql-builder` 和 `data-warehouse-api-codegen` 两个 Skill 会随之自动可用。

---

## ⚠️ 关键约束：跨域凭证（必须）

数仓接口依赖用户的 SSO 登录态（OA 认证），前端调用时**必须携带跨域凭证**，否则返回 401。

| 方式 | 关键配置 |
|------|----------|
| `fetch` | `credentials: 'include'` |
| `axios` | `withCredentials: true` |

```javascript
// fetch
const resp = await fetch(url, {
  method: 'POST',
  credentials: 'include',  // ← 必须
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ sql: 'SELECT ...' }),
});

// axios
const resp = await axios.post(url, { sql: 'SELECT ...' }, {
  withCredentials: true,  // ← 必须
});
```

> 💡 `credentials: 'include'` 让浏览器在跨域请求中自动携带 SSO Cookie。缺少此配置 = 匿名访问 = 401。

---

## 常见问题排查

| # | 典型错误 | 检查内容 | 修复方式 |
|---|----------|----------|----------|
| 0 | Skill 不存在 / MCP 工具调用失败 | `hr-ai-data` 插件是否已安装并启用 | [安装教程](https://km.woa.com/articles/show/654881?ts=1773723655) |
| 1 | 401 Unauthorized / "未登录" | `credentials: 'include'` 或 `withCredentials: true` 是否设置 | 补充跨域凭证配置 |
| 2 | 401（凭证已设置） | 用户 SSO Cookie 是否过期 | 引导用户访问 OA 登录页重新认证 |
| 3 | 404 / CORS error | 是否使用 `data-warehouse-api-codegen` 生成的标准接口地址 | 重新用 Skill 生成接口代码 |
| 4 | 后端报"无 SSO 身份" | 是否在后端代码中调用了数仓接口 | ⚠️ 数仓接口**只能在前端/浏览器端调用**，移至前端 |
