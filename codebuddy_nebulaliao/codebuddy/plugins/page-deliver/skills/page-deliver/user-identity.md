# 用户身份规范（前端展示 + 后端读取）

---

## Part 1: 前端展示

### 数据来源

Gateway（Nginx）在每个 HTTP 响应中注入以下头：

| 响应头 | 说明 | 示例 |
|--------|------|------|
| `X-Staff-Name` | 用户英文名 | `leonqhe` |
| `X-Staff-Id` | 用户工号 | `12345` |

### 获取用户信息

```javascript
async function getStaffInfo() {
  try {
    const resp = await fetch(window.location.href, { method: 'HEAD' });
    const staffName = resp.headers.get('X-Staff-Name');
    const staffId = resp.headers.get('X-Staff-Id');
    if (!staffName) return null;
    return {
      staffName,
      staffId,
      avatarUrl: `https://r.hrc.woa.com/photo/150/${staffName}.png?default_when_absent=true`
    };
  } catch (e) {
    return null;
  }
}
```

### 展示要求

| 规则 | 说明 |
|------|------|
| **位置** | 页面顶部 Header/导航栏**右侧**（左侧放标题/Logo） |
| **内容** | 圆形头像（32px）+ 用户英文名 |
| **降级** | 获取不到用户信息时**静默隐藏**，不报错、不占位 |
| ✅ **必须** | 嵌入 Header 组件内，作为正常文档流一部分 |
| ✅ **必须** | Header 使用 `display: flex; justify-content: space-between; align-items: center` |
| ❌ **禁止** | 使用 `position: fixed/absolute` 浮层 — 会遮挡页面内容 |

### 布局结构

```
┌──────────────────────────────────────────────────┐
│  🏠 页面标题              [👤 头像] username     │  ← Header（正常文档流）
├──────────────────────────────────────────────────┤
│                                                  │
│              页面主内容区域                        │  ← Main（不会被遮挡）
│                                                  │
└──────────────────────────────────────────────────┘
```

> 💡 AI 应根据项目使用的技术栈（React / Vue / 纯 HTML 等）自行生成对应的组件实现，遵循上述规则即可。

### 头像 URL 拼接规则

```
https://r.hrc.woa.com/photo/150/{staffName}.png?default_when_absent=true
```

`?default_when_absent=true` 确保无照片时返回默认头像。

---

## Part 2: 后端读取

### 核心原则

**后端涉及用户相关逻辑时，一律从 HTTP 请求头获取用户身份，禁止任何其他方式。**

Gateway（Nginx）在转发请求前自动注入以下**请求头**（从 `x-tai-identity` JWE 令牌解密）：

| 请求头 | 说明 | 示例 |
|--------|------|------|
| `X-Staff-Id` | 用户工号 | `12345` |
| `X-Staff-Name` | 用户英文名 | `leonqhe` |

### ✅ 正确做法

```javascript
// 封装中间件，统一注入
function userIdentity(req, res, next) {
  req.staffId = req.headers['x-staff-id'] || null;
  req.staffName = req.headers['x-staff-name'] || null;
  next();
}
app.use(userIdentity);

// 业务代码直接使用
app.post('/api/save', (req, res) => {
  if (!req.staffId) return res.status(401).json({ error: '未获取到用户身份' });
  db.save({ ...req.body, createdBy: req.staffId });
});
```

> 💡 Python (Flask)、Java (Spring Boot) 等其他技术栈同理 — 从请求头读取，封装为中间件/拦截器。

### ❌ 禁止做法

| 禁止方式 | 原因 |
|----------|------|
| 从请求体 `req.body.userId` 获取 | 可被伪造 |
| 从查询参数 `req.query.user` 获取 | 可被伪造 |
| 硬编码用户 ID | 无法适应多用户 |
| 后端自行解密 `x-tai-identity` | 这是 Nginx 的职责 |

### 适用场景

数据存储（`created_by`）、权限判断、审计日志、数据隔离（"我的xxx"）、个性化内容 — 均必须使用请求头中的用户身份。

### 本地开发降级

```javascript
const staffId = req.headers['x-staff-id'] || (isDev ? 'dev-user' : null);
```

---

## ⚠️ 注意事项

- 用户身份信息**仅在 `*.app.hrainative.woa.com` 域名**下由 Gateway 注入
- 本地开发（`localhost`）无用户身份头，前端和后端均需做降级处理
