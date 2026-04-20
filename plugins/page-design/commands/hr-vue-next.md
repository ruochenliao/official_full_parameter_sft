---
description: 使用 HR-Vue-Next 组件库快速设计和生成 HR 业务页面（Vue 3 + TDesign + HR 专属组件）
argument-hint: 页面描述，如"员工管理列表页"、"组织架构选择表单"、"包含员工选择器的审批页面"
---

# HR 业务页面设计

你正在帮助用户使用 `@tencent/hr-vue-next` 组件库设计和生成 HR 业务页面。请严格按照 `hr-vue-next` Skill 的规范执行。

> **技术栈**：Vue 3.x + TDesign 设计规范 + HR-Vue-Next 组件库（20 个企业级组件）

## 前置要求

在开始任何操作之前，**必须先执行**：

1. **加载 hr-vue-next Skill**：读取 `hr-vue-next/SKILL.md` 获取完整的组件库使用指南
2. **分析用户需求**：根据用户描述确定需要使用的组件和页面布局
3. **查阅组件文档**：根据需要读取 `hr-vue-next/assets/` 目录下对应的组件详细文档

## 用户输入

$ARGUMENTS

## 可用组件

### 人事组件（12 个）

| 组件 | 标签 | 说明 |
|------|------|------|
| HrStaffSelector | `<hr-staff-selector>` | 员工选择器 |
| HrUnitSelector | `<hr-unit-selector>` | 组织选择器 |
| HrPostSelector | `<hr-post-selector>` | 岗位选择器 |
| HrDictSelector | `<hr-dict-selector>` | 基础字典选择器 |
| HrContractCompanySelector | `<hr-contract-company-selector>` | 合同公司选择器 |
| HrAreaSelector | `<hr-area-selector>` | 工作地选择器 |
| HrCitySelector | `<hr-city-selector>` | 省市选择器 |
| HrManageSubjectSelector | `<hr-manage-subject-selector>` | 管理主体选择器 |
| HrOfficeBuildingSelector | `<hr-office-building-selector>` | 办公大厦选择器 |
| HrPositionCascader | `<hr-position-cascader>` | 职位级联选择器 |
| HrPositionLevel | `<hr-position-level>` | 职级选择器 |
| HrStaffSubtypeSelector | `<hr-staff-subtype-selector>` | 员工子类型选择器 |

### 通用组件（8 个）

| 组件 | 标签 | 说明 |
|------|------|------|
| HrAvatar | `<hr-avatar>` | 头像组件 |
| HrLangSwitch | `<hr-lang-switch>` | 语言切换 |
| HrWecom | `<hr-wecom>` | 企微组件 |
| HrPrivacySwitch | `<hr-privacy-switch>` | 隐私开关 |
| HrCountdownButton | `<hr-countdown-button>` | 倒计时按钮 |
| HrTextOverflow | `<hr-text-overflow>` | 文本溢出 |
| HrPageFooter | `<hr-page-footer>` | 页脚组件 |
| HrPageHeader | `<hr-page-header>` | 页头组件 |

## 执行流程

### Step 1: 需求分析

根据用户描述，确定：
- 页面类型（列表页 / 表单页 / 详情页 / 仪表盘 / 其他）
- 需要使用的 HR 组件（从上方组件列表中选择）
- 页面布局结构（是否需要页头、页脚、侧边栏等）
- 交互逻辑（表单提交、数据筛选、列表操作等）

### Step 2: 查阅组件文档

根据 Step 1 确定的组件列表，**逐一读取**对应的组件详细文档：

```
hr-vue-next/assets/hr-staff-selector.md    # 员工选择器详情
hr-vue-next/assets/hr-unit-selector.md     # 组织选择器详情
hr-vue-next/assets/hr-page-header.md       # 页头组件详情
...（根据实际需要读取）
```

确保了解每个组件的：
- 全部 Props（属性）
- Events（事件）
- Slots（插槽）
- 实例方法（如 setSelected、clearSelected）

### Step 3: 生成页面代码

**代码生成规范**：

1. **引入顺序必须严格遵守**：Vue 3 → TDesign CSS → TDesign JS → HR-Vue-Next CSS → HR-Vue-Next JS
2. **注册顺序**：`.use(TDesign).use(window.HrVueNext)`
3. **环境变量**：必须设置 `window.HR_BUILD_ENV = 'prd'`
4. **使用 UMD 方式引入**（适用于无构建工具的场景）：

```html
<!-- 依赖引入（顺序严格） -->
<script src="https://unpkg.com/vue@3.5.27/dist/vue.global.js"></script>
<link rel="stylesheet" href="https://unpkg.com/tdesign-vue-next@1.18.0/dist/tdesign.min.css">
<script src="https://unpkg.com/tdesign-vue-next@1.18.0/dist/tdesign.min.js"></script>
<link rel="stylesheet" href="https://cdn.m.tencent.com/hr-web/hr-vue-next.min.css">
<script src="https://cdn.m.tencent.com/hr-web/hr-vue-next.min.js"></script>
```

5. **使用 Composition API**（`setup()` 函数）编写逻辑
6. **使用 TDesign 组件**搭配 HR-Vue-Next 组件构建完整页面
7. **页面必须包含**：合理的布局、美观的样式、响应式设计
8. **初始值设置**：使用 `setSelected` 方法，不要直接修改 `v-model`

### Step 4: 页面结构优化

确保生成的页面包含以下最佳实践：

- ✅ 使用 `<hr-page-header>` 作为页头（如适用）
- ✅ 使用 `<hr-page-footer>` 作为页脚（如适用）
- ✅ 表单使用 TDesign 的 `<t-form>` 进行布局和校验
- ✅ 表格使用 TDesign 的 `<t-table>` 展示数据
- ✅ 合理使用 TDesign 的布局组件（`<t-row>`、`<t-col>`、`<t-space>` 等）
- ✅ 添加适当的加载状态和空状态处理
- ✅ 页面样式美观、间距合理、配色统一

### Step 5: 交付结果

1. 输出完整的 HTML 文件（可直接在浏览器中打开）
2. 对页面结构和使用的组件进行简要说明
3. 列出使用到的 HR-Vue-Next 组件及其关键配置
4. 如有复杂交互逻辑，补充说明数据流转和事件处理方式

## 注意事项

- 整个流程应尽量自动化，减少用户交互
- 如果用户描述模糊，主动推荐合适的组件和页面布局
- 优先使用 HR-Vue-Next 组件替代手动实现（如用 `<hr-staff-selector>` 而非自定义员工搜索）
- 生成的代码应可直接运行，无需额外配置
- 遵循 TDesign 设计规范，保持视觉一致性
- 如需查阅特定组件的完整 API，读取 `hr-vue-next/assets/` 下的对应文档
