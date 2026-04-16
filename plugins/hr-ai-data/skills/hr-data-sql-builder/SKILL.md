---
name: hr-data-sql-builder
description: 生成HR数仓StarRocks查询SQL。覆盖员工信息/人员异动/绩效/梯队等查询，含术语映射、业务规则和SQL模板。表结构从MCP resources动态获取。用户提出数据查询需求时必须使用本Skill。
---

## 前置步骤：版本检查

> 在执行本 Skill 的业务逻辑之前，**必须先加载并执行 `version-update-checker` Skill** 进行插件版本检查与更新。版本检查失败不影响本 Skill 的后续执行。

## 概述

根据用户HR数据需求，生成StarRocks SQL查询腾讯HR数仓宽表。

> ⚠️ **核心原则见 RULES**：本 Skill 遵循 `hr-starrocks-query-conventions` 规则（禁止权限控制类WHERE条件）和 `hr-sql-safety` 规则（仅允许SELECT、必须LIMIT、统计优先SQL完成）。

## 数据源

表结构**必须从MCP resources动态获取**，禁止硬编码。

### MCP服务：`hr_data_service`

**执行查询**：工具 `starrocks_query`，参数 `sql`（必填）+ `userQuestion`（必填）

**获取表结构**：
- 表列表：resource `starrocks://tables` → 获取 `table_code`/`table_name`/`table_desc`/`write_sql_background`/`default_parameters`
- 单表字段：resource `starrocks://tables/{table_code}` → 获取 `columns` 数组（含 `column_code`/`column_name`/`column_alias`/`column_type`/`column_use`/`column_group`/`sample`/`group_by_able`/`aggregate_type`）

**术语知识**：
- 术语清单：resource `starrocks://slangs` → 获取所有HR业务术语名称及同义词列表，用于识别用户问题中涉及的术语
- 术语定义查询：工具 `slang_query`，输入术语名称或同义词 → 返回匹配术语的完整定义（含术语名称、定义、分类、同义词）

### 选表策略

- 在职人数/员工现状/绩效/结构分布 → **员工信息宽表**
- 入职/离职/调动/晋升等异动 → **人员变动信息宽表**

---

## SQL生成工作流

### Step 1：术语识别与需求分析

1. **术语识别**（MCP优先，本地降级）：
   1. 从MCP resource `starrocks://slangs` 获取术语清单（含术语名称和同义词）
   2. 结合用户问题，推测哪些术语与用户意图相关（匹配关键词、简称、同义词）
   3. 使用MCP工具 `slang_query` 查询相关术语的完整定义，补充业务知识以准确理解用户意图
2. 确定：查询目标（统计/明细/趋势/对比/分布）、数据范围（组织/时间/人群）、分析维度
3. 根据选表策略，从MCP resources获取目标表字段定义

### Step 2：SQL构建
1. **SELECT**：统计类用聚合函数+GROUP BY字段；明细类用业务相关字段
2. **FROM**：选择正确的表
3. **WHERE**：默认过滤条件（从`default_parameters`获取）+ 组织条件（`org_full_name LIKE`）+ 业务条件。⚠️ 禁止添加权限控制条件；
4. 调用专业术语-指标口径背景知识生成sql时，注意遵循指标口径定义的可选条件默认值
5. **GROUP BY**：统计类必选
6. **ORDER BY**：按业务逻辑排序
7. **LIMIT**：大结果集默认限制行数

### Step 3：SQL校验清单

- [ ] 已从MCP获取表结构，表名含catalog前缀
- [ ] 默认过滤条件齐全
- [ ] 统计人数用 COUNT(DISTINCT staff_id8)
- [ ] 专业职级字段类型正确（字符串 vs 数字）
- [ ] 组织查询用 org_full_name + LIKE
- [ ] 异动查询指定 move_type_name
- [ ] 绩效等级码值正确（Outstanding/Good/Underperform）
- [ ] 大结果集有LIMIT（见 `hr-sql-safety`）
- [ ] 仅SELECT，禁止写操作（见 `hr-sql-safety`）
- [ ] 无权限控制类过滤条件（见 `hr-starrocks-query-conventions`）

### Step 4：输出SQL

---

## 业务规则参考

### 组织信息

- `org_full_name`：组织全路径（BG/线/部门/中心/组），WHERE查询组织优先用此字段 + LIKE
- `org_name`：末级组织节点名称，查单个组织节点时用
- BG/线/部门/中心/组：分层级字段，按层级分布统计时用对应字段GROUP BY
- 示例：xx线各部门在职人数 → `WHERE org_full_name LIKE '%xx线%' GROUP BY dept_name`

### 专业职级

- 专业人员：`pro_position_level_name IS NOT NULL AND manager_level_name IS NULL`
- x级专业人员：`pro_position_level_num = x`
- x族x级（如T9）：`pro_position_level_name = 'T9'`
- x级以上（带族如T9+）：`pro_position_level_name` IN 含T且数值>=9的值
- x级以上（不带族如9级+）：`pro_position_level_num >= 9`
- 职级分布GROUP BY优先用 `pro_position_level_num`

### 异动查询

- 类型映射：入职→`雇佣`、离职→`离职`、调动→`调动`、专业变化→`专业变化`、管理变化→`管理变化`
- A组织入职/离职/专业变化/管理变化：`to_org_full_name LIKE '%A组织%'`
- A组织调入：`to_org_full_name LIKE '%A组织%' AND from_org_full_name NOT LIKE '%A组织%'`
- A组织调出：`from_org_full_name LIKE '%A组织%' AND to_org_full_name NOT LIKE '%A组织%'`

---

## 安全约束

> 详细安全规范见 `hr-sql-safety` 规则和 `hr-starrocks-query-conventions` 规则，此处仅列出校验清单摘要。

1. 仅允许 SELECT（见 `hr-sql-safety`）
2. 大结果集必须加 LIMIT（见 `hr-sql-safety`）
3. 禁止权限控制类 WHERE 条件（见 `hr-starrocks-query-conventions`）

---

## 回答规范

执行查询后按以下顺序组织回答：
1. 简要说明需求理解和执行策略
2. 展示SQL/代码
3. 呈现结果（表格/列表）
4. 空结果时分析原因（见空结果处理规则）
5. 识别脱敏数据并提示（见脱敏识别规则）
6. 基于数据给出洞察
7. 不确定时明确告知并给出调整建议

---

## 数据脱敏识别规则

> 脱敏特征、识别方法和处理规范的完整定义见 `hr-data-desensitization` 规则。

执行查询后，按 `hr-data-desensitization` 规则检测结果中的脱敏数据。发现脱敏时在结果表格后提示用户，深入排查可用 `data-permission-checker` Skill。

---

## 查询结果为空的处理规则

返回0行数据时，主动分析原因而非简单告知"没有数据"：

**可能原因**：
1. **条件传值有误**：组织名拼写/简称错误、时间范围不对、枚举值不正确、筛选值不存在
2. **数据权限不足**：无该表/组织的查看权限，服务端返回空结果
3. **数据本身为空**：条件合理但确实无数据

**处理**：自查SQL条件 → 可放宽条件重试 → 向用户列出可能原因并提出调整建议

---

## 调用方式

- **直接查询**：通过MCP生成SQL并执行
- **生成调用代码**：生成SQL后参考 `data-warehouse-api-codegen` Skill 生成前端调用代码