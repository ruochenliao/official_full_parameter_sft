---
description: 智能HR数据助手，根据需求自动选择合适的数据处理策略（SQL查询/权限排查/API代码生成）
argument-hint: 故障描述，如"前端页面报错"、"fetch error"、"数据不更新”、"没有数据”
---

# 问题修复

## 用户输入
$ARGUMENTS

## 处理方案
1. 当用户说"前端页面报错"、"fetch error"等问题时，使用`data-warehouse-api-codegen` SKILL排查是否调用了正确的数仓接口地址，调用地址的代码是否写在前端JS代码中，是否设置了跨域凭证携带的参数。

2. 当用户说"没有数据"时：
   2.1 确认是否在前端代码中调用了数仓接口获取数据，并使用`data-warehouse-api-codegen` SKILL检查代码逻辑是否正确。
   2.2 使用`hr-data-sql-builder` SKILL调试SQL是否写错了过滤条件或使用了错误的码值，确认SQL修改正确后，更新代码中的SQL。
3. 当用户说"数据不更新"时：
   3.1. 检查数据是否硬编码在了代码中，如果是，将硬编码数据从代码移除，使用`data-warehouse-api-codegen` SKILL编写正确的调用数仓接口代码。
   3.2、检查前端代码中获取数据的sql语句是否硬编码了p_mm等日期值，如果是，使用`data-warehouse-api-codegen` SKILL修改为动态值。

4. 如果用户没给具体的问题描述，则按顺序将上述步骤一一执行。