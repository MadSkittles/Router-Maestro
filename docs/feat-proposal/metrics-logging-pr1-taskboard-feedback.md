# Metrics + Logging PR1 Taskboard —— 评审反馈

> 评审范围：`docs/feat-proposal/metrics-logging-pr1-taskboard.md`
> 评审基准：`docs/feat-proposal/metrics-logging-first-pr.md`（已通过三轮评审的提案契约）
> 评审结论：整体结构扎实，14 个 work item 与 PR1 白名单 1:1 对应。
> **不阻塞开工**，但建议补齐第 2 节 5 条澄清，可消除实现期歧义；第 3 节为措辞小修。

## 1. 覆盖核对（全部命中）

| 提案 PR1 白名单条目 | 对应 Work Items |
|---|---|
| `/metrics` 顶级端点 + auth 豁免 / token | PR1-04, PR1-05 |
| HTTP 中间件（request_id / 响应头 / 指标 / 入口出口日志） | PR1-06 |
| `prometheus-client` 依赖 + `uv lock` | PR1-01 |
| docs/README + `CHANGELOG.md` | PR1-12, PR1-13 |
| 测试（HTTP 指标 / 鉴权三态 / X-Request-ID） | PR1-09, PR1-10, PR1-11 |

提案中 PR1 "不动"清单（`routing/router.py`、route handler 字段、provider）也都已经
正确进入 **Out of Scope** 段。

## 2. 建议补齐项（不阻塞，但实现期会被问到）

### 2.1 HTTP 指标的具体名称未引用进 taskboard
- 现状：PR1-02 / PR1-04 用了泛指 "HTTP request counter / duration histogram"。
- 提案 §1 lines 41–42 已经把名称固化为：
  - `router_maestro_http_requests_total{method, path_template, status}`
  - `router_maestro_http_request_duration_seconds{method, path_template, status}`
- 修订动作：PR1-04 的 Verification 列直接断言这两个名称存在
  （`GET /metrics` 响应文本中出现这两个 metric name），避免实现者再回提案查。

### 2.2 `api_kind` 枚举在 PR1 是否同步声明 —— 二选一必须写死
- 现状：PR1-02 创建 `server/observability/metrics.py`、PR1-03 列了常量（buckets、
  bool labels、`path_template="unmatched"`），但 **未提是否在 PR1 一并把 `api_kind`
  枚举写进去**（PR1 不消费它，但 PR2 必用）。
- 提案 §1 lines 28–38 已经把 `api_kind` 枚举值列全、并指定位置为该模块的模块级常量。
- 两种处置都行，但 taskboard 必须选定：
  - **Option A（推荐）**：PR1 一次性把 `api_kind` 枚举常量声明完整。
    - 优点：PR2 只做埋点，不再动 metrics 模块结构；PR2 评审更轻。
    - 验证项加一句："`api_kind` 全部 10 个枚举值在常量中可见。"
  - **Option B**：PR1 只声明 HTTP-level 常量，PR2 再加 `api_kind`。
    - 缺点：PR2 多动一处模块结构，评审多一处分歧点。
- 修订动作：在 PR1-03 任务描述中明确选 Option A（推荐），并补齐验证项。

### 2.3 中间件入口 / 出口日志字段范围未交代
- 现状：PR1-06 写了 "request entry/exit logging"，但未说字段集。
- 提案 §4 明确 PR1 **不做** 各 route 字段统一，只提供 `request.state.request_id` 注入。
  这就让 middleware 层日志字段集变得歧义：是否包含 `model` / `provider` / `outcome`？
- 答：不应该 —— 这些是 PR2 的 route handler 职责，middleware 层根本拿不到。
- 修订动作：在 PR1-06 Scope 列里写死 middleware 层日志字段：
  - 入口字段：`request_id, method, path_template`
  - 出口字段：`request_id, method, path_template, status, elapsed_ms`
  - 明确 **不含** `model` / `provider` / `outcome`（PR2 在 route handler 内补充）

### 2.4 PR1-06 缺一条 "401 路径也计入指标" 的验证
- 现状：PR1-06 当前只验证响应头透传 / 生成。
- 提案 §2 line 74 强调："即使 401 也计入 HTTP 指标并带 request_id"。
- 修订动作：PR1-06 加一条 verification：
  - 未带 API key 的请求返回 401；HTTP 计数与直方图仍上报；
    响应头仍包含 `X-Request-ID`。
- 同时可在 PR1-10 测试集合中加一条对应用例。

### 2.5 docs 目标文件未指定
- 现状：PR1-12 写了 "README/docs"，但未说写到哪个文件。
- `AGENTS.md` 列了若干重要 docs；PR1 的最自然落点：
  - `README.md` 加一段 "Metrics & Observability"（最少一段抓取示例）
  - 新建 `docs/observability.md` 放完整说明
  - 或并入现有的 `docs/deployment.md`
- 修订动作：在 PR1-12 选定文件并写死，避免实现者临时拍脑袋。建议：
  - `README.md` 加简短链接段
  - 新建 `docs/observability.md` 放采集示例、指标释义、排障样例

## 3. 措辞小修（一行可改）

### 3.1 PR1-14 "any failures are documented" 措辞偏宽
- 建议改为：
  > `uv run ruff check src/ tests/` 与 PR1 涉及的 focused pytest 全部通过；
  > 如有 skip / xfail 需在 PR 描述中说明原因。

### 3.2 Acceptance Criteria 未呼应 PR1-07
- 现状：验收标准列出 7 条，但未列 "抛异常路径仍上报 HTTP 指标"。
- 建议补一条：
  > Route handlers raising exceptions still record HTTP request counter /
  > duration with `status=5xx`.

### 3.3 PR1-05 验证三态可再细一档
- 当前："no token configured = 200; correct token = 200; missing/wrong token = 401"。
- "missing" 与 "wrong" 实现可能走不同分支（一个是 header 不存在，
  一个是值不匹配），建议拆开测：
  - 已设置 token，请求完全未带 token → 401
  - 已设置 token，请求带了错误 token → 401
  - 已设置 token，请求带了正确 token → 200
  - 未设置 token → 200
- 修订动作：PR1-05 验证项扩为 4 个分支。

## 4. 结论

**Taskboard 已可作为 PR1 实现的工单使用。** 14 个 task 与 PR1 白名单 1:1 对应，
验收标准、可验证性、范围边界都清晰。

剩余反馈集中在 **§2 五条澄清** 与 **§3 三条措辞** —— 全部是 "实现期评审会问到、
但不影响代码结构" 的细节。

**处置建议（二选一）：**

- **路径 A（推荐）**：花 10 分钟补齐 §2 五条澄清 + §3 三条措辞，然后开工。
  PR1 实现期可"按 taskboard 打卡"，零回头。
- **路径 B**：立即基于现版开工，§2 的 5 条作为 PR 内 review comment 一次性处理。
  适合工期紧的情况，但 PR 评审多一轮往返。
