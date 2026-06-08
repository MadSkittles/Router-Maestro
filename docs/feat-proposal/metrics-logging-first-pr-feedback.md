# Metrics + Logging 首个 PR 提案 —— 第三轮评审反馈

> 评审范围：`docs/feat-proposal/metrics-logging-first-pr.md`（再修订版）与
> `docs/feat-proposal/metrics-logging-first-pr-review-checklist.md`
> 评审结论：第二轮全部必改 + 建议项已落地，提案已达到 **ready-to-implement** 状态。
> 仅余 4 条小一致性瑕疵，**不阻塞** 进入 PR1 实现，但建议顺手补齐让文档成为零歧义契约。

## 1. 第二轮必改 / 建议项落地核对

### 1.1 必改项（6 条全部落地）

| # | 第二轮必改项 | 落地位置 |
|---|---|---|
| 2.1 | `api_kind` 枚举值列入正文 + 声明位置 | §1 lines 28–38 |
| 2.2 | `outcome` 三态绑定到 stream duration；non-stream 不带 outcome | §1 lines 52–54、§3 line 78 |
| 2.3 | `provider_errors_total` 标签集 + `error_class` 枚举 | §1 lines 46–49 |
| 2.4 | 中间件异常路径用 finally / Response 回调 | §2 line 69 |
| 2.5 | `/metrics` 顶级挂载 + token 覆盖式鉴权 + 三态测试 | §2 lines 57–61、§5 line 90 |
| 2.6 | `X-Request-ID` 响应头回写 + 测试断言 | §2 line 64、§5 line 94 |

### 1.2 建议项（5 条全部落地）

| # | 第二轮建议项 | 落地位置 |
|---|---|---|
| 3.1 | `router_maestro_` 命名空间前缀 | §1 line 27 + 各指标命名 |
| 3.2 | PR1 "动 / 不动" 白名单 | §「PR 拆分策略」lines 16–22 |
| 3.3 | §4 在 PR1 / PR2 的归属 | §4 line 84 |
| 3.4 | `CHANGELOG.md` 更新 | §5 line 101 |
| 3.5 | `X-Request-ID` 出现在响应头与日志的测试断言 | §5 line 94 |

### 1.3 审阅清单补充

第二轮建议的 8 条新条目已全部加入 checklist
（lines 11, 12, 20, 21, 27, 33, 42, 63, 74）。

## 2. 第三轮剩余瑕疵（不阻塞，建议顺手补）

### 2.1 非流式 provider 耗时直方图未明确名称 / 标签 / buckets
- 现状：§1 line 44 仍是泛泛一句"provider 调用耗时直方图（区分 non-stream 与 stream）"。
- 问题：流式部分（`provider_first_chunk_latency_seconds` /
  `provider_stream_duration_seconds`）都给了完整签名，唯独 non-stream 留白。
- 建议固化（写入 §1）：
  - 指标名：`router_maestro_provider_request_duration_seconds`
  - 标签集：`{api_kind, provider}`（与 `provider_first_chunk_latency_seconds` 对齐）
  - buckets：复用 HTTP 直方图的 LLM 定制 buckets
    `[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60, 120]` 秒
  - 仅在 non-stream 路径上报；stream 路径由 first_chunk + stream_duration 两个指标
    分别表达，不双重计数。

### 2.2 `fallbacks_total.reason` 取值未枚举化
- 现状：§1 line 45 把 `reason` 当成自由字符串。
- 问题：`provider_errors_total.error_class` 已枚举化，`reason` 若保持自由文本，
  实现期容易把上游错误 message 塞进去，破坏低基数原则，与 §1 line 49
  "禁止使用原始异常 message" 自相矛盾。
- 建议固化（写入 §1，二选一）：
  - 与 `error_class` 完全对齐：
    `reason ∈ {http_4xx, http_5xx, timeout, connection_error, auth, other}`
  - 或更贴合 fallback 语义：
    `reason ∈ {retryable_error, rate_limit, timeout, auth, other}`

### 2.3 `path_template` 在 404 / unmatched 路由下的回退值未约定
- 现状：§2 line 67 只说"使用 FastAPI 路由模板 `request.scope["route"].path`"。
- 问题：路由未命中时 `request.scope.get("route")` 可能为 `None`；
  不显式约定 fallback，实现期可能让 label 变成空字符串或原始 path，
  造成基数泄漏（攻击者用随机 path 打 404 即可炸 metrics）。
- 建议固化（写入 §2 line 67 后）：
  - 未命中路由（404）使用固定占位符：`path_template = "unmatched"`
  - 测试覆盖一条 404 请求验证占位符行为

### 2.4 `requests_to_provider_total.stream` 标签取值未约定
- 现状：§1 line 43 写了 `stream` 标签但未指定取值类型。
- 问题：与 `retryable` 已经显式约定 `"true"` / `"false"` 字符串保持一致即可，
  否则不同 route 实现可能用 `"1"` / `"0"` 或 `True` / `False`。
- 建议固化（写入 §1）：
  - `stream` 取值为 `"true"` / `"false"` 字符串
    （与 `retryable` 一致，Prometheus 客户端 label 总是字符串）

## 3. 审阅清单可顺手补充

建议在 `metrics-logging-first-pr-review-checklist.md` §2 增加：

- [ ] `router_maestro_provider_request_duration_seconds`（non-stream）名称 /
      标签 / buckets 是否已在提案中固化
- [ ] `fallbacks_total.reason` 取值是否已枚举化
- [ ] `path_template` 对未命中路由（404）是否使用固定占位符 `"unmatched"`
- [ ] 布尔型 label（`stream` / `retryable`）取值是否统一为 `"true"` / `"false"` 字符串

## 4. 结论

**修订版已经 ready-to-implement。** 三轮评审累计闭环 14 条必改 + 8 条建议 + 16 条
checklist 新增条目。

第三轮剩余 4 条都是"实习生也能踩到的小坑"级别的纯一致性瑕疵，
**不阻塞** 合并提案。处置选择：

- 想要 PR1 实现期零分歧 → 请提案作者再花 5 分钟补齐第 2 节 4 条；
- 想立刻进入实现 → 直接基于现版开 PR1，把上述 4 条作为 PR1 review comment
  一次性提即可。
