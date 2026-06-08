# Metrics + Logging 首个 PR 提案

## 目标
- 增加可抓取的 Prometheus 指标端点，评估路由质量与 provider 质量。
- 统一请求级日志字段，能串联“入口请求 -> 路由决策 -> fallback/失败 -> 完成”。

## 变更范围
- 只做 `metrics + logging`。
- 不做 `rate-limiting`、不改协议语义、不改现有路由行为。
- 不引入新的日志框架（沿用现有 Rich + file logging）。

## PR 拆分策略（建议）
- **PR1（推荐先合）**: HTTP 层 metrics + `/metrics` + request middleware + 文档。
- **PR2（随后）**: `routing/router.py` 的 provider/fallback 埋点 + 各 route 日志字段统一。
- 拆分原因：PR1 低风险、纯增量、易审阅；PR2 涉及核心路由日志路径，评审成本更高。
- PR1 落地白名单：
  - `/metrics` 顶级端点 + auth 豁免/token 校验
  - HTTP 中间件（`request_id` 注入、响应头回写、HTTP 指标、入口/出口日志）
  - `prometheus-client` 依赖 + `uv lock`
  - docs/README 采集与排障说明 + `CHANGELOG.md`
  - 测试覆盖 HTTP 指标、`/metrics` 鉴权三态、`X-Request-ID` 响应头
- PR1 不动：`routing/router.py`、各 route handler 内部字段、provider 代码。

## 具体方案
1. **新增观测模块**
   - 新建 `server/observability/metrics.py`，集中定义指标与上报函数。
   - 所有指标统一使用 `router_maestro_` 前缀。
   - 在模块级常量中声明 `api_kind` 固定枚举，避免模块间命名漂移：
     - `openai_chat`
     - `openai_responses`
     - `openai_models`
     - `anthropic_messages`
     - `anthropic_count_tokens`
     - `anthropic_models`
     - `gemini_generate`
     - `gemini_stream`
     - `gemini_count_tokens`
     - `admin`
   - 使用低基数标签：禁止把原始 URL、用户输入、完整错误文本放入 label。
   - 指标首版包含：
     - `router_maestro_http_requests_total{method,path_template,status}`
     - `router_maestro_http_request_duration_seconds{method,path_template,status}`，使用 LLM 定制 buckets：`[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60, 120]` 秒
     - `router_maestro_requests_to_provider_total{api_kind,provider,stream}`
       - `stream` 取值为 `"true"` / `"false"` 字符串
     - `router_maestro_provider_request_duration_seconds{api_kind,provider}`
       - 仅在 non-stream provider 调用路径上报
       - 使用与 HTTP 直方图相同的 LLM 定制 buckets：`[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60, 120]` 秒
       - stream 路径由 first-chunk 与 stream-duration 两个指标表达，不对同一次 stream 调用双重计数
     - `router_maestro_fallbacks_total{api_kind,from_provider,to_provider,reason}`，不带 model
       - `reason` 只能取枚举值：`http_4xx` / `http_5xx` / `timeout` / `connection_error` / `auth` / `other`
     - `router_maestro_provider_errors_total{api_kind,provider,retryable,error_class}`
       - `retryable` 取值为 `"true"` / `"false"` 字符串
       - `error_class` 只能取枚举值：`http_4xx` / `http_5xx` / `timeout` / `connection_error` / `auth` / `other`
       - 禁止使用原始异常 message、HTTP body 或上游错误全文
     - streaming 首包时延与整流时长：
       - `router_maestro_provider_first_chunk_latency_seconds{api_kind,provider}`
       - `router_maestro_provider_stream_duration_seconds{api_kind,provider,outcome}`
       - `outcome` 仅用于 stream duration，取值为 `completed` / `failed` / `cancelled`
     - non-stream 不引入 `outcome` label，成功/失败由 `router_maestro_requests_to_provider_total` 与 `router_maestro_provider_errors_total` 表达。

2. **在应用入口接入**
   - 在 `server/app.py` 挂载顶级 `/metrics`，不放在 `/api/admin` 前缀下。
   - 为 `/metrics` 定义明确鉴权策略：
     - 默认豁免 `verify_api_key`（加入 `middleware/auth.py` 的 public paths 或等价机制）；
     - 若设置 `ROUTER_MAESTRO_METRICS_TOKEN`，则 `/metrics` 改为校验该 token；
     - metrics token 为覆盖式鉴权，不复用 Router-Maestro API key。
   - 增加全局 request middleware：
     - 提取或生成 `request_id`（支持 `X-Request-ID` 透传）
     - 在响应头回写 `X-Request-ID`
     - 记录请求入口/出口日志
     - 上报 HTTP 级指标
   - HTTP path 标签必须使用 FastAPI 路由模板（`request.scope["route"].path`），不使用原始 path。
   - 未命中路由（404）时使用固定占位符：`path_template = "unmatched"`。
   - 明确 middleware 顺序预期：即使 401 也计入 HTTP 指标并带 request_id。
   - HTTP 时延直方图在 `finally` 块或基于 `Response` 回调中上报，保证 route handler 抛异常时仍能记录 `status=5xx` 与 elapsed。

3. **在路由核心埋点**
   - 在 `routing/router.py` 的关键路径上报：
     - 首次路由成功
     - provider 失败
     - fallback 尝试与 fallback 成功
     - stream 场景错误
   - 控制标签维度，避免高基数 label。
   - `outcome` 仅用于 stream 相关指标，明确三态：`completed` / `failed` / `cancelled`。

4. **统一日志字段（轻量）**
   - `chat`、`responses`、`anthropic`、`gemini` 主路径统一字段命名：
     - `request_id`, `model`, `provider`, `stream`, `elapsed_ms`, `outcome`
   - 统一策略：由 middleware 写入 `request.state.request_id`，各 route 复用，避免 `req_id` 与 `request_id` 双轨并存。
   - PR1 仅提供 `request.state.request_id` 注入与 `X-Request-ID` 响应头回写；各 route handler 的字段统一在 PR2 完成。
   - 不记录敏感 prompt 全量内容，仅记录必要元数据。

5. **测试与文档**
   - 新增测试验证：
     - `/metrics` 可访问，且关键指标存在
     - `/metrics` 鉴权三态：未设置 token 返回 200；设置 token 且正确返回 200；设置 token 且缺失/错误返回 401
     - 未命中路由（404）时，HTTP 指标使用 `path_template="unmatched"`
     - 请求后计数与耗时指标变化正确
     - fallback 场景有对应指标变化
     - 日志字段具备统一 request_id
     - `X-Request-ID` 同时出现在响应头与入口/出口日志
     - streaming 场景可区分 completed/failed/cancelled
   - 避免测试污染：Prometheus 指标测试使用独立 `CollectorRegistry()`（或等价清理机制）。
   - 依赖与锁文件：
     - 在 `pyproject.toml` 增加 `prometheus-client>=0.20.0`
     - 执行 `uv lock` 更新 `uv.lock`
   - 更新 README/docs：采集方式、关键指标释义、排障示例。
   - 更新 `CHANGELOG.md` 记录新增可观测能力。

## 验收标准
- 服务启动后可直接抓取 `/metrics`。
- 通过一次正常请求和一次失败/回退请求，可以在指标和日志中看到一致链路。
- `/metrics` 鉴权策略明确且可用（豁免或 metrics token 二选一）。
- route/streaming 的 `request_id` 与 `outcome` 语义一致，不出现双 ID。
- 现有功能行为不回归。
