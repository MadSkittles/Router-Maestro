# Metrics + Logging PR 审阅清单

## 1) 目标与范围
- [ ] PR 目标聚焦 `metrics + logging`，未混入限流或无关重构。
- [ ] 变更未修改现有 API 语义与路由行为。
- [ ] 方案能覆盖“路由质量 + provider 质量”的核心可观测需求。

## 2) `/metrics` 端点与指标设计
- [ ] 暴露了 Prometheus 可抓取的 `/metrics` 端点。
- [ ] `/metrics` 已明确鉴权策略（API key 豁免或独立 metrics token）。
- [ ] `/metrics` 使用顶级路径，不挂在 `/api/admin` 下。
- [ ] `/metrics` 与 `ROUTER_MAESTRO_METRICS_TOKEN` 三态行为有测试覆盖。
- [ ] 指标命名、单位、类型（Counter/Histogram）语义清晰。
- [ ] 指标命名统一使用 `router_maestro_` 前缀。
- [ ] HTTP 维度至少包含请求总数和请求耗时。
- [ ] HTTP 指标 path label 使用 FastAPI 路由模板（非原始 URL）。
- [ ] `path_template` 对未命中路由（404）使用固定占位符 `"unmatched"`。
- [ ] 路由维度至少包含 provider 成功、provider 失败、fallback 触发。
- [ ] 指标标签未引入高基数字段（如原始 prompt、完整错误文本、request body）。
- [ ] 已固定 `api_kind` 枚举，避免跨模块命名漂移。
- [ ] `api_kind` 枚举值已固化在提案 / 代码常量中（不是各模块自由命名）。
- [ ] `router_maestro_provider_request_duration_seconds`（non-stream）名称 / 标签 / buckets 已在提案中固化。
- [ ] `fallbacks_total.reason` 取值已枚举化。
- [ ] `provider_errors_total` 的标签集已固定，`error_class` 取值已枚举化（无原始 message）。
- [ ] 布尔型 label（`stream` / `retryable`）取值统一为 `"true"` / `"false"` 字符串。
- [ ] 直方图 buckets 已针对 LLM 时延定制（非默认 buckets）。

## 3) 请求级日志一致性
- [ ] 请求链路具备统一 `request_id`（透传或自动生成）。
- [ ] middleware 已在响应头回写 `X-Request-ID`。
- [ ] 响应头 `X-Request-ID` 与日志字段一致。
- [ ] 入口与出口日志字段一致（建议含 `model/provider/stream/elapsed_ms/outcome`）。
- [ ] `chat` 与 `responses` 路径日志字段命名一致。
- [ ] `anthropic` 与 `gemini` 路径字段命名也一致。
- [ ] 已消除 `req_id` 与 `request_id` 双轨并存。
- [ ] 日志未记录敏感或高体量内容（尤其用户输入全文）。
- [ ] 如按 PR1/PR2 拆分，PR1 仅提供 request_id 注入与响应头回写，route 字段统一留给 PR2。

## 4) 路由与 fallback 观测完整性
- [ ] 首次路由成功可被指标和日志同时观察到。
- [ ] provider 异常（含 retryable/non-retryable）有对应指标。
- [ ] fallback 尝试与 fallback 成功路径都有可观测信号。
- [ ] streaming 场景错误或中断也有明确日志/指标记录。
- [ ] fallback 指标维度已收敛（不携带 model 等高基数字段）。
- [ ] outcome 三态（completed/failed/cancelled）语义清晰且可观测。
- [ ] `outcome` label 仅出现在 stream 相关指标（non-stream 不带 outcome）。
- [ ] streaming 已区分首包时延与整流时长指标。

## 5) 测试覆盖
- [ ] 有 `/metrics` 可访问与关键指标存在性的测试。
- [ ] 有请求执行后指标值变化的测试（至少 Counter 增长）。
- [ ] 有 fallback 场景指标变化测试。
- [ ] 有 request_id 贯穿日志链路的测试或等价断言。
- [ ] 有 `X-Request-ID` 同时出现在响应头与入口/出口日志的测试或等价断言。
- [ ] 新测试在本地可稳定通过，不依赖真实上游网络。
- [ ] Prometheus 测试已使用独立 `CollectorRegistry()` 或等价隔离机制。

## 6) 文档与可运维性
- [ ] 文档说明了如何抓取 `/metrics`（本地/容器场景至少一种）。
- [ ] 文档解释了关键指标含义及排障用途。
- [ ] 文档包含最小排障示例（例如：provider 抖动与 fallback 激增）。
- [ ] `CHANGELOG.md` 已记录新增可观测能力。

## 7) 风险与回归检查
- [ ] 没有引入明显性能回退（日志/指标开销可控）。
- [ ] 中间件顺序不会影响认证与现有接口行为。
- [ ] 抛异常路径仍能上报 HTTP 直方图（try/finally 或 Response 回调）。
- [ ] 失败路径不会吞错或改变原有 HTTP 状态码返回。
- [ ] 对现有日志系统（Rich + file）兼容，不破坏当前输出。
- [ ] 未引入额外日志框架（沿用现有 logging 体系）。
- [ ] `pyproject.toml` 新增依赖后已更新 `uv.lock`。

## 8) 合并建议（审阅结论）
- [ ] 可直接合并：核心能力完整，风险可接受。
- [ ] 需小修后合并：列出必须修复项并标注优先级。
- [ ] 建议拆分 PR：如果当前改动过大或跨越范围。
- [ ] 如拆分，PR1 与 PR2 边界清晰、可独立验证。
- [ ] PR1 的“动 / 不动”白名单清单清晰。
