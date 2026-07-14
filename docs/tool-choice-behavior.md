# tool_choice 与 finish_reason 行为分析

## 直连 OpenAI API 的 `finish_reason` 行为

以下表格记录本页引用的原生 OpenAI API 观测结果，不是
Router-Maestro 规范化后的输出：

| tool_choice | finish_reason | tool_calls | 说明 |
|---|---|---|---|
| `"auto"` | `"tool_calls"` | 存在 | 模型主动决定调用工具 |
| `"required"` | `"stop"` | 存在 | 强制使用工具，视为正常结束 |
| `{"type":"function","function":{"name":"..."}}` | `"stop"` | 存在 | 强制调用指定函数，视为正常结束 |

**三种格式下 `tool_calls` 都正确返回**，唯一差异是 `finish_reason`。流式与非流式行为一致。

## Copilot API 与原生 OpenAI API 的差异

| tool_choice | OpenAI 官方 API | GitHub Copilot API |
|---|---|---|
| `"auto"` | `finish_reason=tool_calls` | `finish_reason=tool_calls` |
| `"required"` | **`finish_reason=stop`** | **`finish_reason=tool_calls`** |
| `{"type":"function",...}` | `finish_reason=stop` | `finish_reason=stop` |

Copilot API 在 `"required"` 下返回 `"tool_calls"`，与原生 OpenAI API（返回 `"stop"`）有差异。function-specific 格式下两者一致。

## OpenAI 官方回应与社区讨论

上述 `finish_reason` 行为是 OpenAI 的**设计决策（by design）**：当 `tool_choice` 为 `"required"` 或指定具体函数时，模型是被**强制**调用工具而非主动选择，因此 `finish_reason` 返回 `"stop"`（正常结束）而非 `"tool_calls"`（主动调用工具）。只有 `tool_choice: "auto"` 下模型自主决定调用工具时，才返回 `finish_reason: "tool_calls"`。

### 官方回应

OpenAI 员工 @brianz-oai 在 [社区论坛](https://community.openai.com/t/new-api-feature-forcing-function-calling-via-tool-choice-required/731488) 中解释：

> To provide a bit more context, before we introduced this new feature, when you set `tool_choice: {"type": "function", "function": {"name": "my_function"}}`, the `finish_reason` would always be `stop` rather than `tool_calls`. Only when you used the default `tool_choice: "auto"` option, and the model chose to use a tool, the `finish_reason` would be `tool_use`. So when we designed this new feature, we thought it made more sense to provide the consistent behavior as `tool_choice: {"type": "function", "function": {"name": "my_function"}}` as the two are more similar (i.e. model is forced to use a tool).
>
> Fixing this now could potentially break some users' integration, but we will almost certainly fix this when we release the next API version.

### 社区讨论

这一行为在社区中引发了长期讨论（[帖子跨度超过 15 个月](https://community.openai.com/t/function-call-with-finish-reason-of-stop/437226)），主要观点：

- 社区用户进行了 500 次请求的统计测试，确认 `tool_choice={"type":"function",...}` 时 500/500 返回 `finish_reason: "stop"`，行为完全确定性
- 多位开发者报告生产环境因依赖 `finish_reason` 而出现故障
- 社区共识是检查 `message.tool_calls` 而非 `finish_reason`：

  > There should not be any decision-making done based on the finish reason, except to report "length" as a cause of problems from setting max_completion_tokens too low, or "content_policy" as a result of detected recitation (copyright) interrupting the stream.

## Router-Maestro 代理行为

上表记录的是直连 API 的观测行为。Router-Maestro 对 Copilot Chat
响应做 canonical normalization，不保证逐字节透传：它会合并
Copilot 分开返回的文本与工具 choices，将泄漏在文本中的 XML
工具调用恢复为结构化 `tool_calls`，并在实际存在工具调用时将
`finish_reason` 统一为 `"tool_calls"`。Anthropic 和 Gemini 入口还会
分别编码为它们的原生工具终态。

Router-Maestro 会在上游 I/O 前校验所选 provider/model 是否支持请求的
operation、tools 和 parallel tool calls。支持的 `tools` / `tool_choice` 会按目标
provider 的原生格式透传或等价翻译；无法表达的显式选项会以入口协议的 HTTP
400 错误返回，不会静默丢弃，也不会借此切换到另一模型。Copilot Responses
还会拒绝其不支持的 tool type；因此本页的 `finish_reason` 表格只描述已经通过
这些能力与选项校验、实际发往 Copilot Chat transport 的请求。

## 客户端注意事项

`finish_reason` 不应作为是否存在工具调用的判断依据：

```javascript
// 正确：直接检查 tool_calls 字段
if (message.tool_calls && message.tool_calls.length > 0) {
  handleToolCalls(message.tool_calls);
}

// 错误：依赖上游或某一 API 版本的 finish_reason 差异
if (finish_reason === "tool_calls") {
  handleToolCalls(message.tool_calls);
}
```

不要为了改变 `finish_reason` 而改写 `tool_choice`；两者表达的请求
语义不同。客户端应直接检查规范化后的 `tool_calls`。
