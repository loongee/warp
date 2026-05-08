# Warp Local Proxy

> 让 [Warp](https://www.warp.dev) 终端通过本地代理直接调用 [DeepSeek](https://www.deepseek.com)（或任何 OpenAI 兼容模型）的本地后端。

本项目是一个用 **FastAPI** 实现的轻量级反向代理 / Mock 服务器，模拟 Warp 客户端所需的全部后端接口（GraphQL、Auth、模型列表、配额等），并把核心的 AI 对话请求 `POST /ai/multi-agent` 翻译成 OpenAI 兼容的 `chat/completions` 流式请求转发给 DeepSeek，再把响应按 Warp 的 protobuf + SSE 协议流回客户端。

借此你可以：

- 在不登录 Warp 官方账号的情况下，本地运行自编译的 Warp 客户端；
- 把 AI 后端切换为 DeepSeek（默认 `deepseek-v4-flash`），自带工具调用（运行命令、读文件、应用 diff）；
- 自由替换为任何 OpenAI 兼容的模型 / 网关。

---

## 目录结构

```
warp-local-proxy/
├── server.py                # FastAPI 入口，挂载所有路由 + 兜底 mock
├── config.py                # 环境变量读取（API Key、模型名、端口）
├── deepseek_client.py       # DeepSeek 流式客户端 + 工具定义 + reasoning 缓存
├── handlers/
│   ├── graphql.py           # /graphql/v2 mock：模型列表、配额等
│   └── multi_agent.py       # /ai/multi-agent：核心 AI 对话端点
├── proto/                   # 由 warp-proto-apis 生成的 *_pb2.py
└── requirements.txt
```

## 工作原理

```
┌──────────────┐   protobuf over HTTP    ┌─────────────────────┐   OpenAI-compatible    ┌───────────┐
│  Warp 客户端 │  ─────────────────────▶ │  warp-local-proxy   │  ────────────────────▶ │ DeepSeek  │
│ (warp-oss)   │  ◀─────────────────────  │  (FastAPI, :8765)   │  ◀────────────────────  │   API     │
└──────────────┘   protobuf SSE (b64url) └─────────────────────┘   SSE stream           └───────────┘
```

请求/响应流程：

1. Warp 客户端发出 `POST /ai/multi-agent`，body 是序列化后的 `Request` protobuf。
2. 代理解析 protobuf，提取用户输入、工具调用结果以及历史会话，重建成 OpenAI `messages` 数组。
3. 调用 DeepSeek `/chat/completions`（流式 + function calling）。
4. 把 DeepSeek 的 `content` / `tool_calls` 增量翻译成 `ResponseEvent` protobuf，base64url 编码后通过 **SSE** 推回 Warp。
5. GraphQL、Auth、`/api/v1/*` 等接口由 mock handler 直接返回最小可用数据，避免客户端报错。

> DeepSeek 思考模式下要求把 `reasoning_content` 回传给后续请求，代理使用进程内 `_reasoning_cache` 按 `conversation_id` 缓存。

## 已支持的工具调用

`deepseek_client.py` 中向模型暴露了 3 个工具，对应 Warp 端的交互 UI：

| 工具                | 作用                          | Warp 中的表现                |
| ------------------- | ----------------------------- | ---------------------------- |
| `run_shell_command` | 执行 shell 命令               | 弹出"运行 / 拒绝"确认气泡    |
| `read_files`        | 读取一个或多个文件            | 文件内容回填到下一轮会话     |
| `apply_file_diffs`  | 创建 / 修改文件（unified diff）| 触发 Warp 的 diff 预览界面  |

## 快速开始

### 1. 安装依赖

推荐使用 Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

| 变量                | 默认值                          | 说明                                         |
| ------------------- | ------------------------------- | -------------------------------------------- |
| `DEEPSEEK_API_KEY`  | *(空)*                          | DeepSeek API Key。也可以由 Warp 客户端设置中传入 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com`      | 替换为任何 OpenAI 兼容网关均可               |
| `DEEPSEEK_MODEL`    | `deepseek-v4-flash`             | 模型名                                       |
| `PROXY_PORT`        | `8765`                          | 代理监听端口                                 |

> Warp 客户端在请求 settings 中携带的 `openai` API Key 优先级 **高于** `DEEPSEEK_API_KEY` 环境变量，可以直接在 Warp 设置面板里填入 DeepSeek Key。

### 3. 启动代理

```bash
DEEPSEEK_API_KEY=sk-xxxxxxxx uvicorn server:app --port 8765
# 或者直接：
python server.py
```

健康检查：

```bash
curl http://localhost:8765/health
# {"status":"ok"}
```

### 4. 启动 Warp 客户端指向本地代理

以 `warp` 仓库自编译的 `warp-oss` 为例：

```bash
WARP_SERVER_ROOT_URL=http://localhost:8765 ./target/debug/warp-oss
```

之后在 Warp 中正常使用 Agent / Coding 模式即可——所有 AI 请求都会走本地 DeepSeek。

## 模型选择

`handlers/graphql.py` 中通过 `GetFeatureModelChoices` / `FreeAvailableModels` 把唯一一个模型 `deepseek-chat` 暴露给客户端，并对 `agentMode`、`coding`、`cliAgent`、`computerUseAgent` 四个特性都返回它，因此 Warp 的模型选择器里只会看到 DeepSeek。

如果需要支持多模型，可以在 `DEEPSEEK_MODEL_ENTRY` 旁边追加更多条目，并扩展 `deepseek_client.stream_chat` 的 `model` 路由逻辑。

## 配额 / 鉴权

- `/api/v1/auth/*` 全部返回 `{"status":"ok"}`，跳过登录。
- `GetRequestLimitInfo` 返回 `isUnlimited: true`，客户端不会触发配额墙。
- 任何未实现的 `/api/v1/*`、`/ai/*` 接口都会被兜底 handler 接住并打印日志，方便定位还需要补哪些 mock。

## 调试技巧

- 代理控制台会打印每次请求的 API Key 后 4 位、用户 query 摘要、历史 message 角色序列，以及未被显式处理的接口路径。
- DeepSeek 报错时会把请求 payload 打印出来（前 2000 字符），便于定位是不是 `tool_calls` / `tool` 顺序异常。
- 若历史消息中 assistant 携带 `tool_calls` 但没有对应的 `tool` 响应，`multi_agent.py` 会自动剥离 `tool_calls` 转成普通 assistant 文本，避免 DeepSeek 报错。

## 已知限制

- `apply_file_diffs` 目前只透传 `summary`，diffs 的字段映射尚待补全。
- `_reasoning_cache` 仅保存在进程内存中，重启代理会丢失上下文。
- 目前仅实现了 Warp 客户端常用接口的子集，部分高级特性（团队空间、Drive、Notebook 等）未 mock。

## License

仅供个人本地使用。涉及的 protobuf 定义版权归 Warp 所有。
