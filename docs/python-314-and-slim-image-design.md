# Design: Python 3.14 升级 + Docker 镜像瘦身

**Date:** 2026-07-22
**Branch:** `chore/python-314-and-slim-image`
**Status:** Approved (pending spec review)

## 目标

1. 把项目从 Python 3.11 升级到 3.14(项目最低版本抬到 3.14)。
2. 缩小最终 Docker 镜像体积,砍掉运行时不需要的构建期工具。

## 背景与测量

Router-Maestro 是**应用**(CLI + FastAPI server),不是被他人 `import` 的库。它同时发布到 PyPI 和 Docker Hub。因为没有下游项目把它当依赖,抬高 `requires-python` 不会卡住任何传递依赖——`pipx` / `uv tool` / Docker 都会自动拉取 3.14。

### 当前镜像体积构成(本地 arm64 baseline 实测)

最终镜像 126MB(已发布的 amd64 变体为 189MB;musl CPython 在 x86-64 上更大,比例一致)。`docker history` + venv 内部拆解:

| 层 | 大小 | 运行时需要? |
|---|---|---|
| CPython 3.11 解释器(基础镜像内)| 49 MB | ✅ 固定成本 |
| venv 里的 `pip` | 15.8 MB | ❌ |
| venv 里的 `setuptools` + `pkg_resources` | ~11 MB | ❌ |
| `curl`(仅 healthcheck 用)| ~4 MB | ❌ 可用 urllib 替代 |
| rapidfuzz / pydantic_core / tiktoken / regex | ~20 MB | ✅ 真实依赖 |
| alpine base + apk | ~9.7 MB | ✅ |

**结论:** 49MB 是解释器本体(换 3.14 后有等体积等价层,躲不掉)。真正能砍的是我们叠加上去的 **pip + setuptools + curl ≈ 30MB**,不动任何依赖、不牺牲可调试性。

### 3.14 依赖兼容性(已实测通过)

在 `python:3.14-alpine` 容器内 `apk add gcc musl-dev libffi-dev cargo rust` 后,`uv pip install` 全部生产依赖,三个编译扩展均装成并 import 成功:

```
Python 3.14.6
pydantic-core==2.46.4  tiktoken==0.13.0  rapidfuzz==3.14.5  regex==2026.7.19
uvicorn==0.51.0  fastapi(starlette==1.3.1)  h2  httpx[socks]
ALL IMPORTS OK
```

关键风险(编译扩展无 3.14 wheel)已排除。Docker 走 Alpine 源码编译,本就不依赖预编译 wheel。

## Part A — Python 3.14 升级

| 文件 | 现在 | 改成 |
|---|---|---|
| `pyproject.toml` `requires-python` | `>=3.11` | `>=3.14` |
| `pyproject.toml` classifiers | `3.11`, `3.12` | `3.14`(删旧的) |
| `pyproject.toml` `[tool.ruff] target-version` | `py311` | `py314` |
| `.github/workflows/ci.yml` | matrix `["3.11","3.12"]`;lint 用 3.12 | 只测 `3.14`;lint 用 3.14 |
| `.github/workflows/release.yml` | test / pypi build 用 3.12 | 用 3.14 |
| `uv.lock` | `requires-python >=3.11` | `uv lock` 重新生成 |

**决策(已确认):** CI 砍到只测 3.14,符合「彻底 3.14 项目」定位。

## Part B — Docker 镜像瘦身(稳妥清理,不碰依赖)

改 `Dockerfile`,两个 stage 基础镜像都 `python:3.11-alpine` → `python:3.14-alpine`,并在 builder 阶段清理运行时无用产物。预期 **126MB → ~92MB(约 -30MB)**。

1. **剥掉 pip / setuptools / pkg_resources**(~26MB):`uv pip install` 完成后,从 venv 的 site-packages 里删除这几个目录(RM 运行时不加载它们)。
2. **干掉 curl**(~4MB):healthcheck 从 `curl -f http://.../health` 换成 Python 自带 `urllib` 一行脚本——解释器本就在,零额外体积。运行时层不再 `apk add curl`,仅保留 `libffi`。
3. **清 `__pycache__`**:copy venv 前清掉编译缓存(`PYTHONDONTWRITEBYTECODE=1` 已设,运行时按需生成)。

体积账:pip 15.8 + setuptools/pkg_resources ~11 + curl ~4 ≈ 30MB;`__pycache__` 清理为小额外收益。实际数字以第 3 步实测为准。

**决策(已确认):** healthcheck 换 urllib——等价功能,省 4MB,代价是命令从一眼可读的 curl 变成一行 python,值得。

### 明确不做(YAGNI)

- 不上 distroless / chainguard(留待本次清理后实测再评估是否值得多砍那几十 MB 承担无 shell 的运维代价)。
- 不动任何一个真实依赖。
- 不改任何应用代码(`src/`)。

## 验证方案

1. **单元测试:** `uv run pytest tests/` —— 508+ 测试全绿。
2. **Lint:** `uv run ruff check src/ tests/` + `ruff format --check`。
3. **本地构建镜像:** 用新 Dockerfile `docker build`,`docker images` 记录新体积并与 126MB baseline 对比。
4. **对 docker 容器跑全量 integration test(核心验证):**
   - **临时**修改 `integration_tests/conftest.py`:给 `live_server` fixture 加一条分支——当环境变量 `RM_INTEGRATION_BASE_URL` 存在时,复用该 URL 并跳过自起 uvicorn 子进程(默认行为不变)。此改动仅为完成本次验证,不作为本 PR 的正式产物(验证后回退)。用户已明确批准临时改 conftest。
   - 用本地配置/auth 启动刚 build 的容器:挂载 `~/.config/router-maestro` 与 `~/.local/share/router-maestro`(含 Copilot auth),映射端口,`ROUTER_MAESTRO_API_KEY` 注入。
   - 等 `/health` 返回 200,healthcheck 变绿。
   - `RM_INTEGRATION_BASE_URL=http://127.0.0.1:<port> uv run pytest integration_tests/ -v` 跑**全量** integration test,请求全部打到容器。
5. **跳过:** 本地直起(`make integration-test`)那轮 integration test 按用户要求跳过——docker 版本通过即代表运行路径通过。

## 交付物(正式进 PR)

- `pyproject.toml`、`ci.yml`、`release.yml`、`uv.lock`(Part A)
- `Dockerfile`(Part B)
- 本设计文档

`conftest.py` 的临时改动**不进 PR**。

## 验证结果(实测)

- **单元测试:** 3475 passed(Python 3.14.6);`ruff check` + `ruff format --check` 全 clean(采纳 py314 target 后修掉 F811 重名 bug + PEP 695/UP 现代化)。
- **镜像体积:** 126MB(baseline-311)→ **88.5MB**(314-slim),**-37.5MB / -30%**,优于 ~92MB 预估。venv 已无 pip/setuptools/pkg_resources,curl 已移除,容器内 `from router_maestro.server import app` + tiktoken/pydantic-core/rapidfuzz 均 import OK。
- **全量 integration test 打到 docker 容器:** 76 passed, 2 failed, 3 skipped。两个 failure 均已证明**与本分支无关**:① `test_stream_open_errors[anthropic-beta]` 在 master(02abffa)上同样失败——既有问题;② `test_anthropic_claude_thinking_budget_matrix` 为 live Copilot 后端偶发(重试 1 passed)。3 个 skip 为环境条件性跳过,非回归。

