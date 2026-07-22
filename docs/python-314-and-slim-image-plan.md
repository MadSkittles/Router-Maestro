# Python 3.14 Upgrade + Docker Image Slimming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Router-Maestro 从 Python 3.11 升级到 3.14,并把最终 Docker 镜像瘦身约 30MB(剥除运行时无用的 pip/setuptools/curl)。

**Architecture:** Part A 改元数据与 CI(pyproject/ruff/两个 GH workflow/uv.lock),纯配置无代码逻辑变更。Part B 改单个 Dockerfile:两个 stage 基础镜像升 3.14,builder 阶段清理 venv,healthcheck 由 curl 换 Python urllib。验证阶段临时给 integration conftest 加一个 `RM_INTEGRATION_BASE_URL` 开关,让全量 integration test 打到本地 docker 容器。

**Tech Stack:** Python 3.14, uv, hatchling, Docker (multi-stage, python:3.14-alpine), GitHub Actions, pytest。

## Global Constraints

- Python 最低版本:`requires-python = ">=3.14"`(应用非库,可直接抬地板)。
- ruff `target-version = "py314"`。
- CI 只测 Python `3.14`(彻底 3.14 项目)。
- Docker 两个 stage 基础镜像都用 `python:3.14-alpine`。
- **不动任何真实依赖**(`dependencies` 列表不变),**不改任何应用代码**(`src/` 只读)。
- **不上 distroless**(YAGNI,留待后续评估)。
- 分支:`chore/python-314-and-slim-image`(已创建,勿在 master 改动)。
- 已实测事实:① 全部依赖在 python:3.14.6-alpine 下装成并 import OK;② 剥除 pip/setuptools/pkg_resources 后 `router_maestro.server:app` 仍能 import(无运行时依赖它们)。
- conftest 的临时改动**不进 PR**(验证后回退)。

---

## File Structure

| 文件 | 动作 | 责任 |
|---|---|---|
| `pyproject.toml` | Modify | requires-python / classifiers / ruff target |
| `.github/workflows/ci.yml` | Modify | 测试矩阵与 lint 的 Python 版本 |
| `.github/workflows/release.yml` | Modify | test / pypi-build 的 Python 版本 |
| `uv.lock` | Regenerate | `uv lock` 更新 requires-python 约束 |
| `Dockerfile` | Modify | 基础镜像升级 + venv 清理 + healthcheck 换 urllib |
| `integration_tests/conftest.py` | Modify(临时,不进 PR) | 加 `RM_INTEGRATION_BASE_URL` 开关 |

---

## Task 1: Python 3.14 元数据 + CI 升级(Part A)

**Files:**
- Modify: `pyproject.toml:7`(requires-python), `pyproject.toml:17-20`(classifiers), `pyproject.toml:68`(ruff target)
- Modify: `.github/workflows/ci.yml:24`(lint python), `.github/workflows/ci.yml:41`(matrix)
- Modify: `.github/workflows/release.yml:61`(test python), `.github/workflows/release.yml:123`(pypi python)
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: 无(纯配置)
- Produces: 一个 requires-python `>=3.14` 的项目;后续 Task 2 的 Dockerfile 与此版本对齐。

- [ ] **Step 1: 改 `pyproject.toml` requires-python**

把 `pyproject.toml:7`
```toml
requires-python = ">=3.11"
```
改为
```toml
requires-python = ">=3.14"
```

- [ ] **Step 2: 改 `pyproject.toml` classifiers**

把 `pyproject.toml:17-20`
```toml
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
```
改为
```toml
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.14",
    "Topic :: Software Development :: Libraries :: Python Modules",
```

- [ ] **Step 3: 改 `pyproject.toml` ruff target-version**

把 `pyproject.toml:68`
```toml
target-version = "py311"
```
改为
```toml
target-version = "py314"
```

- [ ] **Step 4: 改 `.github/workflows/ci.yml`**

把 `.github/workflows/ci.yml:24`
```yaml
        run: uv python install 3.12
```
改为
```yaml
        run: uv python install 3.14
```
把 `.github/workflows/ci.yml:41`
```yaml
        python-version: ["3.11", "3.12"]
```
改为
```yaml
        python-version: ["3.14"]
```

- [ ] **Step 5: 改 `.github/workflows/release.yml`**

该文件有两处 `run: uv python install 3.12`(约在第 61 行 test job、第 123 行 publish-pypi job)。两处都改为:
```yaml
        run: uv python install 3.14
```
用 `grep -n "uv python install 3.12" .github/workflows/release.yml` 确认命中两处,全部替换。

- [ ] **Step 6: 重新生成 uv.lock**

Run: `uv lock`
Expected: `uv.lock` 内 `requires-python = ">=3.14"`;命令退出码 0。

- [ ] **Step 7: 用 3.14 跑单元测试 + lint**

Run:
```bash
uv python install 3.14
uv run --python 3.14 pytest tests/ -q
uv run --python 3.14 ruff check src/ tests/
uv run --python 3.14 ruff format --check src/ tests/
```
Expected: pytest 全绿(508+ passed);ruff check 无 error;format --check 无 diff。

若 `format --check` 报 diff,运行 `uv run ruff format src/ tests/` 后重跑;把格式化改动纳入本 Task 的提交。

- [ ] **Step 8: 校验元数据一致**

Run: `grep -n "3.11\|3.12\|py311" pyproject.toml`
Expected: 无输出(3.11/3.12/py311 已全部清除)。

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock .github/workflows/ci.yml .github/workflows/release.yml
git commit -m "chore: upgrade to Python 3.14 (floor >=3.14, CI 3.14 only)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Dockerfile 升级 3.14 + 镜像瘦身(Part B)

**Files:**
- Modify: `Dockerfile`(builder 基础镜像:3、runtime 基础镜像:31、runtime apk:36-38、venv COPY 前清理、HEALTHCHECK:60-61)

**Interfaces:**
- Consumes: Task 1 的 `pyproject.toml`(requires-python >=3.14)。
- Produces: 一个 python:3.14-alpine 的多阶段镜像,venv 已剥除 pip/setuptools/pkg_resources,healthcheck 用 urllib,运行时不装 curl。

- [ ] **Step 1: 改 builder 基础镜像**

把 `Dockerfile:3`
```dockerfile
FROM python:3.11-alpine AS builder
```
改为
```dockerfile
FROM python:3.14-alpine AS builder
```

- [ ] **Step 2: builder 安装依赖后清理 venv**

把 `Dockerfile:26-28`
```dockerfile
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache .
```
改为
```dockerfile
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache .
# Strip build-time-only tooling not needed at runtime (~26MB): pip, setuptools,
# pkg_resources. Verified no runtime dependency imports them.
RUN rm -rf /opt/venv/lib/python*/site-packages/pip \
           /opt/venv/lib/python*/site-packages/pip-*.dist-info \
           /opt/venv/lib/python*/site-packages/setuptools \
           /opt/venv/lib/python*/site-packages/setuptools-*.dist-info \
           /opt/venv/lib/python*/site-packages/pkg_resources \
           /opt/venv/lib/python*/site-packages/_distutils_hack \
    && find /opt/venv -depth -type d -name __pycache__ -exec rm -rf {} + \
    && find /opt/venv -name '*.pyc' -delete
```

- [ ] **Step 3: 改 runtime 基础镜像**

把 `Dockerfile:31`
```dockerfile
FROM python:3.11-alpine
```
改为
```dockerfile
FROM python:3.14-alpine
```

- [ ] **Step 4: runtime 层去掉 curl**

把 `Dockerfile:35-38`
```dockerfile
# Install runtime dependencies only (no build tools)
RUN apk add --no-cache \
    libffi \
    curl
```
改为
```dockerfile
# Install runtime dependencies only (no build tools).
# curl removed: healthcheck uses Python stdlib urllib instead (~4MB saved).
RUN apk add --no-cache libffi
```

- [ ] **Step 5: healthcheck 换 urllib**

把 `Dockerfile:60-61`
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
```
改为
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=5).status==200 else 1)"]
```

- [ ] **Step 6: 构建镜像**

Run: `docker build -t router-maestro:314-slim .`
Expected: 构建成功,退出码 0。tiktoken/pydantic-core/rapidfuzz 在 builder 阶段编译(耗时数分钟属正常)。

- [ ] **Step 7: 校验体积下降**

Run:
```bash
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep router-maestro
```
Expected: `router-maestro:314-slim` 明显小于 `router-maestro:baseline-311`(126MB baseline → 预期 ~92MB;至少下降 >20MB)。若未下降,`docker history router-maestro:314-slim` 排查。

- [ ] **Step 8: 冒烟——容器内 import + 无 pip**

Run:
```bash
docker run --rm --entrypoint sh router-maestro:314-slim -c \
  'python -c "from router_maestro.server import app; print(\"OK\")" && ! command -v pip && ! command -v curl && echo "SLIM OK"'
```
Expected: 打印 `OK` 与 `SLIM OK`(app 可 import,pip/curl 均不存在)。

- [ ] **Step 9: Commit**

```bash
git add Dockerfile
git commit -m "build: base on python:3.14-alpine and slim runtime image ~30MB

Strip pip/setuptools/pkg_resources from venv; replace curl healthcheck
with stdlib urllib.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: 全量 integration test 打到 docker 容器(验证,conftest 改动不进 PR)

**Files:**
- Modify(临时): `integration_tests/conftest.py:416-454`(`live_server` fixture)

**Interfaces:**
- Consumes: Task 2 产出的 `router-maestro:314-slim` 镜像。
- Produces: 无正式产物;仅验证结果。conftest 改动在 Step 8 回退。

**前置:** 本机已 `router-maestro auth login github-copilot`(否则 integration test 会 skip)。宿主机 `~/.config/router-maestro` 与 `~/.local/share/router-maestro` 存在且含 Copilot auth。

- [ ] **Step 1: 给 `live_server` fixture 加外部 URL 分支**

在 `integration_tests/conftest.py` 的 `live_server` fixture(约 416-454 行)最开头、`_require_github_copilot_auth()` 之后,插入外部 URL 短路分支。把:
```python
@pytest.fixture(scope="session")
def live_server() -> Iterator[LiveServer]:
    """Start a local RM server against the user's existing config/auth files."""
    _require_github_copilot_auth()

    api_key = os.environ.get("RM_INTEGRATION_API_KEY") or get_current_context_api_key()
    if not api_key:
        api_key = DEFAULT_API_KEY
```
改为:
```python
@pytest.fixture(scope="session")
def live_server() -> Iterator[LiveServer]:
    """Start a local RM server against the user's existing config/auth files."""
    _require_github_copilot_auth()

    api_key = os.environ.get("RM_INTEGRATION_API_KEY") or get_current_context_api_key()
    if not api_key:
        api_key = DEFAULT_API_KEY

    # TEMP (not for PR): reuse an already-running server (e.g. docker container)
    external_url = os.environ.get("RM_INTEGRATION_BASE_URL")
    if external_url:
        base_url = external_url.rstrip("/")
        with httpx.Client(timeout=2.0) as probe:
            deadline = time.time() + STARTUP_TIMEOUT_SECONDS
            while time.time() < deadline:
                try:
                    if probe.get(f"{base_url}/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
            else:
                pytest.fail(f"External server at {base_url} did not become healthy")
        yield LiveServer(base_url=base_url, api_key=api_key, process=None)  # type: ignore[arg-type]
        return
```
(其余原有 `port = _find_free_port()` 起子进程逻辑保持不变,作为默认分支。)

- [ ] **Step 2: 确认无外部 URL 时默认行为不变**

Run: `uv run python -c "import ast; ast.parse(open('integration_tests/conftest.py').read()); print('parse OK')"`
Expected: `parse OK`(语法正确)。逻辑上 `RM_INTEGRATION_BASE_URL` 未设时走原分支。

- [ ] **Step 3: 用本地配置/auth 启动容器**

Run:
```bash
docker rm -f rm-e2e 2>/dev/null; \
docker run -d --name rm-e2e -p 18080:8080 \
  -e ROUTER_MAESTRO_API_KEY="router-maestro-integration-test" \
  -v "$HOME/.config/router-maestro:/home/maestro/.config/router-maestro:ro" \
  -v "$HOME/.local/share/router-maestro:/home/maestro/.local/share/router-maestro:ro" \
  router-maestro:314-slim
```
Expected: 打印容器 ID。

注意:镜像内用户为 `maestro`(uid 1000),HOME=`/home/maestro`。挂载只读避免测试污染本地状态。若容器因权限/路径无法读取 auth,改用可写挂载(去掉 `:ro`)或 `docker cp` 注入 auth.json 后重试。

- [ ] **Step 4: 等容器 healthy**

Run:
```bash
sleep 5 && curl -fsS -H "Authorization: Bearer router-maestro-integration-test" \
  http://127.0.0.1:18080/health && echo " <- health OK" \
  || docker logs rm-e2e --tail 50
```
Expected: `/health` 返回 200(`health OK`)。失败则打印容器日志排查(常见:auth 未挂上 → 换可写挂载或 docker cp)。

- [ ] **Step 5: 跑全量 integration test 打到容器**

Run:
```bash
RM_INTEGRATION_BASE_URL=http://127.0.0.1:18080 \
RM_INTEGRATION_API_KEY=router-maestro-integration-test \
uv run pytest integration_tests/ -v
```
Expected: 测试全部 passed 或(无相关模型时)skipped;**无 failed/error**。请求全部命中容器(可 `docker logs -f rm-e2e` 旁证)。

若大面积 skip,说明容器没读到 Copilot auth——回到 Step 3 用可写挂载或 docker cp 注入 auth 后重跑。

- [ ] **Step 6: 记录结果**

把镜像体积(Task 2 Step 7)与 integration 结果(passed/skipped 计数)记录到 PR 描述用的笔记里(可写入 `docs/python-314-and-slim-image-design.md` 末尾的「验证结果」小节)。

- [ ] **Step 7: 清理容器**

Run: `docker rm -f rm-e2e`
Expected: 打印 `rm-e2e`。

- [ ] **Step 8: 回退 conftest 临时改动(不进 PR)**

Run:
```bash
git checkout -- integration_tests/conftest.py
git status --short
```
Expected: `conftest.py` 恢复原状,`git status` 不含 conftest 改动。

---

## Task 4: 收尾 + PR

- [ ] **Step 1: 全量单元测试最后一遍**

Run: `uv run --python 3.14 pytest tests/ -q`
Expected: 全绿。

- [ ] **Step 2: 确认工作树只含应进 PR 的文件**

Run: `git status --short && git log --oneline master..HEAD`
Expected: 无未追踪的临时改动(conftest 已回退);提交历史含 design/Task1/Task2 三个 commit。

- [ ] **Step 3: 推分支 + 开 PR**

前置(项目记忆):`gh auth switch --user MadSkittles`;git/gh 需系统代理 127.0.0.1:7890。
```bash
git push -u origin chore/python-314-and-slim-image
gh pr create --title "chore: Python 3.14 upgrade + slim Docker image (~30MB)" \
  --body "见 docs/python-314-and-slim-image-design.md。镜像 126MB→~92MB;全量 integration test 已对 docker 容器跑通。"
```
Expected: PR 创建成功,返回 URL。

---

## Self-Review

**1. Spec coverage:**
- Part A(requires-python/classifiers/ruff/ci/release/uv.lock)→ Task 1 全覆盖 ✅
- Part B(基础镜像 3.14 / 剥 pip+setuptools / 去 curl+urllib healthcheck / 清 pycache)→ Task 2 全覆盖 ✅
- 验证(临时 conftest 开关 / build docker / 挂本地配置起容器 / 全量 integration 打容器 / 跳过本地直起)→ Task 3 全覆盖 ✅
- 交付物边界(conftest 不进 PR)→ Task 3 Step 8 + Task 4 Step 2 ✅

**2. Placeholder scan:** 无 TBD/TODO;每个代码步骤含完整 before/after;docker 失败路径给了具体回退(可写挂载/docker cp)。✅

**3. Type consistency:** `LiveServer(base_url, api_key, process)` 字段与 conftest 定义一致(external 分支传 `process=None`);环境变量名 `RM_INTEGRATION_BASE_URL` 在 Task 3 Step 1/5 一致;镜像 tag `router-maestro:314-slim` 在 Task 2/3 一致。✅
