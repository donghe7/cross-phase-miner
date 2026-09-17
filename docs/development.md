# 开发与验证

在仓库根目录执行。Python 依赖统一维护于 `pyproject.toml`：核心只需 NumPy，
`server` 包含 API、PostgreSQL 与 MQTT，`experiment` 包含绘图，`dev` 包含 Ruff 与 build，
`browser` 包含 Playwright。

```bash
python3 -m pip install -e '.[server,experiment,dev,browser]'
npm ci
python3 -m playwright install chromium
make check
```

Ruff 检查未定义名称、未使用导入、语法与导入顺序，并统一 Python 格式。
Prettier 使用 npm lockfile 固定版本，检查原生 HTML/CSS/JS；不引入前端构建步骤。
`make format` 可统一格式；`make build` 构建包含网页静态资源的 wheel 和源码包。
包内仅包含当前运行代码，不包含测试、归档、数据库或密钥。

## 测试分层

| 命令 | 覆盖与隔离 |
|---|---|
| `make test` | 核心学习、因果性、时钟、实时状态、学习进度、SQLite、MQTT 协议与合批时序；默认跳过 PostgreSQL |
| `make test-postgres` | 读取本机既有配置，或使用 `CP_TEST_POSTGRES_URL`；每个存储测试使用随机独立 schema |
| `make test-browser` | 启动临时 SQLite 后端，检查 20 台机器人分页、跟随、固定尺寸、移动端与五档倍速 |
| `make test-mqtt` | 连接既有 broker，在唯一 topic 前缀下验证上报、重复到访、订阅查询和 SQLite / PostgreSQL 持久化 |
| `make benchmark` | 在临时 SQLite 中测量状态内存、摄取吞吐与学习开销 |
| `./run.sh v1 test` | 当前离线核心测试 |
| `./run.sh v2 test` | 从归档解压并隔离执行历史 v2 测试 |

数据库集成测试需要可创建 schema 的实验账号。它们只删除自己创建的随机 schema，
不清空演示表。浏览器测试清除继承的 `CP_*` 和 `DATABASE_URL` 配置，使用独立临时数据库；
截图写入忽略提交的 `test-results/`。需要分开使用浏览器和服务端 Python 时，
可设置 `CP_SERVER_PYTHON=/path/to/python`。

CI 位于仓库根目录 `.github/workflows/check.yml`，运行格式、单元、浏览器和打包检查，
并用独立 PostgreSQL / Mosquitto 服务运行存储与消息集成检查。常规单元测试不要求 Docker。

## 修改约定

- 修改业务规则时同步对应测试与当前文档；历史归档保持不可变。
- 实时观测、模型预测与仿真真值分别处理，避免真值进入学习路径。
- 数据库状态变更应同时验证 SQLite 与 PostgreSQL 存储契约。
- 新实验写入 `experiments/runs/`；保存随机种子和输出日志。
- 数据库连接串、`.env`、备份、生成缓存不进入版本控制。

旧路径迁移：`crossphase_miner/main.py` → `tests/test_core.py`，
`server/test_*.py` → `tests/server/`，`server/bench_scale.py` → `scripts/bench_scale.py`，
旧 `server/e2e_check.py` 被隔离数据的 `tests/e2e/` 检查替代。
