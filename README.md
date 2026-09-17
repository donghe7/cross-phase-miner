# CrossPhase Miner

从多台机器人的信号灯检测中学习路口周期，并在网页中查看机器人位置、实时观测、模型预测和学习进度。当前以本机实验为主要用途。

## 启动

要求 Python 3.10+；启用本机 PostgreSQL / MQTT 时需要 Docker Compose。以下命令均在仓库根目录执行。

```bash
# 可使用已配置好的 conda 环境，或自行创建 .venv
python3 -m pip install -e '.[server,experiment]'
./run.sh --postgres --mqtt
```

访问 <http://127.0.0.1:8000>。不使用 Docker 时运行 `./run.sh`，使用 SQLite 与 HTTP。

只需根目录的 `run.sh` 即可启动后端、车队和网页。离线实验使用 `experiments/run.sh`，不是另一个在线启动步骤。
默认保留数据；`--fresh` 才会清空应用数据。Ctrl-C 停止后端和车队，PostgreSQL / MQTT 容器继续运行。

## 当前行为

- 到访可靠上报：本地持久化补传、HTTP／MQTT 落库确认，服务重启后继续处理学习任务。
- 检测频率：每台机器人 15 FPS（仿真时间），上报合批；支持 MQTT 或 HTTP。
- 仿真速度：1×、2×、5×、10×、20×。
- 模型可预测条件：当前时段至少 6 次跳变、置信度 ≥30%，且周期有效。
- 观测单帧接收门槛和车队共识门槛默认均为 70%，可独立配置。
- 实时观测与模型预测左右显示；学习进度、到访记录与剩余侦察次数估算可查看。
- 机器人列表固定每页 5 台，可通过全部机器人编号直接定位、翻页和跟随。

## 机器人通行规则

机器人到达路口后按模式决定何时通行（`server/fleet.py` 的 `Robot.run_arrival`）：

- **normal 模式**（服务器已有可预测模型）：不等待完整周期；绿灯以置信度 ≥0.70 的帧连续确认 5 秒（`CONFIRM_SECONDS`）即通行。
- **scout 模式**（无可信模型，驻留测量周期）：驻留到确认第二次 RED→GREEN 跳变（即测满一个完整周期）为止，随后在确认为绿灯的帧直接通行，不再等 5 秒连续确认。
- 颜色确认需经过去抖：新颜色需持续 ≥2 秒（`TRANSITION_MIN_DURATION`）才记为跳变。
- 超时未满足条件则放弃通行：scout 450 秒（`SCOUT_TIMEOUT_SECONDS`）、normal 300 秒，按 `TIMEOUT` 上报。

## 目录

```text
├── run.sh                 # 在线启动入口（后端 + 车队 + 网页控制台）
├── crossphase_miner/      # 当前算法、离线仿真、评估与绘图
├── server/                # 在线 API、车队、持久化、MQTT、静态网页
│   ├── static/
│   └── compose.*.yaml     # 本机 PostgreSQL / Mosquitto
├── tests/                 # 单元、存储契约及可选端到端检查
│   ├── server/
│   └── e2e/
├── scripts/               # 隔离数据库测试、规模基准
├── docs/                  # 当前运行、开发、架构文档
├── experiments/
│   ├── run.sh             # 离线实验入口
│   ├── archive/           # 历史源码、报告、五组实验结果及 SHA-256 清单
│   └── runs/              # 新离线实验输出（忽略提交）
├── pyproject.toml         # Python 包、依赖分组、代码质量配置
└── Makefile               # 常用开发命令
```

现有 `server_state.sqlite3`、`server/.env.postgres`、`server/backups/` 保留原路径并忽略提交。
数据不会打入 Python 安装包。

## 开发入口

```bash
python3 -m pip install -e '.[server,experiment,dev]'
npm ci                      # 仅格式检查使用；网页运行不需要 Node
make check                  # Python/JS/CSS/HTML 检查与单元测试
make test-postgres          # 在随机独立 schema 中验证存储契约
make build                  # 构建 wheel 与源码包
```

[运行与协议](docs/operations.md) · [开发与验证](docs/development.md) ·
[架构与边界](docs/architecture.md) · [历史实验](experiments/README.md) ·
[原始需求 SPEC](SPEC.md)（历史设计，不代表当前全部已实现）
