# 运行、数据库与消息协议

从仓库根目录执行。

## 1. 一分钟跑起来

| 命令 | 作用 |
|------|------|
| `./server/run_demo.sh --postgres` | 自动启动 PostgreSQL 容器，再启动后端 + 20 台机器人 + 控制台 |
| `./server/run_demo.sh --postgres --mqtt` | 同时启动本机 MQTT broker，机器人改用 MQTT 上报并订阅时钟 |
| `./server/run_demo.sh` | 不启用 Docker，默认使用原 SQLite 文件；设置 DATABASE_URL 时连接该数据库 |
| `./server/run_demo.sh --intersections 5000 --robots 40 --speed 20` | 放大规模 |
| `./server/run_demo.sh --resume` | 保留数据（现在已是默认行为；旧命令仍兼容） |
| `./server/run_demo.sh --postgres --fresh` | 显式清空 PostgreSQL 中 CrossPhase 的六张表，重新实验 |
| `make test-mqtt` | 真实 MQTT + SQLite/PostgreSQL 学习与持久化验收 |
| `python3 -m scripts.bench_scale --intersections 20000` | 规模基准（内存 / 吞吐 / 学习开销 / LRU） |

控制台打开约 1–2 分钟后，配送区内带描边的格子会从灰色变成浅红/浅绿——
表示该路口已经有了可用的周期模型，点进去可以看学到的周期与真值的差。收敛时间取决于速度、路由、信号周期与样本质量。

### 本机 MQTT

```bash
python3 -m pip install -e '.[server]'
./server/run_demo.sh --postgres --mqtt
```

Mosquitto 容器仅映射到本机 `127.0.0.1:1883`，本机实验使用匿名连接。
PostgreSQL 保存学习与到访数据；broker 负责消息传递，不替代数据库。
Ctrl-C 停止后端和车队，broker 保留运行；可用
`docker compose -f server/compose.mqtt.yaml down` 停止它。
连接自行管理的 broker 可设置 `CP_MQTT=1`、`CP_MQTT_HOST`、`CP_MQTT_PORT`，
可选 `CP_MQTT_USER`／`CP_MQTT_PASSWORD`，然后运行不带 `--mqtt` 的脚本，避免自动启动本机容器。
并行实验应为各后端／车队设置相同且独立的 `CP_MQTT_PREFIX`。

默认 topic 前缀是 `crossphase/v1`：

| Topic | 用途 |
|---|---|
| `robots/{robot_id}/observations` | 15 FPS 检测结果合批上传，QoS 0，不 retain |
| `robots/{robot_id}/travel` | 行程，QoS 1 |
| `robots/{robot_id}/arrivals` | 到访与跳变，QoS 1；数据库按报告 ID 去重 |
| `clock` | 订阅服务器仿真时钟，替代车队频繁 HTTP 轮询 |
| `crossings/{id}/state` | 订阅有观测历史路口的实时状态与模型；每 0.2 秒发布，带 `sim_now` |
| `query/{client_id}` | 发布 `{"request_id":"q1","intersection_id":"seongsu_station"}` 查询详情 |
| `replies/{client_id}` | 先订阅此 topic，再发查询；响应含对应 `request_id` 与 `detail` 或 `error` |

机器人消息使用 `{"session":"客户端会话ID","seq":1,"payload":{...}}` 信封，
`payload` 与相应 HTTP 接口一致。同一会话内每台机器人序号递增，重复或落后序号不会
回退实时位置；重发到访仍需使用原 `record_id`。实时消息不 retain，避免重连时回放旧灯色。
`GET /v1/transport` 可查看连接、接收量、错误和队列；控制台显示当前上传协议。
浏览器仍通过 HTTP／WebSocket 访问后端，模型和历史详情的 HTTP 查询继续可用。

当前为单后端本机实验实现：QoS 1 不等于数据库提交确认，尚无离线到访补传日志。
观测断流时仍会过期。MQTT 长连接减少连接开销，合批减少消息数；15 FPS 的检测数据量
不会因换协议自动减少。低延迟配置关闭 Nagle 合并，可能增加小 TCP 包数量。

### 本机 PostgreSQL

需要 Docker Engine、Docker Compose 及 Python 依赖，命令均从仓库根目录执行。
Compose 使用官方 `postgres:18` 镜像，数据库只绑定 `127.0.0.1:5432`。
首次运行自动生成 `server/.env.postgres`，内含随机密码，文件权限为 `0600` 并由 git 忽略。
账号和数据库名默认为 `crossphase`。后端仍在本机 conda 环境中运行，不需要容器化。

数据库文件位于 Docker 命名卷 `crossphase-postgres_postgres_data`，挂载到 PostgreSQL 18
要求的 `/var/lib/postgresql`。停止或重建容器会保留卷；不要用 `docker compose down -v`
删除需要保留的数据。持久化卷不是备份，应另外导出数据库。

```bash
./server/postgres.sh status   # 查看数据库容器状态
./server/postgres.sh psql     # 进入 SQL 控制台；用 \q 退出
./server/postgres.sh backup   # pg_dump 备份至 server/backups/*.dump
./server/postgres.sh down     # 停止并移除数据库容器，保留数据卷
./server/postgres.sh up       # 重新启动数据库
```

`Ctrl+C` 只停止后端和机器人，数据库继续运行；下次 `--postgres` 会连接已有数据库。
网页底部“观测与学习”显示当前使用 PostgreSQL 还是 SQLite。

如果 5432 已被占用，首次启动可指定 `CP_PG_PORT=55432 ./server/run_demo.sh --postgres`；
端口会写入配置文件。已有配置也可修改文件中的 `CP_PG_PORT` 后重启容器。
`CP_POSTGRES_ENV` 可指定另外的配置文件。此文件使用不带引号的 `KEY=value` 格式，
值只支持字母、数字、`_`、`-`。已有数据卷的用户／密码不会随环境变量变化而自动修改。

连接自己管理的 PostgreSQL 时，在环境中设置 `DATABASE_URL` 后运行 `./server/run_demo.sh`。
连接地址格式为 `postgresql://用户:密码@主机:端口/数据库`，特殊字符需要 URL 编码。
优先级为命令行 `--db` > `DATABASE_URL` > `CP_DB` > 默认 SQLite 文件；`--postgres`
显式选择本项目管理的本机容器，不能与 `--db` 同用。连接失败时直接报错，不会回退 SQLite。

### 原 SQLite 数据迁移（可选）

默认不自动导入旧文件。新 PostgreSQL 实验不会修改 `server_state.sqlite3`。
若要导入，先停止使用原 SQLite 的后端，**在目标数据库尚无 CrossPhase 数据时**执行：

```bash
./server/postgres.sh up
export DATABASE_URL="$(python3 -m server.postgres_env url)"
python3 -m server.migrate_sqlite --source server_state.sqlite3
unset DATABASE_URL
./server/run_demo.sh --postgres
```

迁移以只读方式读取 SQLite（包含 WAL），在一个事务中导入全部六张表；旧版本缺少
`red_measurements`、`model_history` 或 `visits` 时可导入其余已有表。
目标非空则拒绝覆盖，源文件保持不变。
若目标已有需要保留的实验，请另建空数据库，通过 `DATABASE_URL` 连接后再导入。
旧程序没有落盘的学习器状态、观测历史无法通过迁移补回。

### 保存范围与恢复

| 表 | 内容 |
|---|---|
| `intersections` | 路口名称、区域、示意坐标 |
| `models` | 各路口／时段的周期模型 |
| `learners` | 学习器参数与用于继续学习的样本 |
| `red_measurements` | 完整红灯时长的测量样本 |
| `model_history` | 各路口／时段首次建模、首次可预测、最近更新的时间与触发机器人 |
| `visits` | 已结束到访：机器人、侦察／普通模式、通行／超时、等待时间、上报跳变数量 |

PostgreSQL 的模型／学习器载荷使用 JSONB；SQLite 保留兼容的 JSON 文本格式。
每次学习完成后，受影响时段的模型、学习器、红灯测量值和模型里程碑在一个事务中提交。
到访记录在接收报告时落盘；报告 ID 去重，重复提交不会重复计数或重复学习。
SQLite 的同步设置改为 `FULL`。默认启动不会再清库。

`postgres.sh backup` 生成 PostgreSQL custom 格式备份，可用 `pg_restore` 恢复。
建议恢复到**新建的空数据库**验证结果，再通过 `DATABASE_URL` 切换后端连接；不要直接
覆盖唯一在用的数据库。备份应另存到其他磁盘或主机。

这仍是单后端的实验服务：机器人位置、实时投票、本次运行的吞吐计数、尚未处理的学习队列在内存中，
并非所有上传帧都有持久化历史。数据库重启后应重启后端以重新建立连接；暂不支持
运行中自动重连、持久消息队列或多后端共享学习状态。默认启动恢复模型，仿真时钟仍按
演示起点重新开始；要做独立、可复现的新实验请使用独立数据库或显式 `--fresh`。

存储与时钟回归：

```bash
python3 -m unittest server.test_store server.test_console_state server.test_clock server.test_learning_progress server.test_fleet_timing server.test_mqtt_transport -v
export CP_TEST_POSTGRES_URL="$(python3 -m server.postgres_env url)"
python3 -m unittest server.test_store -v
unset CP_TEST_POSTGRES_URL
```

PostgreSQL 测试会为每个用例创建并删除独立 schema，不清空现有实验表。

---

## 2. 架构

```
   机器人（20 台，各自一个线程）
        │  ① 2Hz 视觉观测，本地 0.70 置信度闸门
        │  POST /v1/observations   ← 批量（默认每 1 仿真秒一批）
        │
        │  ② 端侧 extract_transitions()，只上报 2~4 个相位跳变
        │  POST /v1/arrivals
        ▼
┌──────────────────────────── 总服务器 (FastAPI) ────────────────────────────┐
│  live_state.py   实时状态：置信度闸门 + 70% 车队共识 → observed/disputed   │
│  service.py      摄取 O(batch)，学习进后台队列（贝叶斯更新 5~15ms）        │
│  store.py        PostgreSQL / SQLite 持久化 + LRU 学习器工作集          │
│  city.py         1000+ 路口的真值世界（仅用于演示对照，不参与学习）        │
└──────────┬──────────────────────────────────────────────┬─────────────────┘
           │ WS /ws  每 0.5s 推送 1200 格快照（约 13 KiB）  │ GET /v1/predict/{id}
           ▼                                              ▼
   static/index.html（控制台）                     机器人决策 SCOUT / NORMAL
```

### 关键取舍：原始帧不进学习路径

检测频率默认 **15 FPS**，与实际机器人一致：每隔约 **66.7 ms 仿真时间**产生一帧检测。
车队启动时读取服务器的 `obs_rate_hz`，详情面板显示该值；可用 `CP_OBS_RATE` 覆盖。
倍速仅压缩真实运行时间，不改变每个仿真秒的检测帧数。检测结果仍按批上传，
检测频率与 HTTP 上传频率不同。
默认每 1 个仿真秒上传一次，启动时车队还会把上传间隔限制在投票窗口与过期门槛的
三分之一以内。15 FPS 检测不再被 5 秒的上传批次阻断；处理落后时一次补齐已发生的
检测，保持帧序列连续，不发送一串已经过期的小批次，也不提前发送未来帧。

一台机器人等待约 120 秒会产生约 1,800 帧；20 台均等待相同时长则约 36,000 帧。
让服务器吞下原始帧再提取跳变，在 1000 路口规模上是纯浪费。所以：

* **端侧提取**：机器人自己跑 `core/transitions.py`（去抖 2s + 0.70 置信度过滤），
  每次到达只上报 2–4 个跳变事件；服务器的学习输入量与观测频率**解耦**。
* **原始帧仍然上传**，但只服务于两件事：控制台的实时状态、以及多机器人共识。
  这条路径是无状态的 O(batch) 计数与投票，不触碰学习器。
* **学习异步**：单次贝叶斯更新实测中位数 12.9ms / p95 21.1ms，
  绝不能挂在 ingest 请求里。因此 `submit_arrival` 只入队，后台线程消费。

### TOD 与键设计

所有模型按 `(路口, TOD时段)` 建键，时段沿用 v1 的 `SIMPLE_TOD_PERIODS`（day/night）。
现场约束「绿灯时长跨时段恒定，只有红灯变」由 `_apply_shared_green()` 维持：
各时段用自己的 `T_cycle - T_red` 投票，按逆方差加权得到共享绿灯时长，再回写为
`T_red(时段) = T_cycle(时段) - T_green`，让夜间这种样本稀缺的时段继承精确红灯长度。

---


## 7. API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 控制台 |
| GET | `/v1/clock` | 仿真时钟（机器人本地推算用） |
| POST | `/v1/clock/speed` | `{"speed": 1\|2\|5\|10\|20}`，即时调整全局仿真速度，保持仿真时间连续 |
| GET | `/v1/config` | 当前阈值、规模、配送区/枢纽路口列表 |
| GET | `/v1/stats` | 摄取 / 学习 / 缓存计数器 |
| GET | `/v1/intersections` | 全部路口静态元数据（与快照同序） |
| GET | `/v1/intersections/{id}` | 实时状态 + 各时段模型 + 真值 + 相位 |
| GET | `/v1/predict/{id}` | 机器人真正调用的那个接口 |
| POST | `/v1/observations` | 一批观测（`{robot_id, intersection_id, mode, observations[]}`） |
| POST | `/v1/arrivals` | 一次到达的汇总 + 端侧提取的跳变 |
| POST | `/v1/travel` | 行程上报：`robot_id, destination_id, started_at, arrives_at`（仿真时间） |
| GET | `/v1/snapshot` | 全城紧凑快照（WebSocket 断线时轮询用） |
| WS | `/ws` | 同一份快照，0.5s 推送 |

紧凑快照里 `tiles` 是与 `/v1/intersections` 同序的 `[状态码, 距绿灯秒数]` 数组，
状态码：0 未知 / 1 观测到红 / 2 观测到绿 / 3 模型推算红 / 4 模型推算绿 / 5 机器人分歧。

---

## 8. 前端

`server/static/index.html` + `console.css` + `console.js`，无构建步骤、无外部字体依赖，
由 FastAPI 直接托管。

顶部可选择 **1×、2×、5×、10×、20×**。切换作用于整个仿真（所有页面与机器人），
不会重置仿真时间或已学模型；当前档位以服务器快照为准。机器人车队同步
服务器时钟的间隔随倍速调整（最多 0.1 秒真实时间且不超过 0.5 秒仿真时间），
正在行驶或等待观测上传的机器人也会按新速度继续等待。
断线或请求处理中会禁用按钮，切换失败时显示提示。初始速度仍由 `--speed` 指定。

* **机器人位置优先**：列表展示路口名称、状态、最近一帧灯色及置信度。点击机器人
  编号可持续跟随；行驶时展示“起点 → 目的地”和预计到达时间，到达后自动切换详情。
  全部机器人编号常驻在列表上方，可直接跳到对应页并跟随；每页固定 5 台，明确显示
  总数、页码和上一页／下一页。表格固定列宽与行高，跟随区固定高度；长名称省略显示，
  悬停可看全文。窄屏仅需横向查看表格列，更多机器人使用分页，不再藏在纵向滚动区域内。
* **路口与模型进度**：路口卡片直接标出 `R001 在此` 或 `→ R002 前往`。可切换配送区／
  全部路口，每页 48 个；全城色块概览保留。支持按名称、区域、ID 搜索。
  位置为上报路口／行程示意，**不是 GPS**；
  首次出发没有已知起点时明确显示“起点未上报”，过期位置标记为待更新。
* **threshold 常驻显示**：单帧接收门槛与车队共识门槛分别从 `/v1/config` 获取，
  不写死 70%。详情展示每台机器人的 `confidence / threshold`、接收／丢弃结果，
  以及车队加权一致率。单机 100% 一致率不代表视觉置信度为 100%。
* **实时与预测左右对照**：详情左栏显示机器人实时观测，右栏独立显示模型预测及预计倒计时，
  桌面和手机均并排展示。观测与预测灯色不一致时提示差异；没有有效观测／可靠模型时
  各自显示空状态，过期数据标记为待更新。详情的 `model_prediction` 独立于机器人决策，
  因此车队观测有分歧时仍能查看模型预测。学习周期与仿真真值对照保留。
  最近一帧可能被拒绝，而此前通过门槛的帧仍在共识窗口内，两者分别展示。
  新前端遇到旧后端缺少学习进度字段时，会提示重启后端，不显示虚假的达标进度。
* **学习进度与剩余侦察次数**：每个路口显示当前时段的模型置信度、已学习跳变数、
  首次生成／首次可预测时间。详情分别显示样本门槛（当前 6 次）与置信度门槛（当前 30%），
  直接从 `PeriodModel` 读取。进度条仅表示样本数量，两项都达标才可预测；置信度是
  学习器评分，不是视觉识别准确率。
  剩余次数优先按最近 20 次到访中相关侦察的本时段跳变平均产出估算，包含零产出的侦察；
  没有历史时按完整仿真侦察约 3–4 次跳变估算。历史产出为零或仅缺置信度时不承诺次数。
  **这只是补齐样本的估算，不保证再路过固定次数就能学到可靠模型**；普通路过可能没有
  跳变，日夜分别学习。在场采样于到访结束后上传，已上传待处理批次单独提示。
* **到访与建模记录**：区分不同机器人数量与累计到访次数，并显示侦察／普通、通行／超时。
  详情列出最近 20 次已结束到访的机器人、等待时长及上报跳变数。模型时间线包含仿真时间、
  实际记录时间、当时样本量／置信度和触发更新的机器人。数据写入 SQLite 或 PostgreSQL；
  升级自动增表，不清除已有模型。旧版本未记录的到访与首次建模时间不会补造。
* **数据时效**：所有上报年龄与行程时间使用仿真秒；离开路口后清除当前位置和旧观测，
  行驶目的地不会被当成已到达。断线展示最后快照提示，并自动重连／轮询。
  观测与位置的过期判定以各自服务器快照的时间为准，避免高倍速下在两次刷新之间
  被前端误判为断流；详情每 0.2 秒刷新。真正过期的观测及断线仍明确标记。
* 桌面和手机布局、键盘焦点及减少动画偏好均有处理。

升级后需同时重启 server 与 fleet，再刷新页面。保留已学模型可使用：

```bash
./server/run_demo.sh --postgres
```

快照 `robots[]` 新增 `updated_at`、`origin_id`、`destination_id`、
`travel_started_at`、`travel_arrives_at` 和 `observation`。
`observation` 保留最近一帧的路口、时间、灯色、置信度及接收结果，即使该帧未通过门槛。
旧机器人客户端不报行程时，只能展示最后上报路口，过期后会标记待更新。
`learning[id]` 提供已有到访／模型路口的学习摘要；详情 API 另含 `recent_visits`。
`POST /v1/arrivals` 新增可选 `record_id` 与 `action`（`CROSS`／`TIMEOUT`）；
旧客户端缺少 ID 时按机器人、路口、起止时间、模式和结果生成稳定 ID，缺少结果时默认通行。

服务状态与速度切换回归：`python -m unittest tests.server.test_console_state tests.server.test_clock -v`。

---

