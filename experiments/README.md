# 离线实验与历史归档

当前在线演示使用 `../server/run_demo.sh`。本目录用于离线实验和历史复现。

```bash
# 在仓库根目录执行；默认固定 seed=2024，可通过 SEED 覆盖
./run.sh v1
SEED=2025 ./run.sh v1 --robots 3
./run.sh v2 --scout
./run.sh v2 test
```

新结果位于 `runs/<v1|v2>/<时间戳>-<进程号>/`，包含仿真输入、随机种子日志和评估图表。

`archive/legacy-20260722.tar.gz` 保存整理前的两套算法源码、旧 README / 设计说明、
概念动画，以及历史 v2 的五组完整实验结果。`archive/manifest.json` 记录归档与每个文件的
SHA-256；归档完成后已逐文件校验。它不包含数据库、密钥、备份或 Python 缓存。
名称中的日期指历史实验日期，不是当前服务的版本日期。

历史 v2 有不同的 Scout 学习判据，因此保留可复现快照，不将其混入当前算法包。
`run.sh v2` 只解压所需源码到临时目录，退出后清理，不解压历史数据；输出仍写入本目录的 `runs/`。

```bash
# 从仓库根目录查看历史文件，或解压到自己选择的空目录
python3 -m scripts.verify_archive
tar -tzf experiments/archive/legacy-20260722.tar.gz
mkdir -p /tmp/crossphase-history
tar -xzf experiments/archive/legacy-20260722.tar.gz -C /tmp/crossphase-history
```

旧说明里的路径和数字对应当时版本。当前行为以项目 README 与 `docs/` 为准。
