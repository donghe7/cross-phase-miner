# CrossPhase Miner

多机器人信号灯观测、周期学习与实验控制台。可安装工程位于 [`project/`](project/README.md)。

```bash
cd project
python3 -m pip install -e '.[server,experiment]'
./server/run_demo.sh --postgres --mqtt
```

打开 <http://127.0.0.1:8000>。仅需在线演示时，无需运行 `run.sh`。

- [安装与工程目录](project/README.md)
- [运行、数据库、MQTT 与界面说明](project/docs/operations.md)
- [开发与验证](project/docs/development.md)
- [架构与边界](project/docs/architecture.md)
- [历史版本与实验归档](project/experiments/README.md)
- [原始需求 SPEC](SPEC.md)（历史设计，不代表当前全部已实现）
