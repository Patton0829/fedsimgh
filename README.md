# FedSimGH · 三节点 CPU 故障诊断 Demo

基于历史相似度引导的梯度协调联邦故障诊断最小演示。**纯 CPU、三个 Slurm 计算节点、约 2.9 MiB 数据**，浏览器显示训练日志，以及每条测试信号的故障类别。

保留 PFLlib `FedSimGHHistNoSize` 的冲突投影、协调后方向的历史相似度 EMA、相似度加权聚合和不按样本量加权的行为。诊断网络改为轻量 1D CNN，并用 GroupNorm 替代 BatchNorm，避免导出模型缺少客户端运行统计量。此项目演示算法流程，不复现论文的完整模型规模和实验指标。

## 三节点部署

前提：Linux、推荐 Python 3.11、三个 CPU 计算节点；**仓库、虚拟环境和 `runtime/` 必须位于所有节点相同路径的共享目录**。Web 进程运行在允许长期启动轻量服务且可以调用 `sbatch/squeue/scancel` 的主机上。计算由 Slurm 作业完成，Web 进程不加载 PyTorch、不在登录节点训练。

```text
浏览器 → Web 控制台（提交作业 / 读取结果）
                  │ sbatch，srun
                  ├─ 计算节点 0：客户端 0 + 全局协调端
                  ├─ 计算节点 1：客户端 1
                  └─ 计算节点 2：客户端 2
                     ↕ 共享目录交换模型更新与日志
```

在共享目录执行：

```bash
git clone https://github.com/Patton0829/fedsimgh.git
cd fedsimgh

# 如果默认 python3 不是所需版本，可指定 PYTHON_BIN=python3.11。
bash scripts/setup.sh

# 根据集群需要填写；不需要的变量不要设置。
export SLURM_PARTITION=cpu
# export SLURM_ACCOUNT=your_account
# export SLURM_QOS=normal

.venv/bin/python app.py --mode slurm --host 127.0.0.1 --port 8000
```

仓库也支持 SSH 地址 `git@github.com:Patton0829/fedsimgh.git`；HTTPS 拉取公开仓库无需登录 GitHub。

在自己电脑另开终端，转发 Web 端口：

```bash
ssh -N -L 8000:127.0.0.1:8000 your_user@your_server
```

打开 `http://127.0.0.1:8000`，点击「开始训练与诊断」。如果服务器本身允许直接访问 Web 服务，也可用 `--host 0.0.0.0` 并通过服务器地址访问。停止终端中的 Web 服务不会取消已提交的 Slurm 作业，重启服务后可重新查看最新任务。

Slurm 脚本在 [`slurm/train.sbatch`](slurm/train.sbatch)：固定 `3 nodes × 1 task/node`，默认每任务 2 CPU、每节点 2 GiB 内存、20 分钟上限，无 GPU 资源申请。Web 的 `--threads 2` 同时控制作业 `--cpus-per-task` 和 PyTorch 线程数。根据集群队列修改脚本中的时间、内存；分区/账户/QoS 使用环境变量即可。

必须一次分配三个不同节点，程序会检查主机名。客户端 0 与协调端共用节点 0，因此总共只占用三个计算节点。节点间无需开放应用端口，但共享目录必须能及时看到原子写入的文件。

## 页面提供什么

- 配置全局轮次、本地轮次、学习率、相似度系数、历史平滑系数，提交/停止作业。
- 查看 Slurm 作业号、排队原因、节点主机名、每轮客户端 loss、冲突投影次数、历史相似度和聚合权重。
- 显示准确率曲线和 Macro-F1；每轮评估只是观察，不据此选择最佳模型。
- 训练完成后，用**最后一轮全局模型**对全部 80 条测试信号逐条预测，输出「正常 / 内圈故障 / 外圈故障 / 滚动体故障」。
- 前端显示每条预测的日志、真实类别、模型概率、正确/错误、信号波形及四类概率；支持筛选、分页、CSV 与原始日志下载。

波形为了页面展示抽取为 128 点，训练和预测始终使用完整 2,048 点。当前页面使用内置测试集，不提供任意设备信号的通用诊断接口。预测概率不是正确率；模型不包含未知故障检测或寿命预测。

## 小数据集

`data/` 已随 Git 提交，无需额外下载完整数据集。

| 项目 | 数量 |
| --- | ---: |
| 客户端 | 3 |
| 每客户端训练窗口 | 96 |
| 全部训练窗口 | 288 |
| 测试窗口 | 80（每类 20） |
| 每窗口 | 2048 点，单通道，float32 |
| 整个 data 目录 | 约 2.9 MiB |

数据取自用户 PFLlib 仓库中预处理的 HUST 轴承振动信号、Y 通道；保留其逐窗口标准化：`(x - mean) / max(std, 1e-6)`。分别从原训练、测试分片抽样，不重新混合两种分片，再把训练子集划成三个逻辑客户端。三个客户端覆盖四类但类别比例不同，以产生更新方向差异。这是重新划分的演示客户端，并非声称对应三个真实工厂。

`data/manifest.json` 记录随机种子、类别、上游文件/样本索引、原划分策略和 NPZ 校验值。NPZ 仅包含普通数值数组，运行时 `allow_pickle=False`。数据来源说明见 [`data/README.md`](data/README.md)。

如需从原仓库重新抽样：

```bash
.venv/bin/python scripts/prepare_data.py --source ../pfllib/dataset/hust
```

## 没有 Slurm 时本地验证

直接在终端训练并逐条输出故障诊断（不启动网页、不占用端口）：

```bash
.venv/bin/python -u train.py --rounds 100 --threads 1
```

终端实时打印每轮训练 loss、冲突投影次数、历史相似度权重和准确率，随后打印全部 80 条信号的预测故障类别、概率、真实类别和正确与否。最后显示 `predictions.csv` 和 `model.pt` 的完整路径。默认产物存于 `runtime/terminal/<任务编号>/`，与网页任务分开；按 `Ctrl+C` 停止本次任务。可通过 `--epochs`、`--lr`、`--tau`、`--history-lambda` 调整参数。

终端入口也支持 `--mode slurm`，使用相同的三节点 Slurm 提交配置，在终端跟踪排队与训练日志。

如需使用网页：

```bash
.venv/bin/python app.py --mode local --threads 1 --port 8000
```

若提示 `Address already in use`，说明端口已被占用，可改用 `--port 8001`；终端入口没有端口要求。

这个模式启动三个独立 CPU 进程，使用与 Slurm 完全相同的 worker、聚合器和文件协议；仅用于验证，页面会标明「本地验证」。终端和网页默认 100 全局轮（可配置 1–1000 轮）、每客户端每轮 1 个本地 epoch、batch size 16、SGD 学习率 0.03、τ=3、λ=0.6。低轮次只用于流程演示，不保证分类准确率。服务器耗时取决于 CPU、共享存储和排队情况。

## 输出和排错

每次任务存于 `runtime/<任务编号>/`；运行产物不提交 Git：

```text
config.json         本次超参数
submission.json     Slurm 作业号 / 本地进程信息
state.json          训练状态、各轮评估、节点信息
events.jsonl        前端读取的结构化训练与逐条诊断日志
rank-0.log          协调端及客户端 0 原始输出
rank-1.log          客户端 1 原始输出
rank-2.log          客户端 2 原始输出
slurm-<job>.log     Slurm batch 输出（Slurm 模式）
model.pt            最后一轮模型 state_dict、类别和配置
predictions.json    80 条逐条诊断结果
predictions.csv     便于下载的中文 CSV
```

- 一直排队：检查页面中的 Slurm 状态和 `squeue -j <作业号>`，确认三个节点可同时分配。
- 提示找不到 Python/模块：检查 `.venv/bin/python` 在三个节点均可访问，且与计算节点系统架构兼容。
- 等待文件超时：检查共享目录、节点日志和作业退出状态。每轮等待默认上限 300 秒。
- 执行 `sacct -j <作业号> --format=JobID,State,ExitCode`，定位 `TIMEOUT`、`OUT_OF_MEMORY`、`FAILED`。
- 页面可「下载日志」。停止操作会写停止标记；Slurm 模式同时调用 `scancel`。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
node --check web/app.js   # Node 仅用于可选语法检查，部署运行不需要 Node
bash -n scripts/setup.sh slurm/train.sbatch
```

测试涵盖投影的解析结果、EMA 按客户端保留、零更新、数据校验与训练/测试隔离、Slurm 提交/取消接口，以及三个真实本地进程完成训练、保存后重新加载模型并核对全部 80 条预测。Slurm 命令测试使用模拟调度器；实际三台服务器上的调度需要在你的集群确认。

算法改编来源：[`Patton0829/pfllib`](https://github.com/Patton0829/pfllib) 的 `system/flcore/servers/serversimghhistnosize.py`。代码许可和来源保留见 `LICENSE`、`NOTICE`。
