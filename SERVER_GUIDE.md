# 在 4090D 上运行，并跟着输出理解代码

## 1. 环境与第一次运行

Linux 服务器，Python 3.10 或以上。先检查 `nvidia-smi`，再创建环境：

```bash
git clone https://github.com/Zzh0104/Genbid.git
cd Genbid
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
python -m pytest -q
python ogb.py --device cuda --updates 300 --output runs/smoke
```

若 `torch.cuda.is_available()` 为 False，按 https://pytorch.org/get-started/locally/ 的当前安装选择器安装匹配服务器驱动的 CUDA wheel。代码不需要 flash-attn、transformers 或 CUDA 编译扩展。

这是小模型，合成数据上 CPU 也很快。4090D 不会自动解决数据分布、教师标签和仿真误差；先读输出，再增加模型大小。

## 2. 你最应该先看哪几个细节

按照以下顺序读函数，每一步配一个输出文件。

### A. 出价究竟是什么

`Replay.outcome`：模型输出倍率 m，不是每个 PV 独立生成 bid。PV 的 `bid=m×CPA×pValue`，达到固定市场价则竞得。预算不足时按 PV 顺序跳过无法支付的机会。读取 `rollout_example.csv` 的 action/cumulative_cost，看相同 PV 数据下三档 RTG 的消耗差异。

### B. bid* 标签从哪里来

`mpc`：当前状态下枚举 7×7 个未来两步出价序列。计算两步后的累计效用和 pacing 误差，选择最好序列的第一步，然后在下一个真实回放状态重新算。`trajectory` 用选择动作实际推进回放，生成新的 state/action/reward，整条轨迹一致。

读取 `teacher_example.csv`：last_multiplier 应与上一行 target_bid_multiplier 一致；cost_per_budget 单调不减；RTG[t]−RTG[t+1] 应等于 reward[t]（末行 RTG 等于末行 reward）。老师看到未来两步价格是离线 hindsight 标注，DT state 不看到这些价格。

### C. 为什么监督学习也被叫作决策 Transformer

`DT.forward`：输入 `(R0,s0,a0,R1,s1,a1,...)`。训练不是预测每个 token，而是在 s_t 的隐向量上回归 a_t。因果 mask 保证 s_t 只能看到 R_t、s_t 和之前的轨迹；标签 a_t 在后面，不能被抄答案。用 `test_no_action_or_future_leakage` 对照看这一点。

### D. teacher forcing 与真实 rollout 的差别

`batch`：训练使用教师过去动作；`Policy.act`：线上过去动作是 DT 自己产生的。`pred_vs_target.csv` 的 MSE 是 teacher forcing 验证，`cases.csv` 是闭环真实 rollout。这两个指标可能不一致：拟合好老师并不保证遇到自己造成的偏离状态时也好。

### E. RTG 如何更新

`Policy.act`：第一步目标是训练轨迹总效用的 80% 分位数，然后每步 `remaining -= previous_reward`。输入模型时除以训练 return_scale，但日志输出恢复原单位。看 `rollout_example.csv` 的 desired_rtg 与 reward，可追踪控制目标。RTG 高于训练覆盖范围可能产生错误动作，不能把 RTG 无限制加大。

### F. attention 说明了什么

`attention.npz` 每层形状是 `[1,heads,3K,3K]`，第 3t 个位置是 RTG，第 3t+1 是 state，第 3t+2 是 action；观察 state query 对哪些历史 token 关注。整个十维 state 已经过一层线性投影，attention 不是十个特征的重要性图。

## 3. 训练真实公开数据

从上游仓库下载并解压 period CSV，放入 data/traffic。数据需要至少三个完整投放周期。先用少量完整 period；不要截取 CSV 前 N 行，截断 trajectory 会使预算与 RTG 不正确。

```bash
python ogb.py --csv data/traffic/period-{7,8,9,10,11}.csv \
  --device cuda --updates 3000 --batch-size 128 \
  --width 128 --layers 3 --heads 4 --context 20 \
  --output runs/public_seed42
```

程序先在 CPU 构造 MPC 标签，GPU 训练会在这之后开始。数据读取目前每文件读入内存，且保存完整机会数组；关注系统 RAM，而不仅是显存。MPC 网格选中机会已缓存，预算足够时直接向量求和；预算不足仍有逐 PV 循环。大数据第一版先用少数周期，不建议直接全量 25GB。

轨迹缓存 `trajectories.pt` 在同一 output 下自动复用，并校验内容 hash、label 和 penalty。输入数据、lambda 或 label 改变时用新输出目录。当前严格要求 48 时段完整 CSV；提前终止日志不支持反事实全天复建，会明确报错。

## 4. 断点续训与输出

`model.pt` 保存验证集最佳模型；`last.pt` 保存最近训练模型、optimizer、随机数状态。`--updates` 是总目标步数，续训需要其余训练参数保持一致，并保留最佳 model.pt。

```bash
python ogb.py --csv data/traffic/period-{7,8,9,10,11}.csv \
  --device cuda --updates 6000 --batch-size 128 \
  --width 128 --layers 3 --heads 4 --context 20 \
  --output runs/public_seed42 --resume
```

`config.json` 保存运行参数；`history.json` 实时保存 train/val MSE；`report.json` 保存划分和测试汇总；`pred_vs_target.csv` 对比验证标签/预测；`teacher_example.csv` 展示教师轨迹；`rollout_example.csv` 展示第一条测试轨迹；`cases.csv` 是每个 episode 的闭环效果。

## 5. 建议实验顺序

先保持默认规模跑 smoke，再跑真实数据；固定同一数据/seed，各次修改一个变量，使用不同 output：

- `--context 1` 对比 8/20：长历史是否有用？
- `--label fixed` 对比 mpc：教师标签是否带来收益？fixed 是退化对照，不是完整历史日志 BC。
- `--penalty 0/1/2`：观察 CPA 违约与转化如何变化。若数据本身没有超成本压力，lambda 对比可能不显著。
- `--seed 42/43/44`：看收益是否稳定。

三档 RTG 会自动在相同测试集回放；先比较效用、转换数、成本和约束，再看 MSE。不要通过测试集挑模型后把同一测试集当最终结论。

当前重要边界：两步 hindsight MPC 不是生产预测 MPC；固定阈值不模拟竞争反馈；期望转化不模拟延迟回传；十维状态不是内部 schema；没有 CBO 多广告组共享预算。优先通过以上输出理解算法，效果跑出来后再判断需要扩展哪个环节。
