# OGB：MPC + Decision Transformer 公开复现

**服务器训练与代码细读：先看 [SERVER_GUIDE.md](SERVER_GUIDE.md)。** 支持 CUDA、可配置模型/batch、轨迹缓存和断点续训；输出教师标签、逐步 rollout 和预测/标签对比。

对应 zzh.pdf 第一部分 OGB，完整实现：固定市场价回放 → MPC bid* 轨迹 → DT action MSE 监督学习 → 指定 RTG 的序列出价 → 独立测试周期评估。它是根据简历和公开资料构建的算法复现，不能代表字节内部 OGB，也不能验证简历中的 advv +32.16%、cost +12.31%。本项目不实现第二部分 Auto-Research/DevAgent/GRPO。

## 直接运行

在 PowerShell 中：

```powershell
cd D:\6908963\ogb_reproduction
python ogb.py --updates 300 --output runs/demo
python -m pytest -q
```

默认使用固定种子的合成流量，CPU 即可运行。GPU 可加 `--device cuda`。输出 model.pt、report.json、cases.csv、attention.npz。report 包含按投放周期隔离的训练/验证/测试划分、验证 action_mse 曲线、固定倍率策略、MPC 教师、三档 RTG 的 DT 效果。只用验证集选 checkpoint，测试集不参与训练、归一化或目标 RTG 选择。

## 公开数据

阿里妈妈仓库 README 提供数据下载链接：
https://github.com/alimama-tech/NeurIPS_Auto_Bidding_General_Track_Baseline

下载并解压完整 period CSV 后运行：

```powershell
python ogb.py --csv ..\NeurIPS_Auto_Bidding_General_Track_Baseline\data\traffic\period-7.csv ..\NeurIPS_Auto_Bidding_General_Track_Baseline\data\traffic\period-8.csv ..\NeurIPS_Auto_Bidding_General_Track_Baseline\data\traffic\period-9.csv --updates 3000 --output runs/public --device cuda
```

至少需要三个完整周期，建议更多。此读取器每文件载入内存，只读八列；完整比赛数据应先按周期/广告主分片处理。不自动下载约 25GB 的比赛全量数据。CSV 必须包含 48 个时段，不支持提前终止后的不完整轨迹；缺失未来机会时无法可靠重建反事实投放，不应以零流量补齐冒充完整回放。

## 架构和口径

1. episode = 投放周期 × 广告主；每个时段选择一个出价倍率 m，请求 bid = m × CPA_target × pValue。m 属于 [0,2]。这与公开基线 alpha × pValue 的参数化等价，其中 alpha=m×CPA_target。
2. 回放使用 leastWinningCost 作为固定阈值/支付价格，按 pvIndex 顺序赢取机会，并硬限制累计预算。转化使用赢取机会的 pValue 之和，避免候选方案随机采样噪声。它与公开 OfflineEnv 的阈值逻辑接近，但采用确定性期望转化，未复制多广告位曝光、预测 sigma、延迟转化或竞争反馈。
3. 教师 MPC 每步枚举两步动作序列（7²），以累计效用加软 pacing 正则择优，执行第一步后重规划。bid* 是有限候选网格、有限预测窗口下的最优第一步，不是全天全局最优解。默认用未来真实市场价/概率构造离线 hindsight 教师，只用于标注；线上 MPC 若要运行必须替换为预测模型。测试中的 MPC 是具有未来信息的参考。
4. U=C−lambda×max(cost/CPA_target−C,0)，reward_t=U_t−U_(t−1)，RTG 为未折扣的未来 reward 和。CPA 是软约束，预算是硬约束；lambda 默认 1。训练状态和 reward 使用教师实际执行轨迹，避免将日志状态与反事实 action/reward 错接。
5. DT token 顺序为 `(RTG_t,state_t,action_t)`，模态独立线性 embedding 加时段 embedding；pre-LN 多头因果 attention；在 state token 输出 action。训练 MSE 只统计有效位置；右 padding 不参与 loss/attention。当前 action 标签位于 state token 后，无法泄漏进预测。
6. 状态使用时间/剩余预算/pacing/上次倍率/历史 CPA/历史量价/当前可见 pValue 和流量。当前及未来市场价不进入状态。FEATURES 固定十维，checkpoint 保存字段顺序、schema、训练集归一化和 RTG 尺度。与根目录 dt_bid_from_dataflow.py 的 67 维业务特征不同，checkpoint 不可混用。
7. 推理先用训练轨迹首步 RTG 的 80% 分位数设定目标，每一步减去上一时段已反馈 reward，再生成动作；保留最近 K=8 步的 RTG/state/action 历史。线上反馈需要与训练 reward 单位一致。三档 RTG 扫描只是控制敏感性检查，不保证 RTG 越大实际转化越大。

核心实现位于 ogb.py：Replay/state/outcome、mpc、trajectory、DT、Policy、run。训练只使用 action MSE，没有 TD/Q-learning/策略梯度更新。

## 接入现有基线

保持项目目录在 PYTHONPATH，设置 checkpoint 后，在基线的 bidding_train_env/strategy/__init__.py 中选择适配器：

```powershell
$env:PYTHONPATH='D:\6908963\ogb_reproduction;D:\6908963\NeurIPS_Auto_Bidding_General_Track_Baseline'
$env:OGB_CHECKPOINT='D:\6908963\ogb_reproduction\runs\public\model.pt'
```

```python
from strategy import OgbBiddingStrategy as PlayerBiddingStrategy
```

随后运行基线 main/main_test.py。适配器遵循 budget/name/cpa/category、reset、bidding 参数协议，并由基线更新 remaining_budget。适配器用历史竞得 pValue 计算期望 reward，并从 checkpoint 读取 penalty，与本复现的训练口径一致。基线评估自身采样实际转化，因此它的最终 CPA 指标与这里的期望 CPA 不完全相同。适配器提供接入点；未下载公开数据前没有完成官方评测。

## 消融与评估

```powershell
python ogb.py --label fixed --updates 300 --output runs/fixed_labels
python ogb.py --context 1 --updates 300 --output runs/context1
python ogb.py --penalty 2 --updates 300 --output runs/penalty2
```

使用相同 seed/数据划分比较：bid* 标签与固定倍率标签、长序列与单步、CPA 惩罚权重。cases.csv 包含每个测试 episode 的转化、成本、CPA、违约、效用、预算利用率和平均倍率。expected conversions 是公开代理指标，不等价于内部 advv。小样本运行只能验证代码链路，效果结论需公开数据、多种子、置信区间。

attention.npz 是每层/每头的时间 token attention，不能直接视为十个 state 特征的重要性，更不能证明特征因果关系。本复现不冒充简历第二部分的特征消融诊断系统。

## 来源

- 论文：https://arxiv.org/abs/2106.01345
- 官方 DT：https://github.com/kzl/decision-transformer ，gym/decision_transformer/models/decision_transformer.py 的 RTG/state/action token 和 state-token action head 是本实现对齐依据。
- 自动出价基线：https://github.com/alimama-tech/NeurIPS_Auto_Bidding_General_Track_Baseline ，复用公开数据格式与策略接口，参考 OfflineEnv 的竞价阈值口径。

DT backbone 为独立 PyTorch 实现，未复制上游代码、预训练权重或许可证文件；模型结构不是逐字复刻官方 GPT2，包括没有官方 dropout 配置、没有 state/return 辅助 head。完成的是 OGB 架构映射与可运行实验，而非原论文 Gym/Atari benchmark 复现。
