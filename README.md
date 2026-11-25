# 足球比分预测模型（泊松混合模型）

基于 LightGBM 双输出泊松回归 + Dixon-Coles 修正的足球比赛预测模型，
输出 **主胜/平/客胜概率、最可能比分、总进球数分布** 及其概率，概率经过
isotonic 校准，评估指标为 Log Loss 与 Brier Score。

**双数据源**：
- 训练数据：football-data.co.uk（五大联赛 2013-2025 历史赛果 + 赔率）
- 实时预测：500 彩票网竞彩足球（中文队名 + 竞彩 SP 赔率 + 让球 + 球队近 10 场战绩）

---

## 1. 项目结构

```
football-prediction/
├── data/raw/               # 原始 CSV（football-data.co.uk，五大联赛 2013-2025）
├── models/                 # 训练产物（运行 train.py 后生成）
│   ├── model.pkl           # 模型（LightGBM 双输出 + 校准器 + ρ）
│   ├── feature_state.pkl   # 特征增量状态（预测时快速构建特征）
│   ├── train_meta.json     # 训练元数据（超参、评估指标、切分范围）
│   ├── training_curve.png  # Log Loss 训练曲线
│   └── training_history.txt
├── src/
│   ├── data_loader.py      # CSV 加载/标准化 + 模拟数据生成（兜底）
│   ├── features.py         # 特征工程（实力/状态/交锋/赔率/赛程 + 缺失处理）
│   ├── poisson_utils.py    # 泊松网格、Dixon-Coles τ 修正、ρ 优化、衍生概率
│   ├── model.py            # PoissonLightGBM（自定义 Poisson NLL + isotonic 校准）
│   └── evaluation.py       # Log Loss / Brier / 比分对数概率 / MAE / 区间覆盖率
├── train.py                # 训练主脚本（Optuna + 时序分割 + 评估）
├── predict.py              # 预测接口（CLI + 库函数 predict_match）
├── predict_jczq.py         # 500 竞彩实时数据预测入口
├── train_jczq.py           # 500 竞彩专属模型训练（让球玩法）
├── src/jczq_fetcher.py     # 500 比赛列表/球队战绩抓取
├── src/jczq_history.py     # 500 历史开奖抓取（训练数据）
├── models_jczq/            # 500 专属模型产物
└── requirements.txt
```

## 2. 环境与安装

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

要求 Python 3.10+。已在 Python 3.13 验证通过。

## 3. 数据

### 3.1 真实数据（默认）
数据来自 [football-data.co.uk](https://www.football-data.co.uk)，
含英超(E0)、西甲(SP1)、德甲(D1)、意甲(I1)、法甲(F1) 共 **2013/14 ~ 2024/25
12 个赛季、约 2 万场比赛**，字段包括：日期、主客队、全场/半场进球、射门/射正、
Bet365 初始赔率（胜平负 + 大小球 + 让球盘口）。

数据下载脚本（也可手动放置到 `data/raw/`）：

```bash
mkdir -p data/raw && cd data/raw
for s in 1314 1415 1516 1617 1718 1819 1920 2021 2122 2223 2324 2425; do
  for l in E0 SP1 D1 I1 F1; do
    curl -s -o ${s}_${l}.csv "https://www.football-data.co.uk/mmz4281/${s}/${l}.csv"
  done
done
```

### 3.2 模拟数据（兜底）
若无法联网获取真实数据，`data_loader.generate_synthetic_data()` 会生成符合
真实统计特征的模拟数据（主队优势 ≈ +0.35 球/场、场均总进球 ≈ 2.7、胜/平/负
频率 ≈ 45%/25%/30%），并附带由强度换算的市场赔率。训练时加
`--use-synthetic` 即可使用。**本仓库默认使用真实数据。**

## 4. 快速开始

### 训练

```bash
python train.py                        # 完整流程（Optuna 15 trials，约几分钟）
python train.py --trials 30            # 更多超参搜索
python train.py --skip-optuna          # 跳过搜索，用默认参数（最快）
```

训练流程：
1. 加载数据 → 特征工程（基础实力/近期状态/交锋/赔率/赛程，缺失值 -1 填充
   + 缺失指示特征，全部特征只用历史信息、**无未来泄漏**）
2. 严格按日期时序分割：训练(≤2022-12-31) / 验证(2023) / 测试(≥2024-01-01)
3. Optuna 搜索 `num_leaves / learning_rate / feature_fraction /
   min_child_samples`，目标为验证集**泊松 NLL**
4. 最终模型：训练集训练 + 验证集早停；训练集上对 λ 做 **isotonic 校准**；
   **Dixon-Coles ρ** 在验证集上按对数似然优化
5. 测试集评估：Log Loss / Brier / 命中率 / 比分对数概率 / 进球 MAE /
   80% 预测区间覆盖率，并与**赔率隐含概率基线**对比
6. 输出训练 Log Loss 曲线（`models/training_curve.png`）与模型文件

### 预测

```bash
# CLI
python predict.py "Man City" "Liverpool" 2025-08-23
python predict.py --league E0 --odds 2.60,3.50,2.70 "Arsenal" "Chelsea" 2025-08-24

# 库调用
from predict import predict_match
r = predict_match("Man City", "Liverpool", "2025-08-23",
                  odds=(2.10, 3.40, 3.60), league="E0")
```

> 注：`--odds` 用逗号分隔字符串（避免 argparse `nargs="+"` 吞掉位置参数）；
> 形式为 `主,平,客` 或 `主,平,客,大2.5,让球盘`。

### 预测：500 竞彩实时数据（推荐日常使用）

```bash
python predict_jczq.py              # 列出今天全部竞彩比赛（中文队名/SP赔率/让球）
python predict_jczq.py 周五008      # 预测指定编号比赛（自动抓取两队近10场战绩）
```

### 500 竞彩专属模型（train_jczq.py）

用 500 网历史开奖训练独立模型，覆盖竞彩全部赛事（日职/荷甲/德乙/韩职/挪超/
五大联赛/欧战等）。**让球胜平负玩法上显著超越竞彩官方让球SP**：

```bash
python src/jczq_history.py --days 365   # ① 抓历史开奖（~5000场）→ data/jczq_history.csv
python train_jczq.py                    # ② 训练 500 专属模型 → models_jczq/
python predict_jczq.py 周五001          # ③ 预测（自动选用 500 模型）
```

| 集合 | 让球 Log Loss | 让球SP基线 | 让球命中率 | 基线 |
|---|---|---|---|---|
| 测试集 (最近30天, 204场) | **0.920** | 1.043 | **58.3%** | 51.0% |
| 验证集 (513场) | **0.898** | 1.078 | **59.5%** | 46.2% |

> 模型在『让球胜平负』上大幅优于官方让球SP（官方固定赔率定价保守、更新慢，
> 而模型融合了 500 实时战绩信号）。预测输出包含：
> - **让球胜平负概率**（模型最强维度，平移比分网格计算）
> - **比分分布 / 总进球**（真实 λ，与玩法无关）
> - **不让球市场参考**（比赛列表页不让球 SP 去抽水）
> - 置信度：`highest`(两队均在主模型训练集) / `high`(500 战绩+让球SP) / `low`

示例（周五001 日职 横滨水手 vs 鹿岛鹿角，鹿岛让1球）：

```
→ 最可能比分: 1:0  预期进球: 主 1.62 / 客 0.80
→ 大 2.5 球: 52.0%   小 2.5 球: 48.0%
→ 让球+1后 胜平负: 主 78.8% / 平 12.8% / 客 8.5% (模型专用维度)
→ 不让球市场参考: 主 28.1% / 平 28.6% / 客 43.3%
→ 模型: jczq-500专用(让球玩法)  置信度: high
```

### 计算结论（竞彩投注参考卡）

每场预测自动生成「计算结论」，包含胜平负 / 让球胜平负 / 双P / 比分参考：

```
【周六003 安养FC vs 大田市民】
胜平负推荐  : S 主胜（70.4%）  [备选 P 平 19.3%]
让球胜平负  : RS 让胜（89.7%）  让球 +1  [备选 RP 让平 7.3%]
双P参考    : ─ 不推  半全场平平≈7.5%（半场平 39.1% × 全场平 19.3%），让平 7.3%
比分参考    : 2:0（13.2%）、1:0（12.8%）
```

- **S/P/F**：胜平负（不让球）主推 + 备选
- **RS/RP/RF**：让球胜平负主推 + 备选（含竞彩让球数）
- **双P**：半全场『平平』+ 让球『平』组合参考；让平概率 ≥20% 且
  半全场平平概率可观时推荐（半场概率由半场 λ≈45%×全场 λ 近似估算）
- **比分参考**：Top 1-2 个比分（概率接近时自动追加，如 1:1/0:1/0:2）

批量模式同样输出每场计算结论：`python predict_jczq.py 周六003-010`，
结果含推荐字段保存于 `outputs/batch_predict.csv`。

> 数据来源说明：抓取自 [500彩票网](https://trade.500.com/jczq/)（竞彩足球数据
> 大厅、[历史开奖](https://zx.500.com/jczq/kaijiang.php)、球队/联赛页），
> GB2312 编码，缓存于 `data/jczq_cache/` 与 `data/jczq_history_raw/`；
> 数据仅用于个人研究与娱乐预测，请遵守网站条款。

输出示例：

```json
{
  "home_win_prob": 0.42,
  "draw_prob": 0.26,
  "away_win_prob": 0.32,
  "score_probs": { "1:0": 0.12, "2:1": 0.08, "0:0": 0.07, "1:1": 0.09, "..." : "..." },
  "total_goals_distribution": { "0": 0.05, "1": 0.12, "2": 0.22, "3": 0.21, "..." },
  "most_likely_score": "1:0",
  "lambda_home": 1.72,
  "lambda_away": 1.08,
  "dixon_coles_rho": -0.031
}
```

说明：
- `odds` 可选参数 `(主胜, 平, 客胜)` 或 `(主胜, 平, 客胜, 大2.5赔率, 让球盘)`；
  不提供时模型仅靠历史战绩推断（对应特征按缺失处理）。
- 预测日期若早于模型训练截止日（约 2025-05），会自动回退到全量重建特征
  （较慢但准确）。

## 5. 技术细节

### 5.1 特征向量（每场 60+ 维）
| 类别 | 特征 |
|---|---|
| 基础实力 | 主/客队近 365/180/90 天场均进球、失球、射正（xG 代理，指数加权） |
| 近期状态 | 最近 3/5/8 场场均积分、进球、失球、赢盘率（近场权重更高） |
| 交锋记录 | 历史交锋场均进球、主队胜率、平局率（最近 10 次） |
| 赔率特征 | 初始主/平/客赔率、去抽水隐含概率（Basic 法）、赔率对数比、大小球、让球盘 |
| 赛程体能 | 主/客队休息天数、主队一周双赛标记、客队连续客场数 |
| 缺失处理 | 数值 -1 填充 + `*_missing` 二值指示列 |

### 5.2 模型：Poisson NLL 双输出回归（方案 A）
- 主/客队各一棵 LightGBM，共享特征矩阵；
- 自定义目标函数：`Loss = mean( λ_h − y_h·log λ_h + λ_a − y_a·log λ_a )`，
  输出经 `softplus` 保证 λ > 0；梯度与 Fisher hessian 手推解析式；
- 训练后在训练集上对 λ 做 **isotonic regression 校准**（`out_of_bounds='clip'`）；
- 预测时用 Dixon-Coles 网格：`P(i,j) ∝ Pois(i|λ_h)·Pois(j|λ_a)·τ(i,j)`，
  τ 修正低比分区 (0:0, 1:0, 0:1, 1:1)，ρ 通过对数似然网格搜索优化。

### 5.3 评估与基线
- **Log Loss / Brier Score**：赛果三分类；
- **比分预测**：真实比分处的对数概率均值（越高越好）；
- **进球预测**：λ 的 MAE + 泊松 80% 预测区间覆盖率；
- **基线**：Bet365 赔率去抽水隐含概率的 Log Loss / Brier，
  模型必须显著优于基线才算有效。

### 5.4 方案 B（进阶，可选）
`src/model.py` 为 LightGBM 方案。Deep Poisson Network（球队嵌入 +
Transformer 状态编码 + MLP 双输出）可参考 `poisson_nll_objective` 的损失
定义自行扩展，本项目默认交付方案 A。

### 5.5 温度标定（Temperature Scaling）
Dixon-Coles 网格概率可能过度自信/不自信——直接在验证集上搜索最优
**温度 T**：P_T(i,j) ∝ P(i,j)^(1/T)，使赛果三分类 Log Loss 最小。
本项目 T ≈ 1.0~1.05（轻微平滑），嵌入 `PoissonLightGBM`，预测与评估自动应用。

### 5.6 球队固定效应：动态 ELO
每队维护一个 ELO 评分（初始 1500，K=30，主场加成 +60），特征包含
`home_elo / away_elo / elo_diff`。ELO 是学术界（Dixon-Coles 等）验证有效的
独立于赔率的球队实力信号，能捕捉市场赔率定价滞后的球队状态变化。

## 6. 已知限制与占位
- **实时特征**（即时伤病、xG、首发阵容等）不在数据源内：相关特征按任务书
  约定使用占位符/缺失值（-1 + 指示列），模型自动忽略；
- 射正缺失的早期赛季用 `射门数 × 0.35` 估算（xG 代理）；
- 比分网格截断到 0~10，尾部概率按比例摊回，λ<4 时截断误差 <2%；
- 预测日期早于训练截止时回退全量重建（~30s），之后预测为增量构建（<0.1s）。

## 7. 性能与复现
- 首次训练：特征构建 ~5 分钟（纯 Python 滚动窗口，20k 场）；
  后续重训走缓存（`models/processed_feat.pkl`）秒级加载
- Optuna 12 trials × LightGBM 训练 ≈ 1 分钟
- 最终模型训练 + 校准 + ρ + 温度 ≈ 1 分钟
- 预测单场 ≈ 0.2 秒

训练完成后 `models/train_meta.json` 会记录本次所有指标。参考结果（五大联赛
2024-01 ~ 2025-05 测试集 2652 场）：

```
测试集 (2024+) 评估：
  赛果 Log Loss : 0.9665  (赔率基线 0.9640)  - 模型与赔率基线持平
  赛果 Brier    : 0.5748  (赔率基线 0.5731)
  赛果命中率    : 0.5441  (赔率基线 0.5381)  - 模型命中率 +0.6pp
  比分对数概率  : -7.99  (赔率基线无法直接给出)
  进球 MAE      : 主 0.94 / 客 0.88
  80% 区间覆盖率: 主 0.88 (宽 2.4) / 客 0.85 (宽 2.0)

模型参数：rho = -0.027, temperature = 1.025, Optuna 最佳超参
  num_leaves=15, learning_rate=0.029, feature_fraction=0.994, min_child_samples=10
```

> 关于"模型未显著超越赔率 Log Loss"：开盘赔率已吸收绝大多数可获得信息
> （市场有效），任何特征组合都难以在赛果概率上明显击败市场定价。模型的
> 真正价值在于赔率无法直接给出的**联合比分分布**（比分对数概率 / 总进球
> 分布 / 区间覆盖率 / 比分概率密度），以及在命中率上的微弱优势。
