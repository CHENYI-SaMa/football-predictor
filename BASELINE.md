# 基线模型（Baseline）—— 2026-08-15 固定

## 模型
- 文件：models_baseline/（model.pkl + train_meta.json + feature_state.pkl + feature_columns.txt）
- 来源：models_v3（与当前正式模型一致）

## 特征体系
- 191 个特征（Glicko 贝叶斯更新 ELO + 去 form + xG 族 + 主客场分窗 + 趋势/排名/轮次/角球/犯规/大小球）
- 训练数据：jczq_history_xg.csv（含德甲补全后 xG，覆盖提升）
- 数据范围：2022-08-08 ~ 2026-08-10

## 性能（测试集 211 场）
- 让球胜平负 Acc：60.2%（市场基线 46.4%，+13.8pp）
- 让球 LogLoss：0.945（市场 1.057）
- 验证集让球 Acc：58.7%

## 关键设计
- ELO 更新：Glicko-1（评分 + 不确定性 RD，信息量自适应）
- form 特征（抢分率/净胜球）：A/B 验证负贡献，已排除（模块保留）
- 评估增强：LR 基线 / AUC / 校准曲线 / RF / XGBoost / CatBoost / Stacking 全对比存档

## 后续任何实验，均以此基线为对照
