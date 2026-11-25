# 对照实验：验证"训练集大小"是否是瓶颈
import pandas as pd, numpy as np
df = pd.read_csv("data/features.csv", parse_dates=["date"])
print("== 时间分布 ==")
print(f"2024-08-01 前: {(df['date'] < '2024-08-01').sum()}")
print(f"2024-08-01~2025-08-07: {((df['date'] >= '2024-08-01') & (df['date'] < '2025-08-07')).sum()}")
print(f"2025-08-07 后: {(df['date'] >= '2025-08-07').sum()}")
print(f"总场次: {len(df)}")
b5 = df[df["league"].isin({"英超","西甲","德甲","意甲","法甲"})]
print(f"五大联赛: 2024-08前 {(b5['date'] < '2024-08-01').sum()} | 24-25年 {((b5['date'] >= '2024-08-01') & (b5['date'] < '2025-08-07')).sum()} | 25-26年 {(b5['date'] >= '2025-08-07').sum()}")
