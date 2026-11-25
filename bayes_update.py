# -*- coding: utf-8 -*-
"""bayes_update.py —— 贝叶斯更新 · 整体（可打包为独立 exe）
============================
一个整体完成 V4 模型自更新：
  生成更新包(从存档回填赛果) → 吃包(并入训练集) → 重训 → 新旧验证替换
模式：
  --auto   仅当 待处理包≥7 或 周一 才重训（GUI 启动用）
  --force  跳过验证强制重训
  --pack-only 只吃包不重训
所有路径相对本项目根(bayes_common)，可项目根跑或打包运行。
"""
from __future__ import annotations
import argparse, io, json, os, pickle, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd

# ---------- 路径（相对本项目根 / 打包 _MEIPASS） ----------
def _find_project():
    """向上查找含 data_v4 的目录作为项目根。
    源码运行=脚本目录；打包exe可位于 distBayes/bayes_update(上级是项目根) 或安装根。"""
    here = Path(__file__).resolve().parent
    for cand in (here, here.parent, here.parent.parent, here.parent.parent.parent):
        if (cand / "data_v4").exists():
            return cand
    return here


HERE = Path(__file__).resolve().parent
PROJECT = _find_project()
DATA4 = PROJECT / "data_v4"
POOL = DATA4 / "update_pool"
BASE = DATA4 / "jczq_all_normalized.csv"
MODELS = DATA4 / "models_v4"
BAK = DATA4 / "models_v4_bak"
OUT_DIR = Path.home() / "Desktop" / "竞彩预测" / "存档"
DONE_MARK = "_done"
MIN_PACKS = 7

sys.path.insert(0, str(PROJECT))
from src.features_v3 import FeatureBuilderV3      # noqa: E402
from src.model import PoissonLightGBM            # noqa: E402
from src.jczq_history import date_range, fetch_day  # noqa: E402

DATA4.mkdir(parents=True, exist_ok=True)
POOL.mkdir(parents=True, exist_ok=True)

# ---------- 常量 ----------
R_LABELS = ["让胜", "让平", "让负"]
_RANG = {"主让3球": -3, "主让2球": -2, "主让1球": -1, "平手盘": 0,
         "客让1球": 1, "客让2球": 2, "客让3球": 3}
EXTRA = ["htg", "htag", "hs", "as", "hst", "ast", "ou_over", "ou_under",
         "xg_h", "xg_a", "hc_h", "ac_h", "hf_h", "af_h", "hy_h", "ay_h",
         "hc_a", "ac_a", "hf_a", "af_a", "hy_a", "ay_a"]
_COLS = ["date", "league", "home_team", "away_team", "fthg", "ftag", "ftr",
         "htg", "htag", "hs", "as", "hst", "ast", "b365h", "b365d", "b365a",
         "ou_over", "ou_under", "ahh", "half_available"]
EVAL_DAYS_DEF = 10
ACC_TOL = 0.03


# ---------- 控制台/管道输出统一 UTF-8（GBK 下无法编码 ✓ 等字符会崩 UnicodeEncodeError） ----------
class _SafeStream:
    """把写操作包一层：目标流为 None / 无效 / 写失败时静默吞掉，
    避免 tqdm 进度条或 print 向无效 stdout 写入抛 OSError(Errno 22) 而崩溃。"""
    def __init__(self, dst):
        self._dst = dst

    def write(self, s):
        if self._dst is not None:
            try:
                return self._dst.write(s)
            except Exception:
                return None

    def flush(self):
        if self._dst is not None:
            try:
                self._dst.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def fileno(self):
        try:
            return self._dst.fileno()
        except Exception:
            return -1

    def reconfigure(self, *a, **k):
        try:
            self._dst.reconfigure(*a, **k)
        except Exception:
            pass

    @property
    def encoding(self):
        try:
            return self._dst.encoding or "utf-8"
        except Exception:
            return "utf-8"


if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
sys.stdout = _SafeStream(sys.stdout)
sys.stderr = _SafeStream(sys.stderr)
os.environ["TQDM_DISABLE"] = "1"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def log(m):
    try:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    except Exception:
        pass


# ================= 1) 生成更新包 =================
def parse_score(s):
    if not s or s in ("未开赛", "-", "None"):
        return None
    try:
        h, a = str(s).split(":"); return int(h), int(a)
    except Exception:
        return None


def make_pack(date_iso: str) -> list[dict]:
    jf = OUT_DIR / f"竞彩预测数据_{date_iso.replace('-', '')}.json"
    if not jf.exists():
        return []
    data = json.load(open(jf, encoding="utf-8"))
    rows = []
    for r in data.get("results", []):
        sc = parse_score(r.get("实际比分", ""))
        if sc is None:
            continue
        rows.append({"date": r.get("日期", date_iso), "league": r.get("赛事", ""),
                     "home_team": r.get("主队", ""), "away_team": r.get("客队", ""),
                     "fthg": sc[0], "ftag": sc[1],
                     "ahh": _RANG.get(str(r.get("让球", "")), 0),
                     "rang_text": r.get("让球", ""), "no": r.get("编号", ""),
                     "命中": r.get("是否命中")})
    return rows


def backfill_archive(date_iso: str) -> int:
    """读取桌面存档预测 JSON，对未回填实际比分的场次联网抓赛果并写回。
    返回本次新增回填场次数。不依赖 GUI 回填按钮（已删除），保证次日
    更新包总能拿到真实赛果。数据源双路兜底：fetch_day + fetch_day_trade。"""
    d8 = date_iso.replace("-", "")
    jf = OUT_DIR / f"竞彩预测数据_{d8}.json"
    if not jf.exists():
        return 0
    data = json.loads(jf.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    todo = [r for r in rows if not str(r.get("实际比分") or "").strip()]
    if not todo:
        return 0
    day = {}
    for fn in (fetch_day, fetch_day_trade):
        try:
            for m in (fn(date_iso, use_cache=False) if fn is fetch_day else fn(date_iso)):
                day.setdefault(m["no"], m)
        except Exception:
            continue
    n = 0
    for r in todo:
        m = day.get(r.get("编号", ""))
        if m is None or m.get("fthg") is None or m.get("ftag") is None:
            continue
        r["实际比分"] = f"{m['fthg']}:{m['ftag']}"
        r["实际结果"] = ("主胜" if m["fthg"] > m["ftag"] else
                         ("平" if m["fthg"] == m["ftag"] else "主负"))
        rang = int(m.get("rang", 0) or 0)
        act = 0 if m["fthg"] + rang > m["ftag"] else (
            1 if m["fthg"] + rang == m["ftag"] else 2)
        r["实际让球结果"] = ["RS 让胜", "RP 让平", "RF 让负"][act]
        n += 1
    if n:
        data.setdefault("回填核对", {})
        jf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[backfill {date_iso}] 回填 {n} 场真实赛果")
    return n


def step_pack(date: str | None):
    ds = date or time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
    rows = make_pack(ds)
    if not rows:
        print(f"[pack {ds}] 无已回填赛果，未生成更新包")
        return
    out = POOL / f"update_{ds.replace('-', '')}.json"
    out.write_text(json.dumps({"date": ds, "n": len(rows), "records": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[pack {ds}] 生成 {out.name}：{len(rows)} 场")


# ================= 2) 吃包 =================
def _mark_done(p: Path):
    """把更新包改名为 _done；目标已存在则先删除，避免 Windows 报错。"""
    target = Path(str(p).replace(".json", DONE_MARK + ".json"))
    if target.exists():
        try:
            target.unlink()
        except OSError:
            return
    try:
        p.rename(target)
    except Exception:
        pass


def step_upsert():
    packs = sorted(POOL.glob("update_*.json"))
    pending = [p for p in packs if not p.name.endswith(DONE_MARK + ".json")]
    if not pending:
        print("[upsert] 无待处理更新包")
        return 0
    raw = pd.read_csv(BASE, parse_dates=["date"])
    for c in _COLS:
        if c not in raw.columns:
            raw[c] = None
    n_before = len(raw); total = 0
    for p in pending:
        pk = json.loads(p.read_text(encoding="utf-8"))
        df = pd.DataFrame(pk.get("records", []))
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in _COLS:
            if c not in df.columns:
                df[c] = None
        df = df[_COLS].dropna(subset=["date", "home_team", "away_team"])
        keys = raw.set_index(["date", "home_team", "away_team"]).index
        df = df[(~df.set_index(["date", "home_team", "away_team"]).index.isin(keys))]
        if len(df) == 0:
            _mark_done(p)
            print(f"[upsert {p.name}] 无新增，标记已处理")
            continue
        raw = pd.concat([raw, df], ignore_index=True)
        total += len(df)
        _mark_done(p)
        print(f"[upsert {p.name}] 新增 {len(df)} 场")
    if total == 0:
        print("[upsert] 无新增写入")
        return 0
    raw = raw.drop_duplicates(subset=["date", "home_team", "away_team"],
                              keep="last").sort_values("date").reset_index(drop=True)
    raw.to_csv(BASE, index=False, encoding="utf-8-sig")
    print(f"[upsert] 训练集 {n_before} → {len(raw)} 场，新增 {total}")
    return total


# ================= 3) 重训 + 验证替换 =================
def build_features():
    raw = pd.read_csv(BASE, parse_dates=["date"])
    for c in EXTRA:
        if c not in raw.columns:
            raw[c] = np.nan
    raw = raw.sort_values(["date", "home_team", "away_team"],
                          kind="stable").reset_index(drop=True)
    raw["rang"] = pd.to_numeric(raw["ahh"], errors="coerce").fillna(0)
    return raw


def train_v4(feat_cols=None):
    feat, _ = FeatureBuilderV3(build_features()).build()
    feat = feat.reset_index(drop=True)
    feat = feat.sort_values(["date", "home_team", "away_team"],
                            kind="stable").reset_index(drop=True)
    ex = {"fthg", "ftag", "date", "league", "home_team", "away_team", "rang"}
    cols = ([c for c in feat.columns if c not in ex and "form" not in c]
            if feat_cols is None else [c for c in feat_cols if c in feat.columns])
    maxd = pd.to_datetime(feat["date"]).max()
    cut = maxd - pd.Timedelta(days=90)
    tr = feat[pd.to_datetime(feat["date"]) <= cut]
    va = feat[pd.to_datetime(feat["date"]) > cut]
    model = PoissonLightGBM(base_params={})
    model.fit(tr[cols].values.astype(float), tr["fthg"].values.astype(float),
              tr["ftag"].values.astype(float),
              va[cols].values.astype(float), va["fthg"].values.astype(float),
              va["ftag"].values.astype(float),
              num_boost_round=2000, early_stopping=100)
    return model, cols, tr, va, feat


def range_probs(model, X, rang):
    grids = model.predict_grids(X); n = grids.shape[1]
    ih, ja = np.indices((n, n)); sh = ih[None, :, :] + rang
    pr = np.clip(np.stack([np.sum(grids * (sh > ja), axis=(1, 2)),
                           np.sum(grids * (sh == ja), axis=(1, 2)),
                           np.sum(grids * (sh < ja), axis=(1, 2))], 1), 1e-12, 1.0)
    return pr / pr.sum(1, keepdims=True)


def jczq_row(h, a, rang, date, league, odds):
    rh, rd, ra = (odds if odds else (None, None, None))
    return pd.Series({"date": pd.Timestamp(date), "league": league,
                      "home_team": h, "away_team": a,
                      "fthg": -1, "ftag": -1, "ftr": "X",
                      "htg": np.nan, "htag": np.nan, "hs": np.nan, "as": np.nan,
                      "hst": np.nan, "ast": np.nan,
                      "b365h": rh, "b365d": rd, "b365a": ra,
                      "ou_over": np.nan, "ou_under": np.nan, "ahh": float(rang)})


def build_state():
    _fs, state = FeatureBuilderV3(build_features()).build()
    return state


def evaluate(model, cols, state, days=EVAL_DAYS_DEF):
    to = pd.Timestamp.now().strftime("%Y-%m-%d")
    from_ = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    dates = date_range(from_, to)
    n = hit = 0
    for d in dates:
        try:
            day = fetch_day(d, use_cache=True)
        except Exception:
            continue
        for m in day:
            if m.get("fthg") is None or m.get("ftag") is None:
                continue
            h, a = m.get("home_team"), m.get("away_team")
            if not h or not a:
                continue
            rang = int(m.get("rang", 0) or 0)
            row = jczq_row(h, a, rang, d, m.get("league", ""),
                           (m.get("rsp_h"), m.get("rsp_d"), m.get("rsp_a")))
            fd = FeatureBuilderV3.build_single(state, pd.Series(row))
            X = np.array([[fd.get(c, -1.0) for c in cols]], dtype=float)
            pr = range_probs(model, X, np.array([float(rang)]))[0]
            act = 0 if m["fthg"] + rang > m["ftag"] else (1 if m["fthg"] + rang == m["ftag"] else 2)
            n += 1; hit += (int(pr.argmax()) == act)
    return {"n": n, "acc": hit / n if n else 0}


def step_workly(force: bool, eval_days: int):
    log("=== V4 重训+验证替换（贝叶斯式回喂更新） ===")
    if BAK.exists():
        shutil.rmtree(BAK, ignore_errors=True)
    shutil.copytree(MODELS, BAK)
    try:
        old_cols = json.loads((BAK / "train_meta.json").read_text(encoding="utf-8"))["feature_cols"]
    except Exception:
        old_cols = None
    log(f"[1/4] 已备份（旧特征 {len(old_cols) if old_cols else '?'} 列）")
    model, cols, tr, va, feat = train_v4(old_cols)
    log(f"[2/4] 重训完成：训练{len(tr)} 验证{len(va)}，特征 {len(cols)} 列")
    if not force:
        state = build_state()
        log(f"[3/4] 评估 {eval_days} 天...")
        new = evaluate(model, cols, state, eval_days)
        old_m = PoissonLightGBM.load(BAK / "model.pkl")
        old = evaluate(old_m, old_cols or cols, state, eval_days)
        log(f"[3/4] 新 {new['acc']:.1%}({new['n']}) | 旧 {old['acc']:.1%}({old['n']})")
        if new["n"] == 0:
            log("[3/4] 无评估数据，跳过验证替换")
        elif new["acc"] < old["acc"] - ACC_TOL:
            log(f"[3/4] ✗ 新模型劣化，还原旧模型")
            shutil.rmtree(MODELS, ignore_errors=True)
            shutil.copytree(BAK, MODELS)
            return 1
    log("[3/4] ✓ 新模型可用，替换为生产")
    model.save(MODELS / "model.pkl")   # 关键：重训模型必须落盘，否则下次预测仍用旧模型
    _fs, state = FeatureBuilderV3(build_features()).build()
    with open(MODELS / "feature_state.pkl", "wb") as f:
        pickle.dump(state, f)
    (MODELS / "feature_columns.txt").write_text("\n".join(cols), encoding="utf-8")
    json.dump({"n_train": len(tr), "n_val": len(va), "feature_cols": cols,
               "trained_at": time.strftime("%Y-%m-%d %H:%M"),
               "data_range": [str(feat['date'].min().date()), str(feat['date'].max().date())]},
              open(MODELS / "train_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    shutil.rmtree(BAK, ignore_errors=True)
    log("[4/4] 完成：models_v4 已更新")
    return 0


def should_retrain():
    """每月第一个周一重训模型；平时仅吃包+刷新特征状态（--auto 主路径）。
    min: 只在每月首个周一触发，避免每周小样本重训过拟合近几周噪声。"""
    tm = time.localtime()
    is_first_monday = (tm.tm_wday == 0 and tm.tm_mday <= 7)
    decide = is_first_monday
    print(f"[auto] 每月第一周一{'是' if is_first_monday else '否'} → "
          f"{'重训模型' if decide else '仅吃包+刷新状态'}")
    return decide


def refresh_state():
    """吃包后仅重建特征状态（ELO/近期状态/交锋等），不重训模型权重。
    让滚动特征吃到最新比赛，保持预测新鲜度。"""
    _fs, state = FeatureBuilderV3(build_features()).build()
    with open(MODELS / "feature_state.pkl", "wb") as f:
        pickle.dump(state, f)
    log("[refresh] 特征状态已刷新（模型权重未变）")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pack-only", action="store_true")
    ap.add_argument("--eval-days", type=int, default=EVAL_DAYS_DEF)
    args = ap.parse_args()

    ds = args.date or time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
    backfill_archive(ds)   # 先补回填昨日赛果，保证更新包可生成
    step_pack(ds)
    added = step_upsert()

    if args.pack_only:
        print("\n[pack-only] 已吃包，未重训"); return 0

    do = True
    if args.auto:
        do = should_retrain()
    if do:
        step_workly(args.force, args.eval_days)
        print("\n✅ 贝叶斯更新完成")
    else:
        print("\n[auto] 非重训日，本次仅吃包")
        if added:
            try:
                refresh_state()
            except Exception as e:
                print(f"[auto] 状态刷新失败：{e}")


if __name__ == "__main__":
    main()