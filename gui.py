# -*- coding: utf-8 -*-
"""竞彩预测助手 · 桌面版（tkinter Frame 布局 / macOS 风格）
==============================================================
顶栏：交通灯 + 标题
侧边栏：今日赛程 / 历史赛程（可收起），点击比赛右侧仅显示该场（点标题回全部）
内容区：标题 + 比赛卡片（点击展开/收起完整版）+ 底部按钮
  - 今日视图：导出（分析报告/精简版/推荐表 3 个 txt → 桌面，JSON 存档 outputs/）
  - 历史视图：查看详情（打开当日赛果核对 Excel）/ 返回今日
启动自动联网获取最新赛程。
"""
import json
import os
import re
import shutil
import zipfile
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from path_helper import get_outputs_dir

import tkinter as tk
from tkinter import messagebox

# ---- Windows 高清屏：进程 DPI 感知，避免字体发虚（必须在创建窗口前设置） ----
# 未声明 DPI 感知时，Windows 会对 tkinter 窗口整体做位图缩放 -> 字体模糊；
# 声明后由系统按真实 DPI 渲染，高分辨率屏文字清晰。
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)      # 1=System DPI Aware
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()        # 旧系统兜底
    except Exception:
        pass

# ---- 资源路径 ----
if getattr(sys, "frozen", False):
    BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
else:
    BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.environ["FOOTBALL_BASE_DIR"] = str(BASE_DIR)


# ---------- 窗口进程输出加固（与 bayes_update / predict_v4 一致） ----------
class _SafeStream:
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
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------- 隐藏敏感目录（data_v4 / update_pool / 存档），保留访问但平时不可见 ----------
def _hide_folder(path):
    try:
        import ctypes
        ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x2)  # FILE_ATTRIBUTE_HIDDEN
    except Exception:
        pass


def _setup_hidden():
    root = BASE_DIR if not getattr(sys, "frozen", False) else \
        Path(sys.executable).resolve().parent
    for rel in ("data_v4", "data_v4/update_pool"):
        try:
            _hide_folder(root / rel)
        except Exception:
            pass
    try:
        _hide_folder(get_outputs_dir())
    except Exception:
        pass

# V3 让球胜平负模型依赖 lightgbm/sklearn（仅 managed venv 具备）；
# GUI 本体用系统 Python(tkinter) 启动，故 subprocess 显式调用 venv 解释器运行 predict_v3.py
V3_PY = "C:/Users/CHENYI/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

# ---- macOS 浅色配色 ----
BG = "#F5F5F7"
SIDEBAR = "#F6F6F8"
CARD = "#FFFFFF"
BORDER = "#E5E5E7"
TEXT1 = "#1D1D1F"
TEXT2 = "#86868B"
TEXT3 = "#AEAEB2"
BLUE = "#007AFF"
RED = "#FF3B30"
GREEN = "#34C759"
YELLOW = "#FFCC00"
FONT = ("Microsoft YaHei UI", 11)
FONT_B = ("Microsoft YaHei UI", 11, "bold")
FONT_S = ("Microsoft YaHei UI", 9)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("竞彩预测助手")
        self.minsize(960, 620)
        self.configure(bg=BG)
        # 软件图标（替代默认角标）：
        # ① iconbitmap 设窗口/任务栏图标（.ico，Windows）
        # ② iconphoto 兜底（.png 加载，某些环境任务栏更可靠）
        try:
            ico = BASE_DIR / "app_icon.ico"
            if ico.exists():
                self.iconbitmap(str(ico))
        except Exception:
            pass
        try:
            png = BASE_DIR / "app_icon.png"
            if png.exists():
                img = tk.PhotoImage(file=str(png))
                self.iconphoto(True, img)
                self._icon_img = img      # 防 GC 回收
        except Exception:
            pass
        # 配置（记住窗口大小 + 侧边栏宽度）：
        # 打包版写 %APPDATA%/竞彩预测（用户级可写，避免 _internal 只读导致记忆失效）；
        # 源码版写项目 data/。
        if getattr(sys, "frozen", False):
            self._cfg_path = (Path(os.environ.get("APPDATA", str(Path.home())))
                              / "竞彩预测" / "gui_config.json")
        else:
            self._cfg_path = BASE_DIR / "data" / "gui_config.json"
        self._cfg = self._load_config()
        self.geometry(self._cfg.get("geometry") or "1120x720")   # 恢复上次窗口大小
        self._sb_width = int(self._cfg.get("sidebar_width", 240))
        self._cache = self._load_matches()
        self._v3 = {}                # V3 让球胜平负（按 no），打开软件时并行预测；旧模型 V2 已移除
        self._filter_no = None        # 侧边栏单场过滤（None=全部）
        self._expanded_no = None      # 当前展开的卡片编号
        self._view = "today"          # today | history
        self._view_date = None        # history 时的 8 位日期

        self._titlebar()
        self._body()
        self._status("就绪")
        self.protocol("WM_DELETE_WINDOW", self._on_close)   # 关闭前保存窗口大小
        self._auto_predict_all()      # 打开软件即自动预测今日赛事

    def _on_close(self):
        """关闭窗口：保存当前窗口大小后退出。"""
        try:
            self._cfg["geometry"] = self.geometry()
            self._save_config()
        except Exception:
            pass
        self.destroy()

    # ---------------- 配置 ----------------
    def _load_config(self) -> dict:
        try:
            if self._cfg_path.exists():
                return json.loads(self._cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_config(self):
        try:
            self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
            self._cfg_path.write_text(
                json.dumps(self._cfg, ensure_ascii=False, indent=1),
                encoding="utf-8")
        except Exception:
            pass

    # ---------------- 滚动条（细样式，始终显示，网页式行为） ----------------
    def _make_sb(self, parent, cmd):
        """创建细滚动条（width=8），始终显示。"""
        sb = tk.Scrollbar(parent, orient="vertical", command=cmd, width=8,
                          activebackground="#C7C7CC", troughcolor="#F5F5F7",
                          bg="#E5E5E7", relief="flat", bd=0)
        return sb

    def _sb_show(self, sb):
        pass  # 滚动条始终显示，无需闪现

    def _sb_hide(self, sb):
        pass  # 滚动条始终显示，无需隐藏

    def _scroll_canvas(self, canvas, e):
        """网页式滚动：内容不满一屏时滚轮无效（定格），满屏才滚动。"""
        try:
            bb = canvas.bbox("all")
            content_h = bb[3] if bb else 0
            if content_h <= canvas.winfo_height():
                return           # 内容放得下，不滚动
            canvas.yview_scroll(int(-e.delta / 120), "units")
        except tk.TclError:
            pass

    # ================= 数据 =================
    def _load_matches(self):
        p = BASE_DIR / "data" / "jczq_cache" / "matches.json"
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _match_date(self, m) -> str:
        """从 match_time '08-17 18:00截止' 提取比赛日期 '2026-08-17'。"""
        mm_dd = (m.get("match_time") or "")[:5]
        if len(mm_dd) == 5 and mm_dd[2] == "-":
            return f"{time.strftime('%Y')}-{mm_dd}"
        return ""

    WEEK_CN = {"Monday": "周一", "Tuesday": "周二", "Wednesday": "周三",
               "Thursday": "周四", "Friday": "周五", "Saturday": "周六",
               "Sunday": "周日"}

    def _today_prefix(self) -> str:
        """今日竞彩编号前缀（今天星期几对应 '周一'~'周日'）。"""
        return self.WEEK_CN.get(time.strftime("%A"), "周一")

    def _today_matches(self) -> list:
        """预测范围内的比赛：今天+明天两个期次（编号前缀）的全部场次。

        与 predict_v3 口径一致（按期次过滤，不是按截止日期）——
        明天期次即使截止到后天凌晨（如周三002 截止08-20）也要显示+预测。
        后天及更远的期次不展示（避免"显示了但不预测"的困惑）。
        """
        week = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        today_p = week[datetime.now().weekday()]
        tomorrow_p = week[(datetime.now() + timedelta(days=1)).weekday()]
        return [m for m in self._cache
                if (m.get("no") or "")[:2] in (today_p, tomorrow_p)]

    def _today(self):
        return self._today_matches()

    # ================= 顶栏（已去掉：标题在 self.title() 原生标题栏显示） =================
    def _titlebar(self):
        pass  # 原生标题栏足够，自定义 Frame 顶栏去掉避免重复/可拖错觉

    # ================= 主体 =================
    def _body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        self._sb_frame = tk.Frame(body, bg=SIDEBAR, width=self._sb_width,
                                  highlightbackground=BORDER, highlightthickness=1)
        self._sb_frame.pack(side="left", fill="y")
        self._sb_frame.pack_propagate(False)
        self._sidebar(self._sb_frame)

        # 可拖拽分隔条：调整左右宽度并记住
        sash = tk.Frame(body, bg="#D0D0D4", width=4, cursor="sb_h_double_arrow")
        sash.pack(side="left", fill="y")
        sash.pack_propagate(False)
        sash.bind("<B1-Motion>", self._drag_sash)
        sash.bind("<Button-1>", lambda e: self._drag_start())

        ct = tk.Frame(body, bg=BG)
        ct.pack(side="left", fill="both", expand=True)
        self._content(ct)

    def _drag_start(self):
        self._drag_ok = True

    def _drag_sash(self, event):
        if not getattr(self, "_drag_ok", False):
            return
        w = max(140, min(int(event.x_root - self.winfo_rootx()), 520))
        self._sb_width = w
        self._sb_frame.config(width=w)
        self._cfg["sidebar_width"] = w
        self._save_config()

    # ---------------- 侧边栏 ----------------
    def _sidebar(self, parent):
        head = tk.Frame(parent, bg=SIDEBAR)
        head.pack(fill="x", padx=12, pady=(4, 2))
        tk.Label(head, text="赛程", bg=SIDEBAR, fg=TEXT1,
                 font=("Microsoft YaHei UI", 12)).pack(side="left")

        # 滚动 Canvas（侧边栏整体可上下滚）——细滚动条，悬停/滚动时显示
        wrap = tk.Frame(parent, bg=SIDEBAR)
        wrap.pack(fill="both", expand=True)
        self._sb_canvas = tk.Canvas(wrap, bg=SIDEBAR, highlightthickness=0)
        self._sb_sb = self._make_sb(wrap, self._sb_canvas.yview)
        self._sb_canvas.configure(yscrollcommand=self._sb_sb.set)
        self._sb_sb.pack(side="right", fill="y")
        self._sb_canvas.pack(side="left", fill="both", expand=True)
        self._sb_inner = tk.Frame(self._sb_canvas, bg=SIDEBAR)
        self._sb_canvas.create_window((0, 0), window=self._sb_inner, anchor="nw")
        self._sb_inner.bind("<Configure>",
                            lambda e: self._sb_canvas.configure(
                                scrollregion=self._sb_canvas.bbox("all")))
        self._sb_canvas.bind("<MouseWheel>", lambda e: self._scroll_canvas(self._sb_canvas, e))
        # 只在鼠标在侧边栏时滚动侧边栏，离开后恢复全局滚动
        self._scroll_target = "main"
        self._sb_canvas.bind("<Enter>", lambda e: setattr(self, "_scroll_target", "sb"))
        self._sb_canvas.bind("<Leave>", lambda e: setattr(self, "_scroll_target", "main"))

        # 今日赛程：头部点击 → 回到今日视图；历史赛程：头部点击 → 仅展开/收起
        self._group(self._sb_inner, "今日赛程", self._draw_today_list,
                    row=0, header_action=self._render_all)
        self._group(self._sb_inner, "历史赛程", self._draw_history_list, row=2)

    def _group(self, parent, text, draw_fn, row=0, header_action=None):
        """可收起的分组（grid 布局保持顺序：收起再打开仍回原位）。
        - header_action 非空（今日赛程）：头部（标题/空白）点击 → 执行该动作，
          箭头点击 → 单独收起/展开（互不干扰）。
        - header_action 为空（历史赛程）：头部（标题/箭头/空白）点击 → 仅展开/收起，
          不切换主视图（避免误跳到今日）。日期行各自绑定 _show_history。"""
        head = tk.Frame(parent, bg=SIDEBAR, cursor="hand2")
        head.grid(row=row, column=0, sticky="ew", padx=12, pady=(8, 2))
        arrow = tk.Label(head, text="▼", bg=SIDEBAR, fg=TEXT2,
                         font=("Microsoft YaHei UI", 8))
        arrow.pack(side="left")
        title_w = tk.Label(head, text=text, bg=SIDEBAR, fg=TEXT1,
                           font=("Microsoft YaHei UI", 11, "bold"))
        title_w.pack(side="left", padx=(4, 0))

        box = tk.Frame(parent, bg=SIDEBAR)
        box.grid(row=row + 1, column=0, sticky="ew", padx=8)

        state = {"open": True}

        def toggle(_=None):
            state["open"] = not state["open"]
            arrow.config(text="▼" if state["open"] else "▶")
            if state["open"]:
                box.grid()
            else:
                box.grid_remove()

        # 箭头点击 = 收起/展开（不冒泡到 head）
        arrow.bind("<Button-1>", lambda e: (toggle(), "break")[1])
        if header_action:
            # 今日赛程：标题/空白点击 → 回到今日视图（不冒泡）
            title_w.bind("<Button-1>",
                         lambda e: (header_action(), "break")[1])
            head.bind("<Button-1>", lambda e: header_action())
        else:
            # 历史赛程：标题/空白点击 = 收起/展开（不触发视图切换）
            title_w.bind("<Button-1>", lambda e: (toggle(), "break")[1])
            head.bind("<Button-1>", lambda e: toggle())

        draw_fn(box)

    def _draw_today_list(self, box):
        """今日赛程行（仅展示，不可点击）。"""
        self._today_box = box
        self._refresh_today_list()

    def _refresh_today_list(self):
        box = self._today_box
        for w in box.winfo_children():
            w.destroy()
        ms = self._today()
        if not ms:
            tk.Label(box, text="今日暂无赛事", bg=SIDEBAR, fg=TEXT3,
                     font=FONT_S, anchor="w").pack(fill="x", padx=(10, 0), pady=2)
            return
        for m in ms:
            no = m.get("no", "")
            home, away = m.get("home", ""), m.get("away", "")
            rang = m.get("rang", "0")
            row = tk.Frame(box, bg=SIDEBAR)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=no, bg=SIDEBAR, fg=TEXT1, font=FONT_S,
                     width=7, anchor="w").pack(side="left")
            tk.Label(row, text=f"{home} vs {away}", bg=SIDEBAR, fg=TEXT1,
                     font=FONT_S, anchor="w").pack(side="left")
            tk.Label(row, text=f"[{rang}]", bg=SIDEBAR, fg=TEXT3,
                     font=FONT_S).pack(side="right")

    def _draw_history_list(self, box):
        self._hist_box = box
        self._refresh_history_list()

    def _refresh_history_list(self):
        """历史赛程：按日期倒序子分组（今日之外），每日期可展开/收起。"""
        box = self._hist_box
        for w in box.winfo_children():
            w.destroy()
        # 收集历史日期：outputs 导出 + 缓存中已截止（< 今天）的场次。
        # 未截止/未来场次（含预测范围外的后天）既不进今日视图也不进历史。
        today_nos = {m.get("no") for m in self._today()}
        today = time.strftime("%Y-%m-%d")
        dates = {}
        out = get_outputs_dir()
        if out.exists():
            for f in out.glob("竞彩预测数据_*.json"):
                d = f.stem.split("_")[-1]
                if len(d) == 8 and d.isdigit():
                    dates.setdefault(d, {"cnt": 0, "hit": None})
        for m in self._cache:
            if m.get("no") in today_nos:
                continue          # 仍在今日视图的比赛不算历史
            mt = (m.get("match_time") or "")[:5]
            if len(mt) == 5 and mt[2] == "-" and f"{time.strftime('%Y')}-{mt}" < today:
                d = f"{time.strftime('%Y')}{mt.replace('-', '')}"
                dates.setdefault(d, {"cnt": 0, "hit": None})
        if not dates:
            tk.Label(box, text="（暂无历史）", bg=SIDEBAR, fg=TEXT3,
                     font=FONT_S, anchor="w").pack(fill="x", padx=(10, 0), pady=4)
            return
        # 仅显示最近 7 天（含今日往前 7 天）的历史日期，超出自动隐藏
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        recent = [d for d in dates if d >= cutoff]
        for d in sorted(recent, reverse=True)[:7]:
            self._hist_date_group(box, d, dates[d])

    def _hist_date_group(self, parent, d, info):
        """历史日期行：单击直接进入该日赛果详情（无需二次点击）。"""
        head = tk.Frame(parent, bg=SIDEBAR, cursor="hand2")
        head.pack(fill="x", padx=(10, 0), pady=1)
        arrow = tk.Label(head, text="▶", bg=SIDEBAR, fg=TEXT2, font=FONT_S)
        arrow.pack(side="left")
        label_txt = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        if info.get("cnt"):
            label_txt += f" {info['cnt']}场"
        if info.get("hit"):
            label_txt += f" 命中{info['hit'][0]}/{info['hit'][1]}"
        tk.Label(head, text=label_txt, bg=SIDEBAR, fg=TEXT1,
                 font=FONT_S, anchor="w").pack(side="left", padx=(4, 0))
        # 单击整行 → 立即进入该日历史赛果
        for w in (head, arrow) + tuple(head.winfo_children()):
            w.bind("<Button-1>", lambda e, dd=d: self._show_history(dd))

    def _hist_count(self, d):
        try:
            f = get_outputs_dir() / f"竞彩预测数据_{d}.json"
            return len(json.loads(f.read_text(encoding="utf-8")).get("results", []))
        except Exception:
            return 0

    def _hist_hit(self, d):
        try:
            f = get_outputs_dir() / f"竞彩预测数据_{d}.json"
            chk = json.loads(f.read_text(encoding="utf-8")).get("回填核对", {})
            if chk.get("总命中率") is not None:
                return (chk["总命中"], chk["总预测"])
        except Exception:
            pass
        return None

    # ---------------- 内容区 ----------------
    def _content(self, parent):
        head = tk.Frame(parent, bg=BG)
        head.pack(fill="x", padx=20, pady=(2, 4))
        self._title_lbl = tk.Label(head, text="", bg=BG, fg=TEXT1,
                                   font=("Microsoft YaHei UI", 15, "bold"), cursor="hand2")
        self._title_lbl.pack(side="left")
        self._title_lbl.bind("<Button-1>", lambda e: self._render_all())
        self._count_lbl = tk.Label(head, bg="#E9E9EB", fg=TEXT2, padx=8, pady=2,
                                   font=("Microsoft YaHei UI", 10, "bold"))
        self._count_lbl.pack(side="left", padx=(10, 0))

        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="both", expand=True, padx=20, pady=(0, 0))
        self._canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        self._vsb = self._make_sb(wrap, self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vsb.set)
        self._vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self._list = tk.Frame(self._canvas, bg=BG)
        self._canvas.create_window((0, 0), window=self._list, anchor="nw")
        self._list.bind("<Configure>", lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.bind_all("<MouseWheel>", lambda e: (
            self._scroll_canvas(self._sb_canvas, e)
            if self._scroll_target == "sb"
            else self._scroll_canvas(self._canvas, e)))
        self._canvas.bind("<Enter>", lambda e: setattr(self, "_scroll_target", "main"))

        self._bt = tk.Frame(parent, bg=BG)
        self._bt.pack(fill="x", pady=(0, 14))
        self._update_buttons()

        self._set_title()
        self._render()      # 界面元素已就绪，主动渲染初始内容

    def _update_buttons(self):
        """根据当前视图重建底部按钮：今日→导出；历史→查看详情+返回今日。"""
        for w in self._bt.winfo_children():
            w.destroy()
        if self._view == "history":
            row = tk.Frame(self._bt, bg=BG)
            row.pack()
            tk.Button(row, text="查看详情", command=self._open_detail, bg=BLUE, fg="#fff",
                      relief="flat", bd=0, cursor="hand2",
                      font=("Microsoft YaHei UI", 12, "bold"), padx=26, pady=7,
                      activebackground="#0A84FF", activeforeground="#fff").pack(side="left", padx=6)
            tk.Button(row, text="返回今日", command=self._render_all, bg="#E9E9EB",
                      fg=TEXT1, relief="flat", bd=0, cursor="hand2",
                      font=("Microsoft YaHei UI", 12, "bold"), padx=26, pady=7,
                      activebackground="#DCDCE0").pack(side="left", padx=6)
        else:
            row = tk.Frame(self._bt, bg=BG)
            row.pack()
            tk.Button(row, text="导出", command=self._export, bg=BLUE, fg="#fff",
                      relief="flat", bd=0, cursor="hand2",
                      font=("Microsoft YaHei UI", 13, "bold"), padx=30, pady=8,
                      activebackground="#0A84FF", activeforeground="#fff").pack(
                side="left", padx=6)

    def _set_title(self):
        # 标题体现预测范围（今天+明天两个期次），如「周二/周三赛程预测分析」
        tp = self.WEEK_CN.get(time.strftime("%A"), "周一")
        yp = self.WEEK_CN.get((datetime.now() + timedelta(days=1)).strftime("%A"), tp)
        self._title_lbl.config(text=f"{tp}/{yp}赛程预测分析")
        self._count_lbl.config(text=f"{len(self._today())}场")

    # ---------------- 卡片 ----------------
    def _render_all(self):
        self._filter_no = None
        self._view = "today"
        self._view_date = None
        if hasattr(self, "_bt"):       # 初始化未完成时保护
            self._update_buttons()
        if hasattr(self, "_title_lbl"):  # 回到今日时同步刷新大标题与场次计数
            self._set_title()
        self._render()

    def _render(self):
        if not hasattr(self, "_list") or not self._list.winfo_exists():
            return          # 界面元素尚未创建（刷新回调提前触发），跳过
        for w in self._list.winfo_children():
            w.destroy()
        ms = self._today()
        if self._filter_no:
            ms = [m for m in ms if m.get("no") == self._filter_no]
        cur_day = None
        for m in ms:
            # 按期次分组：与导出推荐文件一致的「━━━ 周X ━━━」分割线
            day = (m.get("no") or "")[:2]
            if day != cur_day:
                cur_day = day
                self._day_divider(day)
            self._card(m)

    def _day_divider(self, day):
        """期次分组分割线（同导出格式）：━━━ 周X ━━━"""
        line = tk.Frame(self._list, bg=BG)
        line.pack(fill="x", pady=(12, 2))
        tk.Label(line, text=f"━━━ {day} ━━━", bg=BG, fg=TEXT2,
                 font=("Microsoft YaHei UI", 11, "bold")).pack()

    def _card(self, entry):
        no = entry.get("no", "")
        home, away = entry.get("home", ""), entry.get("away", "")
        league = entry.get("league", "")
        rang = entry.get("rang", "0")
        try:
            ri = int(rang)
        except (TypeError, ValueError):
            ri = 0
        rang_txt = "无让球" if ri == 0 else (f"客让{ri}球" if ri > 0 else f"主让{abs(ri)}球")
        t = entry.get("match_time", "")

        card = tk.Frame(self._list, bg=CARD, highlightbackground=BORDER,
                        highlightthickness=1, cursor="hand2")
        card.pack(fill="x", pady=3)

        # 头部行
        row = tk.Frame(card, bg=CARD, cursor="hand2")
        row.pack(fill="x", padx=12, pady=9)
        tk.Label(row, text=no, bg=CARD, fg=TEXT1, font=FONT_B, width=7,
                 anchor="w").pack(side="left")
        tk.Label(row, text=f"{home} vs {away}", bg=CARD, fg=TEXT1,
                 font=FONT_B, width=26, anchor="w").pack(side="left")
        tk.Label(row, text=f"{league} · {rang_txt} · {t}", bg=CARD, fg=TEXT3,
                 font=FONT_S, width=40, anchor="w").pack(side="left", padx=(8, 0))
        # 预测区：两列固定宽度，保证推荐与胜平负文字各自对齐
        pred_box = tk.Frame(row, bg=CARD)
        pred_box.pack(side="left", padx=(8, 0))
        pred_main = tk.Label(pred_box, text="预测中…", bg=CARD, fg=TEXT3,
                             font=FONT_B, width=15, anchor="w")
        pred_main.pack(side="left")
        pred_ht = tk.Label(pred_box, text="", bg=CARD, fg=TEXT3,
                           font=FONT_S, width=22, anchor="w")
        pred_ht.pack(side="left")
        arrow = tk.Label(row, text="▶", bg=CARD, fg=TEXT2, font=FONT_S)
        arrow.pack(side="right")

        for w in (card, row, arrow) + tuple(row.winfo_children()):
            w.bind("<Button-1>", lambda e, en=entry, c=card, a=arrow: self._toggle(en, c, a))

        # 让球胜平负：V3 模型，打开即后台预测（_auto_predict_v3），完成后 _render 刷新
        if no in self._v3:
            self._fill_pred(pred_box, no)
        else:
            pred_main.config(text="让球预测中…")

    def _safe_text(self, lbl_or_box, text):
        try:
            if not lbl_or_box.winfo_exists():
                return
            if isinstance(lbl_or_box, tk.Frame):
                children = lbl_or_box.winfo_children()
                if children:
                    children[0].config(text=text)
                if len(children) > 1:
                    children[1].config(text="")
            else:
                lbl_or_box.config(text=text)
        except tk.TclError:
            pass

    def _fill_pred(self, lbl_or_box, no):
        """让球胜平负（V4 专项模型）。胜平负(赔率)/总进球 由 derive_extra 给出。"""
        v3 = self._v3.get(no) if no else None
        if not v3:
            if isinstance(lbl_or_box, tk.Frame):
                ch = lbl_or_box.winfo_children()
                if ch:
                    ch[0].config(text="让球预测中…", fg=TEXT3, font=FONT_B)
                if len(ch) > 1:
                    ch[1].config(text="", fg=TEXT3)
            else:
                lbl_or_box.config(text="让球预测中…", fg=TEXT3, font=FONT_B)
            return
        top_label = v3["pred_range"]
        conf = v3.get("conf", 0)
        star = " ★" if conf >= 0.80 else ""   # 与推荐文件一致：★=置信≥80%
        main_txt = f"{top_label} {conf:.0%}{star}"
        color = GREEN if top_label == "让胜" else (YELLOW if top_label == "让平" else RED)
        # 胜平负(赔率反推)：V4 derive_extra 推导
        sp_l = v3.get("sp_label", "-")
        ht_txt = f"胜平负 {sp_l}"

        if isinstance(lbl_or_box, tk.Frame):
            children = lbl_or_box.winfo_children()
            main_lbl = children[0] if children else lbl_or_box
            ht_lbl = children[1] if len(children) > 1 else None
            main_lbl.config(text=main_txt, fg=color, font=FONT_B)
            if ht_lbl:
                ht_lbl.config(text=ht_txt, fg=TEXT3, font=FONT_S)
        else:
            lbl_or_box.config(text=main_txt + "  " + ht_txt, fg=color, font=FONT_B)

    def _toggle(self, entry, card, arrow):
        no = entry.get("no")
        # 收起
        if self._expanded_no == no:
            for w in card.winfo_children()[1:]:
                w.destroy()
            arrow.config(text="▶")
            self._expanded_no = None
            return
        # 详情数据：让球胜平负（V3 专项）
        v3 = self._v3.get(no)
        if not v3:
            messagebox.showinfo("提示", "让球预测还在计算中，稍等再点。")
            return
        # 先收起其他
        for w in card.winfo_children()[1:]:
            w.destroy()
        detail = tk.Frame(card, bg="#FAFAFC", highlightbackground=BORDER,
                          highlightthickness=1)
        detail.pack(fill="x", padx=8, pady=(0, 8))
        txt = self._detail_text(entry, no)
        tk.Label(detail, text=txt, bg="#FAFAFC", fg=TEXT1,
                 justify="left", anchor="w", font=("Microsoft YaHei UI", 10),
                 padx=12, pady=10).pack(fill="x")
        arrow.config(text="▼")
        self._expanded_no = no

    def _detail_text(self, entry, no=None):
        """单场详情（V4 让球专项）。"""
        v3 = self._v3.get(no) if no else None
        rang = int(round((v3 or {}).get("rang", entry.get("rang", 0) or 0)))
        rt = "无让球" if rang == 0 else (f"客让{rang}球" if rang > 0 else f"主让{abs(rang)}球")
        L = [
            f"【{entry.get('no')}】{entry.get('league')}  {entry.get('match_time')}",
            f"球队：{entry.get('home')}  VS  {entry.get('away')}",
            f"让球：{rt}",
            "",
            "── 让球胜平负（V4 让球专项模型）──",
        ]
        if v3 and v3.get("probs"):
            sp = v3["probs"]
            rng_line = (f"让球胜平负（{rt}后）：让胜 {sp[0]:.0%} / 让平 {sp[1]:.0%} / 让负 {sp[2]:.0%}"
                        + ("  （防让平）" if v3.get("rp_second") else "")
                        + ("  ★高置信" if v3.get("conf", 0) >= 0.8 else ""))
            if v3.get("downgrade"):
                rng_line += "  ⚠️低可信联赛，不推荐"
            if v3.get("val") is not None and v3.get("mkt_range"):
                rng_line += f"\n市场对比：{v3['mkt_range']} {v3.get('mkt_conf', 0):.0%}，价值 {v3.get('val', 0):+.0%}"
        else:
            rng_line = "让球胜平负：V4 暂无数据"
        L.append(rng_line)
        L.append("")
        L.append("── 胜平负（赔率反推）──")
        if v3:
            sp_l = v3.get("sp_label", "-")
            sp_probs = v3.get("sp_probs") or []
            sp_s = " / ".join(f"{x:.0%}" for x in sp_probs) if sp_probs else ""
            L.append(f"普通胜平负：{sp_l}（{sp_s}）")
        else:
            L.append("（暂无数据）")
        return "\n".join(L)

    def _show_history(self, date_str):
        """历史赛程：读取当日已生成的预测数据展示（只读）。"""
        f = get_outputs_dir() / f"竞彩预测数据_{date_str}.json"
        if not f.exists():
            self._status(f"{date_str} 无预测记录")
            return
        self._view = "history"
        self._view_date = date_str
        self._update_buttons()
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        rows = data.get("results", [])
        # 隐藏未开赛场次：仅展示已有实际赛果的比赛（避免把「未开赛/次日批次」混入回顾）
        rows = [r for r in rows if str(r.get("实际比分") or "").strip() not in ("", "未开赛", "None")]
        chk = data.get("回填核对", {})
        title = f"{date_str[4:6]}月{date_str[6:8]}日历史预测"
        if chk.get("总命中率") is not None:
            title += f"  命中 {chk['总命中']}/{chk['总预测']}（{chk['总命中率']:.0%}）"
        self._title_lbl.config(text=title)
        self._count_lbl.config(text=f"{len(rows)}场")
        for w in self._list.winfo_children():
            w.destroy()
        if not rows:
            tk.Label(self._list, text="该日无预测数据", bg=BG, fg=TEXT3,
                     font=FONT_S).pack(pady=20)
            return
        for row in rows:
            self._hist_card(row)

    def _hist_card(self, row):
        card = tk.Frame(self._list, bg=CARD, highlightbackground=BORDER,
                        highlightthickness=1)
        card.pack(fill="x", pady=3)
        r = tk.Frame(card, bg=CARD)
        r.pack(fill="x", padx=12, pady=9)
        tk.Label(r, text=row.get("编号", ""), bg=CARD, fg=TEXT1, font=FONT_B,
                 width=7, anchor="w").pack(side="left")
        tk.Label(r, text=f"{row.get('主队','')} vs {row.get('客队','')}", bg=CARD,
                 fg=TEXT1, font=FONT_B, width=26, anchor="w").pack(side="left")
        tk.Label(r, text=f"{row.get('赛事','')} · {row.get('让球','')}", bg=CARD, fg=TEXT3,
                 font=FONT_S, width=40, anchor="w").pack(side="left", padx=(8, 0))
        # 预测推荐 TOP1/TOP2（让球胜平负概率前二）
        _rq_map = {"让胜": 0, "让平": 1, "让负": 2}
        cands = {"让胜": row.get("让胜概率RS", 0) or 0,
                 "让平": row.get("让平概率RP", 0) or 0,
                 "让负": row.get("让负概率RF", 0) or 0}
        ranked = sorted(cands.items(), key=lambda x: -x[1])
        t1, t2 = ranked[0][0], ranked[1][0]
        tk.Label(r, text=f"{t1}/{t2}", bg=CARD, fg=GREEN, font=FONT_B,
                 anchor="w").pack(side="left", padx=(8, 0))
        # 实际赛果 + 命中（TOP1/TOP2 任一命中即打勾）
        score = row.get("实际比分")
        if score and str(score).strip() not in ("", "未开赛", "None"):
            act = str(row.get("实际让球结果", "") or "")
            code_act = _rq_map.get(act.split()[-1], None)   # "RS 让胜" → 让胜 → 0
            hit = (code_act is not None and
                   (code_act == _rq_map.get(t1) or code_act == _rq_map.get(t2)))
            mark = "✓" if hit else "✗"
            color = GREEN if hit else RED
            tk.Label(r, text=f"{score} {mark}", bg=CARD, fg=color, font=FONT_B,
                     anchor="w").pack(side="left", padx=(8, 0))

    # ---------------- 操作 ----------------
    def _refresh(self):
        self._status("正在获取最新赛程…")
        threading.Thread(target=self._refresh_worker, daemon=True).start()

    def _refresh_worker(self):
        try:
            from src.jczq_fetcher import fetch_matches
            ms = fetch_matches(use_cache=False)
            if ms:
                self._cache = ms
                self.after(0, self._refresh_today_list)
                self.after(0, self._set_title)
                self.after(0, self._render_all)
                self.after(0, lambda: self._status(f"已获取最新赛程：{len(ms)} 场"))
            else:
                self.after(0, lambda: self._status("获取失败：未取到比赛，使用缓存"))
        except Exception as e:
            # 用默认参数捕获 e（except 变量在块结束后被删除，直接闭包引用会 NameError）
            self.after(0, lambda e=e: self._status(f"获取失败：{e}，使用缓存"))

    # ---------------- 启动自动预测（打开软件即对今日全部比赛预测） ----------------
    def _auto_predict_all(self):
        """打开软件即后台跑 V3 让球胜平负预测（旧模型 V2 已移除）。"""
        self._auto_predict_v3()

    # ---------- V3 让球胜平负（让球专项模型，替换旧模型的让球推荐） ----------
    def _auto_predict_v3(self):
        """打开即后台跑 predict_v3.py（已有模型，无需重训），完成后载入并刷新。
        防重入：预测进行中 _v3 仍为空，用 _v3_running 标志避免重复起线程并发。"""
        if self._v3 or getattr(self, "_v3_running", False):
            return
        self._v3_running = True
        threading.Thread(target=self._v3_worker, daemon=True).start()

    def _v3_worker(self):
        try:
            import subprocess
            # 打包版（frozen）：调用同目录 predict_v3.exe；源码版：venv python 跑 predict_v3.py
            if getattr(sys, "frozen", False):
                exe = Path(sys.executable).resolve().parent / "predict_v4.exe"
                cmd = [str(exe), "--save"]
            else:
                cmd = [V3_PY, str(BASE_DIR / "predict_v4.py"), "--save"]
            r = subprocess.run(
                cmd,
                cwd=str(BASE_DIR), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=1800)
            if r.returncode != 0:
                raise RuntimeError((r.stdout or "")[-300:] + (r.stderr or "")[-300:])
            self._load_v3_preds()
            if self._v3:
                self.after(0, lambda: (self._render(),
                                       self._status(f"V4 让球胜平负已载入（{len(self._v3)} 场）")))
            else:
                self.after(0, lambda: self._status("V4 无产出（今日无让球数据）"))
        except Exception as e:
            self.after(0, lambda: self._status(f"V4 预测失败：{str(e)[:60]}"))
        finally:
            self._v3_running = False

    def _load_v3_preds(self):
        # 打包版（frozen）：从桌面存档读（predict_v3.exe 写那里）；源码版：项目 outputs/
        if getattr(sys, "frozen", False):
            p = get_outputs_dir() / f"预测_v4_{time.strftime('%Y%m%d')}.json"
        else:
            p = BASE_DIR / "outputs" / f"预测_v4_{time.strftime('%Y%m%d')}.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._v3 = {r["no"]: r for r in data.get("results", [])}
        except Exception:
            self._v3 = {}

    def _export(self):
        """导出：写存档 JSON + 日期文件夹只留 2 张表（赛果预测 / 赛果核对 xlsx）。
        不再生成 让球胜平负推荐 txt/csv、更新包 zip 等中间产物。"""
        if not self._v3:
            self._auto_predict_v3()
            messagebox.showinfo("提示", "V4 让球模型正在预测，稍候再点导出。")
            return
        ms = self._today()
        if not ms:
            messagebox.showinfo("提示", "今日无赛事，无需导出。")
            return

        d = time.strftime("%Y%m%d")
        date_iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        desktop_root = Path(os.path.expanduser("~")) / "Desktop" / "竞彩预测"
        date_dir = desktop_root / d
        date_dir.mkdir(parents=True, exist_ok=True)
        archive = get_outputs_dir()

        # ---- 1) 写存档 JSON（供历史视图/回填/daily_check 用，存档文件夹设为隐藏）----
        results = []
        for m in ms:
            v = self._v3.get(m.get("no"))
            if not v or not v.get("probs"):
                continue
            sp = v["probs"]
            rang = int(v.get("rang", 0) or m.get("rang", 0) or 0)
            results.append({
                "编号": m.get("no"), "赛事": m.get("league", ""),
                "主队": m.get("home"), "客队": m.get("away"),
                "让球": self._rang_text(rang), "日期": date_iso,
                "推荐让球胜平负": v["pred_range"],
                "让胜概率RS": round(sp[0], 3), "让平概率RP": round(sp[1], 3),
                "让负概率RF": round(sp[2], 3), "胜平负": v.get("sp_label", ""),
                "实际比分": None, "是否命中": None,
            })
        json_data = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "model": "models_v4（让球胜平负专项）",
                     "results": results}
        (archive / f"竞彩预测数据_{d}.json").write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # ---- 2) 由 daily_check 生成 赛果预测 + 赛果核对 两张表到日期文件夹 ----
        try:
            import daily_check
            rc = daily_check.run(d)
            ok = (rc == 0)
        except Exception as e:  # noqa: BLE001
            ok = False
            self._status(f"导出失败：{e}")

        # ---- 3) 日期文件夹只保留 2 张 xlsx，清理历史遗留的 txt/csv/推荐目录等 ----
        for f in date_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() != ".xlsx":
                try:
                    f.unlink()
                except OSError:
                    pass
        for dd in list(date_dir.iterdir()):
            if dd.is_dir():
                try:
                    if not any(dd.iterdir()):
                        dd.rmdir()
                except OSError:
                    pass

        _hide_folder(archive)
        self._refresh_history_list()
        if ok:
            self._status(f"已导出 → 桌面/竞彩预测/{d}/（赛果预测 + 赛果核对 两张表）")
            messagebox.showinfo(
                "完成",
                f"已导出到 桌面/竞彩预测/{d}/\n"
                f"· 赛果预测_{d}.xlsx\n"
                f"· 赛果核对_{d}.xlsx\n"
                f"（存档文件夹已设隐藏，保存预测 JSON 供次日核对）")
        else:
            self._status(f"导出未完成：daily_check 返回失败")

    # ---------------- 让球胜平负推荐（V3 专项） ----------------
    def _handicap_for(self, m):
        """返回 (主推, [rs,rp,rf], 来源标签, v3记录或None)。V3 专项，无回退。"""
        no = m.get("no")
        v = self._v3.get(no)
        if v and v.get("probs"):
            p = v["probs"]
            main = v["pred_range"] + ("/让平" if v.get("rp_second") else "")
            return main, [p[0], p[1], p[2]], "V4", v
        return None, [0, 0, 0], "-", None

    def _rang_text(self, rang):
        return ("无让球" if rang == 0 else
                (f"客让{rang}球" if rang > 0 else f"主让{abs(rang)}球"))

    def _rang_of(self, v, m):
        """安全取让球数（V3 为 int；比赛条目 rang 可能为字符串）。"""
        if v:
            return int(v.get("rang", 0) or 0)
        return int(m.get("rang", 0) or 0)

    def _format_handicap_txt(self, matches, d, day=None):
        return format_handicap_txt(matches, self._v3, d, day=day)

    def _format_handicap_csv(self, matches, day=None):
        return format_handicap_csv(matches, self._v3, day=day)

    # ---------------- 第二天核验文档 ----------------
    def _format_verify_txt(self, matches, d, day=None):
        mm = f"{int(d[4:6])}月{int(d[6:8])}日"
        head_txt = f"第二天核验文档（{mm} {day} 预测）" if day else f"第二天核验文档（{mm} 预测）"
        L = [head_txt, "",
             "【使用说明】次日比赛结束后，在「实际比分 / 实际让球结果 / 命中」三栏填写：",
             "  1) 实际比分：主队进球:客队进球",
             "  2) 实际让球结果：主队进球 + 让球数 与 客队进球 比较 → 大于=让胜，等于=让平，小于=让负",
             "  3) 命中：实际让球结果 与 本日推荐(让球胜平负) 一致填 √，否则填 ×",
             ""]
        head = ("编号".ljust(7) + "联赛".ljust(10) + "主队vs客队".ljust(18)
                + "让球".ljust(8) + "本日推荐".ljust(11)
                + "RS".ljust(7) + "RP".ljust(7) + "RF".ljust(7)
                + "实际比分".ljust(10) + "实际让球结果".ljust(13) + "命中")
        L.append(head)
        L.append("-" * 95)
        for m in sorted(matches, key=lambda x: x.get("no", "")):
            if day is not None and (m.get("no") or "")[:2] != day:
                continue
            main, p, src, v = self._handicap_for(m)
            if main is None:
                continue
            rang = self._rang_of(v, m)
            rt = self._rang_text(rang)
            pair = f"{m.get('home')}vs{m.get('away')}"
            rs_s, rp_s, rf_s = f"{p[0]:.0%}", f"{p[1]:.0%}", f"{p[2]:.0%}"
            tag = "" if src == "V4" else f"({src})"
            L.append(
                m.get("no", "").ljust(7) + m.get("league", "").ljust(10)
                + pair.ljust(18) + rt.ljust(8) + (main + tag).ljust(11)
                + rs_s.ljust(7) + rp_s.ljust(7) + rf_s.ljust(7)
                + "__:__".ljust(10) + "____".ljust(13) + "____")
        L.append("")
        L.append("【汇总】预测 __ 场，命中 __ 场，命中率 __%")
        return "\n".join(L) + "\n"

    def _format_verify_csv(self, matches, day=None):
        rows = ["编号,联赛,主队,客队,让球,本日推荐,RS让胜,RP让平,RF让负,"
                "实际比分,实际让球结果,命中,来源"]
        for m in sorted(matches, key=lambda x: x.get("no", "")):
            if day is not None and (m.get("no") or "")[:2] != day:
                continue
            main, p, src, v = self._handicap_for(m)
            if main is None:
                continue
            rang = self._rang_of(v, m)
            rt = self._rang_text(rang)
            rows.append(",".join([m.get("no", ""), m.get("league", ""), m.get("home", ""),
                                  m.get("away", ""), rt, main,
                                  f"{p[0]:.0%}", f"{p[1]:.0%}", f"{p[2]:.0%}",
                                  "", "", "", src]))
        return "\n".join(rows) + "\n"

    # ---------------- 查看详情（打开当日赛果核对 Excel） ----------------
    def _open_detail(self):
        """打开选中当日已生成的赛果核对表：桌面/竞彩预测/{d}/赛果核对_{d}.xlsx。
        对应用户「查看详情」：有核对表就打开文件；无记录时提示。"""
        if not self._view_date:
            return
        chk_xlsx = (Path.home() / "Desktop" / "竞彩预测" / self._view_date
                    / f"赛果核对_{self._view_date}.xlsx")
        if not chk_xlsx.exists():
            self._status(f"{self._view_date} 尚未生成赛果核对表（需先完成当日导出）")
            return
        try:
            import os
            os.startfile(str(chk_xlsx))     # 系统默认程序打开 xlsx
        except Exception as e:
            self._status(f"打开失败：{e}")

    def _status(self, text):
        if not hasattr(self, "status_bar"):
            self.status_bar = tk.Label(self, text=text, bg=BG, fg=TEXT2, anchor="w",
                                       font=("Microsoft YaHei UI", 9))
            self.status_bar.pack(side="bottom", fill="x", padx=20, pady=(0, 6))
        else:
            self.status_bar.config(text=text)

    # ---------------- 启动补跑昨日核对 ----------------
    def _check_pending_backfill(self):
        """打开软件时自动补跑昨日赛果核对：
        昨天有预测记录但今天还没生成核对表 → 后台补跑 daily_check。"""
        def worker():
            try:
                y = time.strftime("%Y%m%d", time.localtime(time.time() - 86400))
                pred = get_outputs_dir() / f"竞彩预测数据_{y}.json"
                chk = get_outputs_dir() / f"竞彩赛果核对_{y}.xlsx"
                if pred.exists() and not chk.exists():
                    import daily_check
                    rc = daily_check.run(y, no_desktop=False)
                    if rc == 0:
                        try:
                            data = json.loads(pred.read_text(encoding="utf-8"))
                            s = data.get("回填核对", {})
                            msg = (f"已补跑昨日赛果核对：命中 {s.get('总命中', 0)}/"
                                   f"{s.get('总预测', 0)} 场，核对表已生成")
                        except Exception:
                            msg = "已补跑昨日赛果核对，核对表已生成"
                    else:
                        msg = "昨日无预测记录，核对跳过"
                    self.after(0, lambda: self._status(msg))
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()

    # ---------------- 启动自动贝叶斯更新（V4 自更新） ----------------
    def _check_bayes_update(self):
        """打开软件时后台调用 bayes_update.exe --auto：
        如果 update_pool 攒够 7 个更新包或今天是周一，则自动 吃包+重训+验证替换。
        仅在有 bayes_update.exe（打包版）或 bayes_update.py（源码版）时触发，防误报。"""
        def worker():
            try:
                if getattr(sys, "frozen", False):
                    exe = Path(sys.executable).resolve().parent / "bayes_update.exe"
                    if not exe.exists():
                        return
                    cmd = [str(exe), "--auto"]
                else:
                    py = "C:/Users/CHENYI/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
                    cmd = [py, str(BASE_DIR / "bayes_update.py"), "--auto"]
                import subprocess
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=3600)
                tail = (r.stdout or "")[-200:].strip()
                if tail:
                    self.after(0, lambda: self._status("贝叶斯更新：" + tail.splitlines()[-1][:60]))
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()


HOVER_BG = "#E9E9EB"


def main():
    _setup_hidden()          # 隐藏 data_v4 / update_pool / 存档
    app = App()
    app.after(300, app._refresh)       # 启动自动获取最新赛程
    app.after(800, app._render_all)    # 先渲染缓存数据
    app.after(1500, app._check_pending_backfill)  # 补跑昨日赛果核对
    app.after(3000, app._check_bayes_update)      # 后台自动贝叶斯更新(V4自更新)
    app.mainloop()


# ---------------- 让球胜平负推荐（V3 专项，模块级，可独立调用） ----------------
def _hl_rang(rang):
    """让球文字：主让 / 客让 / 平手。"""
    rang = int(rang or 0)
    return ("主让" + str(-rang) if rang < 0 else
            "客让" + str(rang) if rang > 0 else "平手")


def format_handicap_txt(matches, v3_map, d, day=None):
    """让球胜平负推荐（V3 专项）——最终确认格式：
    标题(YYYY-MM-DD [周X] 竞彩让球推荐（V3模型 60.2%）) + 副标题(共N场 | ★=置信≥80%)
    + 按比赛日分组 ━━━ 周X ━━━，每场两行（编号 主vs客 [让球] / 主推[次推] 置信%★），场间空行。
    输出全部预测场次（有 probs 即出，含低置信；★=置信≥80%；双推=让平排老二）。
    day 指定期次（如"周二"）时只输出该期次（导出按期次分文件）。"""
    iso = f"{d[0:4]}-{d[4:6]}-{d[6:8]}"
    picked = [(m, v) for m in matches
              if (v := v3_map.get(m.get("no")))
              and v.get("probs")
              and (day is None or (m.get("no") or "")[:2] == day)]
    # 保持 matches（json results）原始顺序：期次按自然日期排列（周三→周四→周五…），
    # 不按周几字母排序，避免跨周时（如 8/12 含 周三/四/五）分组顺序错乱。
    title = f"{iso} 竞彩让球推荐（V4模型）"
    if day:
        title = f"{iso} {day} 竞彩让球推荐（V4模型）"
    L = [title,
         f"共{len(picked)}场 | ★=置信≥80%", ""]
    cur = None
    for m, v in picked:
        no = m.get("no")
        day_g = (no or "")[:2]
        if day_g != cur:
            cur = day_g
            L.append(f"━━━ {day_g} ━━━")
        star = "★" if v["conf"] >= 0.80 else ""
        rng = _hl_rang(v.get("rang", 0))
        main_pick = v["pred_range"] + ("/让平" if v.get("rp_second") else "")
        L.append(f"{no} {m.get('home')}vs{m.get('away')} [{rng}]")
        L.append(f"{main_pick} {v['conf']:.0%}{star}")
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def format_handicap_csv(matches, v3_map, day=None):
    """让球胜平负推荐 CSV（V3 专项，旧模型 V2 已移除）：含市场主推 / 市场同方向概率 / 价值差。
    day 指定期次（如"周二"）时只输出该期次。"""
    RMAP = {"让胜": 0, "让平": 1, "让负": 2}
    rows = ["编号,联赛,主队,客队,让球,主推,置信,RS让胜,RP让平,RF让负,"
            "市场主推,市场主推概率,市场同方向概率,价值差,低可信,来源"]
    for m in sorted(matches, key=lambda x: x.get("no", "")):
        no = m.get("no")
        if day is not None and (no or "")[:2] != day:
            continue
        v = v3_map.get(no)
        if not (v and v.get("probs")):
            continue
        p = v["probs"]
        main = v["pred_range"] + ("/让平" if v.get("rp_second") else "")
        conf = f"{v['conf']:.0%}"
        rng = _hl_rang(v.get("rang", 0))
        pred_i = RMAP[v["pred_range"]]
        mp = v.get("mkt_probs")
        same_p = f"{mp[pred_i]:.0%}" if mp else ""
        mkt = v.get("mkt_range", "")
        mktc = f"{v['mkt_conf']:.0%}" if v.get("mkt_range") else ""
        val = f"{v['val']:+.0%}" if v.get("val") is not None else ""
        down = "是" if v.get("downgrade") else "否"
        src = "V4"
        rows.append(",".join([no, m.get("league", ""), m.get("home", ""),
                              m.get("away", ""), rng, main, conf,
                              f"{p[0]:.0%}", f"{p[1]:.0%}", f"{p[2]:.0%}",
                              mkt, mktc, same_p, val, down, src]))
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
