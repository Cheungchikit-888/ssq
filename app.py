import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
from collections import Counter, defaultdict
from math import comb

st.set_page_config(page_title="双色球辅助选号", layout="centered")
st.title("双色球辅助选号工具")
st.caption("基于历史统计生成参考号码，不保证中奖，仅供娱乐参考。")

PER_BET = 2
MAX_AMOUNT = 20000

# ==================== 数据层 ====================
@st.cache_data(ttl=300)
def load_history():
    # 源1：官方
    try:
        url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
        params = {"name": "ssq", "issueCount": 300, "pageNo": 1, "pageSize": 300}
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cwl.gov.cn/"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()
        rows = []
        for item in data["result"]:
            reds = [int(x) for x in item["red"].split(",")]
            rows.append({"issue": item["code"], "date": item.get("date", ""),
                         "reds": reds, "blue": int(item["blue"])})
        df = pd.DataFrame(rows).sort_values("issue").reset_index(drop=True)
        if len(df) > 50:
            return df, "中国福彩网（官方）"
    except Exception:
        pass

    # 源2：500.com
    try:
        url2 = "https://datachart.500.com/ssq/history/newinc/history.php"
        params2 = {"start": "03001", "end": "99999"}
        headers2 = {"User-Agent": "Mozilla/5.0"}
        r2 = requests.get(url2, params=params2, headers=headers2, timeout=10)
        r2.encoding = "gbk"
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r2.text, "html.parser")
        rows = []
        for tr in soup.select("tr.t_tr1")[:500]:
            tds = tr.find_all("td")
            if len(tds) < 8: continue
            try:
                issue = tds[0].get_text(strip=True)
                reds = sorted([int(tds[i].get_text(strip=True)) for i in range(1, 7)])
                blue = int(tds[7].get_text(strip=True))
                rows.append({"issue": issue, "date": "", "reds": reds, "blue": blue})
            except Exception:
                continue
        df2 = pd.DataFrame(rows).sort_values("issue").reset_index(drop=True)
        if len(df2) > 50:
            return df2, "500.com（第三方兜底）"
    except Exception:
        pass

    # 源3：内置示例
    st.warning("实时数据源暂不可用，使用内置示例数据演示。")
    sample = [
        [1,5,12,20,25,33,7],[3,8,15,22,28,31,12],[2,9,14,19,26,30,5],
        [4,11,16,21,27,32,9],[6,10,13,18,24,29,3],[1,7,17,23,28,33,14],
        [2,8,12,19,25,31,6],[5,9,15,22,27,30,11],[3,10,14,20,26,32,8],
        [4,11,16,21,28,33,2],[6,12,17,23,29,31,13],[1,8,13,18,24,30,7],
    ]
    rows = [{"issue": f"DEMO{i+1:03d}", "date": "", "reds": r[:6], "blue": r[6]}
            for i, r in enumerate(sample)]
    return pd.DataFrame(rows), "内置示例（非真实数据）"

df, data_source = load_history()
st.success(f"数据源：{data_source} ｜ 加载 {len(df)} 期 ｜ 最新：{df.iloc[-1]['issue']}")

# ==================== 评分引擎 ====================
def compute_scores(df):
    reds_list = df["reds"].tolist()
    n = len(reds_list)
    windows = [10, 20, 30, 50]

    freq_scores = {}
    for num in range(1, 34):
        vals = []
        for w in windows:
            recent = reds_list[-w:] if n >= w else reds_list
            cnt = sum(1 for d in recent if num in d)
            vals.append(cnt / len(recent))
        freq_scores[num] = np.mean(vals)

    miss_scores, cur_miss_map = {}, {}
    for num in range(1, 34):
        cur = 0
        for d in reversed(reds_list):
            if num in d: break
            cur += 1
        cur_miss_map[num] = cur
        hist, last = [], None
        for i, d in enumerate(reds_list):
            if num in d:
                if last is not None: hist.append(i - last)
                last = i
        avg = np.mean(hist) if hist else 1
        miss_scores[num] = cur / max(avg, 1)

    last_draw = reds_list[-1]
    reweight = {num: (1.0 if num in last_draw else 0.0) for num in range(1, 34)}

    companion = defaultdict(float)
    for num in range(1, 34):
        total = 0
        for d in reds_list:
            if num in d:
                for other in last_draw:
                    if other != num and other in d: total += 1
        companion[num] = total / max(n, 1)

    def norm(d):
        vals = np.array(list(d.values())); mn, mx = vals.min(), vals.max()
        return {k: 0.5 for k in d} if mx == mn else {k: (v - mn) / (mx - mn) for k, v in d.items()}

    f_n, m_n, c_n, r_n = norm(freq_scores), norm(miss_scores), norm(companion), norm(reweight)
    rows = []
    for num in range(1, 34):
        comp = 0.40*f_n[num] + 0.25*m_n[num] + 0.20*c_n[num] + 0.15*r_n[num]
        rows.append({
            "号码": num, "综合分": round(comp*100, 1),
            "频率共识": round(f_n[num]*100, 1), "遗漏张力": round(m_n[num]*100, 1),
            "当前遗漏": cur_miss_map[num], "伴随关联": round(c_n[num]*100, 1),
            "上期重号": "是" if num in last_draw else "",
        })
    return pd.DataFrame(rows).sort_values("综合分", ascending=False).reset_index(drop=True), cur_miss_map

score_df, cur_miss_map = compute_scores(df)

# ==================== 结构分析与生成 ====================
def structure_analysis(reds):
    s = sum(reds)
    odd = sum(1 for x in reds if x % 2 == 1)
    zones = [0, 0, 0]
    for x in reds:
        if x <= 11: zones[0] += 1
        elif x <= 22: zones[1] += 1
        else: zones[2] += 1
    span = max(reds) - min(reds)
    consecutive = any(reds[i+1] - reds[i] == 1 for i in range(5))
    last_draw = set(df["reds"].tolist()[-1])
    repeat = len(set(reds) & last_draw)

    checks = [
        ("和值", s, 70 <= s <= 130, f"{s} {'落在' if 70<=s<=130 else '偏离'}高频区间"),
        ("奇偶比", f"{odd}:{6-odd}", 2 <= odd <= 4, f"奇偶 {odd}:{6-odd} {'在' if 2<=odd<=4 else '不在'}范围内"),
        ("三区分布", f"{zones[0]}-{zones[1]}-{zones[2]}", min(zones) > 0, "三区" + ("均有号" if min(zones)>0 else "断档")),
        ("跨度", span, 15 <= span <= 32, f"跨度{span} {'在' if 15<=span<=32 else '不在'}区间"),
        ("连号", "有" if consecutive else "无", True, f"{'含连号' if consecutive else '无连号'}"),
        ("上期重号", repeat, 0 <= repeat <= 3, f"重号{repeat}个"),
    ]
    passed = sum(1 for _, _, ok, _ in checks if ok)
    return checks, passed, len(checks)

def group_reason(reds, blue, score_df, top_n=15):
    top_nums = set(score_df.head(top_n)["号码"].tolist())
    hit_top = [x for x in reds if x in top_nums]
    checks, passed, total = structure_analysis(reds)
    reasons = [f"包含高分候选号码 {len(hit_top)} 个"]
    reasons += [f"{name}：{detail}" for name, _, ok, detail in checks if ok]
    if len(reasons) < 3:
        reasons += [f"{name}：{detail}" for name, _, ok, detail in checks if not ok]
    score = round(60 + 40 * (passed / total) * 0.5 + 40 * (len(hit_top) / 6) * 0.5, 1)
    return min(score, 100), reasons

def structure_ok(reds):
    checks, passed, total = structure_analysis(reds)
    return passed >= total - 1

def generate_picks(score_df, n_picks=5):
    weights = dict(zip(score_df["号码"], score_df["综合分"]))
    picks, tries = [], 0
    while len(picks) < n_picks and tries < 8000:
        tries += 1
        pool = [(n, weights[n] + 1) for n in weights]
        reds = []
        for _ in range(6):
            total = sum(w for _, w in pool)
            r = random.uniform(0, total); acc = 0
            for i, (num, wt) in enumerate(pool):
                acc += wt
                if r <= acc:
                    reds.append(num); pool.pop(i); break
        reds.sort()
        if structure_ok(reds) and len(set(reds)) == 6:
            blue = random.randint(1, 16)
            picks.append((reds, blue))
    return picks

def combo_count(n_red, m_blue):
    if n_red < 6 or m_blue < 1: return 0
    return comb(n_red, 6) * m_blue

# ==================== 界面（已适配手机） ====================
tab1, tab2, tab3, tab4 = st.tabs(["📊 评分", "🎯 单式", "📦 复式", "📜 历史"])

with tab1:
    st.subheader("红球候选池")
    mobile_score_df = score_df[["号码", "综合分", "频率共识", "遗漏张力", "上期重号"]]
    st.dataframe(mobile_score_df, use_container_width=True, height=400)
    
    st.subheader("蓝球近 50 期")
    bc = Counter(df["blue"].tolist()[-50:])
    bdf = pd.DataFrame({"蓝球": range(1, 17),
                        "出现次数": [bc.get(i, 0) for i in range(1, 17)],
                        "当前遗漏": [next((j for j, d in enumerate(reversed(df["blue"].tolist())) if d == i), -1) for i in range(1, 17)]
                        }).sort_values("出现次数", ascending=False)
    st.dataframe(bdf, use_container_width=True, height=300)

with tab2:
    st.subheader("生成参考号码")
    n_picks = st.slider("生成组数", 1, 10, 5)
    if st.button("立即生成单式号码", type="primary", use_container_width=True):
        picks = generate_picks(score_df, n_picks)
        if not picks:
            st.error("约束过严，未生成结果，请稍后再试。")
        for i, (reds, blue) in enumerate(picks, 1):
            gs, reasons = group_reason(reds, blue, score_df)
            with st.expander(f"第{i}组 ｜ 评分{gs} ｜ 红球 {' '.join(f'{x:02d}' for x in reds)} ｜ 蓝球 {blue:02d}", expanded=(i==1)):
                for r in reasons:
                    st.write(f"- {r}")
                st.caption("评分反映与历史高频结构的吻合度，不代表中奖概率。")

with tab3:
    st.subheader("复式配置")
    red_pool = st.multiselect("选红球（≥6个）", list(range(1, 34)), default=[1, 7, 12, 18, 24, 30, 33])
    blue_pool = st.multiselect("选蓝球", list(range(1, 17)), default=[7, 12])

    if len(red_pool) >= 6 and len(blue_pool) >= 1:
        n_red, m_blue = len(red_pool), len(blue_pool)
        bets = combo_count(n_red, m_blue)
        amount = bets * PER_BET
        
        st.metric("组合注数", f"{bets:,} 注")
        st.metric("投注金额", f"¥{amount:,}")
        
        if amount > MAX_AMOUNT:
            st.error(f"金额超 20000 元上限，请减少号码。")
        else:
            st.success("✅ 金额合规")
            top6 = sorted(score_df[score_df["号码"].isin(red_pool)].head(6)["号码"].tolist())
            if len(top6) == 6:
                checks6, passed6, total6 = structure_analysis(top6)
                gs, _ = group_reason(top6, blue_pool[0], score_df)
                st.write(f"**复式评分：{gs}**")
                st.write(f"**结构归因：** " + "；".join(f"{name} {detail}" for name, _, ok, detail in checks6 if ok))
    else:
        st.warning("请至少选择 6 个红球和 1 个蓝球。")

with tab4:
    st.subheader("历史开奖")
    show_n = st.slider("最近期数", 10, min(300, len(df)), 50)
    hist_view = df.tail(show_n).iloc[::-1].copy()
    hist_view["红球"] = hist_view["reds"].apply(lambda x: " ".join(f"{i:02d}" for i in x))
    hist_view["蓝球"] = hist_view["blue"].apply(lambda x: f"{x:02d}")
    st.dataframe(hist_view[["issue", "红球", "蓝球"]].rename(columns={"issue": "期号"}), use_container_width=True, height=400)

st.caption("免责声明：双色球开奖为独立随机事件，本工具仅供娱乐参考。")
