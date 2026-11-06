import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
from collections import Counter, defaultdict
from math import comb

st.set_page_config(page_title="双色球最优解计算器", layout="centered")
st.title("双色球最优解计算器")
st.caption("通过海量候选池暴力测算，直接输出结构最优解，不保证中奖。")

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
st.success(f"数据源：{data_source} ｜ 最新期号：{df.iloc[-1]['issue']} ｜ 缓存刷新：5分钟")

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
        rows.append({"号码": num, "综合分": round(comp*100, 1)})
    return pd.DataFrame(rows).sort_values("综合分", ascending=False).reset_index(drop=True), cur_miss_map

score_df, cur_miss_map = compute_scores(df)
blue_counts = Counter(df["blue"].tolist()[-50:])
blue_score_df = pd.DataFrame({
    "蓝球": range(1, 17),
    "热度": [blue_counts.get(i, 0) for i in range(1, 17)]
}).sort_values("热度", ascending=False).reset_index(drop=True)

# ==================== 结构分析与打分 ====================
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
        ("和值", s, 70 <= s <= 130, f"和值{s} {'在' if 70<=s<=130 else '偏离'}高频区"),
        ("奇偶", f"{odd}:{6-odd}", 2 <= odd <= 4, f"奇偶{odd}:{6-odd}"),
        ("区间", f"{zones[0]}-{zones[1]}-{zones[2]}", min(zones) > 0, "三区不断档"),
        ("跨度", span, 15 <= span <= 32, f"跨度{span}"),
        ("连号", "有" if consecutive else "无", True, f"{'含' if consecutive else '无'}连号"),
        ("重号", repeat, 0 <= repeat <= 3, f"重号{repeat}个"),
    ]
    passed = sum(1 for _, _, ok, _ in checks if ok)
    return checks, passed

def get_group_score(reds, score_map):
    checks, passed = structure_analysis(reds)
    # 评分构成：结构占比60%，候选池高分占比40%
    pool_score = sum(score_map[r] for r in reds) / 6
    struct_score = (passed / len(checks)) * 100
    total_score = round(struct_score * 0.6 + pool_score * 0.4, 1)
    return total_score, checks

# ==================== 单式最优解测算 ====================
@st.cache_data(ttl=3600)
def calculate_best_singles(score_list, n_picks=5):
    score_map = dict(score_list)
    # 固定种子，确保每次算出结果一样
    random.seed(42) 
    candidates = []
    # 生成 5 万组候选池
    for _ in range(50000):
        reds = tuple(sorted(random.sample(range(1, 34), 6)))
        candidates.append(reds)
    
    unique_cands = list(set(candidates))
    scored_cands = []
    for cand in unique_cands:
        sc, checks = get_group_score(cand, score_map)
        passed = sum(1 for _, _, ok, _ in checks if ok)
        # 必须满足至少5项结构才进入排序
        if passed >= 5:
            scored_cands.append((sc, list(cand)))
            
    scored_cands.sort(key=lambda x: x[0], reverse=True)
    # 去重（防止选出的组太像）
    final_picks = []
    seen_nums = set()
    for sc, cand in scored_cands:
        if len(final_picks) >= n_picks: break
        if len(set(cand) & seen_nums) >= 5: continue # 如果和前面的重合5个，跳过
        final_picks.append((sc, cand))
        seen_nums.update(cand)
    return final_picks

# ==================== 界面 ====================
tab1, tab2, tab3 = st.tabs(["🎯 最优单式", "📦 最优复式", "📜 历史"])

with tab1:
    st.subheader("测算下一期最优单式")
    n_picks = st.slider("测算前几组", 5, 15, 5)
    if st.button("一键计算最优组合", type="primary", use_container_width=True):
        with st.spinner("正在海量组合中测算最优解..."):
            best_singles = calculate_best_singles(tuple(score_df[["号码", "综合分"]].values.tolist()), n_picks)
            
            for i, (gs, reds) in enumerate(best_singles, 1):
                blue = blue_score_df.iloc[0]["蓝球"] # 默认配最热蓝球
                checks, _ = structure_analysis(reds)
                with st.expander(f"第{i}名 ｜ 评分 {gs} ｜ 红球 {' '.join(f'{x:02d}' for x in reds)} ｜ 蓝球 {blue:02d}", expanded=(i==1)):
                    st.write("**分析原因：**")
                    for name, val, ok, desc in checks:
                        icon = "✅" if ok else "❌"
                        st.write(f"- {icon} {name}：{desc}")
                    st.write(f"- 💡 包含高分号码：{', '.join(map(str, reds))}")

with tab2:
    st.subheader("测算最优复式")
    col1, col2 = st.columns(2)
    with col1:
        n_red = st.selectbox("选几个红球（复式）", [7, 8, 9, 10, 11, 12], index=0)
    with col2:
        m_blue = st.selectbox("选几个蓝球（复式）", [1, 2, 3, 4, 5, 16], index=0)
        
    if st.button("智能推荐最优复式", type="primary", use_container_width=True):
        # 直接取评分最高的前 N 个红球和蓝球
        best_reds = sorted(score_df.head(n_red)["号码"].tolist())
        best_blues = sorted(blue_score_df.head(m_blue)["蓝球"].tolist())
        
        # 复式整体评分：取前6个红球算结构
        top6 = best_reds[:6]
        gs, checks = get_group_score(top6, dict(score_df[["号码", "综合分"]].values.tolist()))
        
        bets = comb(n_red, 6) * m_blue
        amount = bets * PER_BET
        
        st.markdown(f"### 推荐号码")
        st.info(f"**红球：** {' '.join(f'{x:02d}' for x in best_reds)}")
        st.info(f"**蓝球：** {' '.join(f'{x:02d}' for x in best_blues)}")
        
        st.markdown("### 方案评估")
        st.write(f"**方案评分：{gs}**（取红球前6个计算）")
        for name, val, ok, desc in checks:
            if ok: st.write(f"- ✅ {name}：{desc}")
            
        st.markdown("### 金额计算")
        st.metric("组合注数", f"{bets} 注")
        st.metric("总投注金额", f"¥{amount}")

with tab3:
    st.subheader("历史开奖")
    show_n = st.slider("最近期数", 10, 300, 50)
    hist_view = df.tail(show_n).iloc[::-1].copy()
    hist_view["红球"] = hist_view["reds"].apply(lambda x: " ".join(f"{i:02d}" for i in x))
    hist_view["蓝球"] = hist_view["blue"].apply(lambda x: f"{x:02d}")
    st.dataframe(hist_view[["issue", "红球", "蓝球"]].rename(columns={"issue": "期号"}), use_container_width=True, height=400)

st.caption("免责声明：计算基于历史统计结构，不保证中奖，理性购彩。")
