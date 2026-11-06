import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter, defaultdict
from math import comb

st.set_page_config(page_title="双色球最优解计算器", layout="centered")
st.title("双色球最优解计算器")
st.caption("海量数据测算 + 科学惩罚分散 + 概率热力展示。")

PER_BET = 2
MAX_AMOUNT = 20000

# ==================== 数据层 ====================
@st.cache_data(ttl=300)
def load_history():
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

    st.warning("实时数据源暂不可用，使用内置示例数据演示。")
    sample = [[1,5,12,20,25,33,7],[3,8,15,22,28,31,12],[2,9,14,19,26,30,5]]
    rows = [{"issue": f"DEMO{i+1:03d}", "date": "", "reds": r[:6], "blue": r[6]} for i, r in enumerate(sample)]
    return pd.DataFrame(rows), "内置示例（非真实数据）"

df, data_source = load_history()
st.success(f"数据源：{data_source} ｜ 最新：{df.iloc[-1]['issue']} ｜ 刷新：5分钟")

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
        rows.append({"号码": num, "综合分": round(comp*100, 1), "近50期频次": int(freq_scores[num]*n)})
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
        ("和值", s, 70 <= s <= 130, f"和值{s}"),
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
    struct_score = 100 - (6 - passed) * 15
    if passed == 6: struct_score += 5
    pool_score = sum(score_map[r] for r in reds) / 6
    total_score = round(struct_score * 0.6 + pool_score * 0.4, 1)
    return min(total_score, 99.9), checks

def get_rating(score):
    if score >= 85: return "S级 · 极优解 🌟"
    if score >= 78: return "A级 · 优等解 ⭐"
    if score >= 70: return "B级 · 良好解 ✅"
    return "C级 · 普通解"

# ==================== 单式最优解测算（引入评分惩罚机制） ====================
@st.cache_data(ttl=3600)
def calculate_best_singles(score_list, n_picks=5):
    score_map = dict(score_list)
    random.seed(42) 
    candidates = []
    for _ in range(100000):
        reds = tuple(sorted(random.sample(range(1, 34), 6)))
        candidates.append(reds)
    
    unique_cands = list(set(candidates))
    scored_cands = []
    
    for cand in unique_cands:
        base_sc, checks = get_group_score(cand, score_map)
        passed = sum(1 for _, _, ok, _ in checks if ok)
        if passed >= 5 and base_sc > 70:
            scored_cands.append((base_sc, list(cand)))
            
    # 按基础分降序
    scored_cands.sort(key=lambda x: x[0], reverse=True)
    
    final_picks = []
    seen_reds = []
    blue_idx = 0
    
    for base_sc, cand in scored_cands:
        if len(final_picks) >= n_picks: break
        
        # 【科学惩罚机制】
        overlap_penalty = 0
        valid = True
        for selected in seen_reds:
            overlap_count = len(set(cand) & set(selected))
            if overlap_count >= 4: # 如果重合4个及以上，直接淘汰
                valid = False
                break
            elif overlap_count == 3:
                overlap_penalty += 5 # 如果重合3个，扣5分惩罚
        
        if valid:
            final_sc = round(base_sc - overlap_penalty, 1)
            blue = int(blue_score_df.iloc[blue_idx % 16]["蓝球"])
            final_picks.append((final_sc, cand, blue, overlap_penalty))
            seen_reds.append(cand)
            blue_idx += 1
            
    return final_picks

# ==================== 界面 ====================
tab1, tab2, tab3 = st.tabs(["🎯 最优单式", "📦 最优复式", "🔥 概率热力图"])

with tab1:
    st.subheader("测算最优单式")
    st.caption("已启用【温和惩罚机制】：极优解会被保留，重合度过高的组合会被扣分降级。")
    n_picks = st.slider("测算前几组", 5, 15, 5)
    if st.button("一键计算最优组合", type="primary", use_container_width=True):
        with st.spinner("正在测算..."):
            best_singles = calculate_best_singles(tuple(score_df[["号码", "综合分"]].values.tolist()), n_picks)
            
            for i, (gs, reds, blue, penalty) in enumerate(best_singles, 1):
                checks, _ = structure_analysis(reds)
                rating = get_rating(gs)
                
                expander_title = f"第{i}名 ｜ {rating} ｜ 评分 {gs} ｜ 红球 {' '.join(f'{x:02d}' for x in reds)} ｜ 蓝球 {blue:02d}"
                if penalty > 0:
                    expander_title += f" (-{penalty}分散惩罚)"
                
                with st.expander(expander_title, expanded=(i==1)):
                    st.write("**结构分析：**")
                    for name, val, ok, desc in checks:
                        icon = "✅" if ok else "⚠️"
                        st.write(f"- {icon} {name}：{desc}")
                    st.write(f"- 💡 号码质量：平均综合分 {round(sum(dict(score_df[['号码', '综合分']].values)[r] for r in reds)/6, 1)}")

with tab2:
    st.subheader("测算最优复式")
    col1, col2 = st.columns(2)
    with col1:
        n_red = st.selectbox("选几个红球", [7, 8, 9, 10], index=0)
    with col2:
        m_blue = st.selectbox("选几个蓝球", [1, 2, 3, 4, 5, 16], index=0)
        
    if st.button("智能推荐最优复式", type="primary", use_container_width=True):
        best_reds = sorted(score_df.head(n_red)["号码"].tolist())
        best_blues = sorted(blue_score_df.head(m_blue)["蓝球"].tolist())
        
        top6 = best_reds[:6]
        gs, checks = get_group_score(top6, dict(score_df[["号码", "综合分"]].values.tolist()))
        
        bets = comb(n_red, 6) * m_blue
        amount = bets * PER_BET
        
        st.info(f"**红球：** {' '.join(f'{x:02d}' for x in best_reds)}")
        st.info(f"**蓝球：** {' '.join(f'{x:02d}' for x in best_blues)}")
        
        st.markdown(f"### 方案评分：{gs}")
        st.metric("注数", f"{bets} 注")
        st.metric("金额", f"¥{amount}")

with tab3:
    st.subheader("红球近期热度分布")
    # 提取近 50 期的频次
    red_freq = {i: 0 for i in range(1, 34)}
    for draw in df["reds"].tolist()[-50:]:
        for r in draw: red_freq[r] += 1
    
    # 渲染热力图样式
    def color_red_cell(val):
        # 根据频次计算颜色深浅
        intensity = val / max(red_freq.values())
        color = mcolors.to_hex(plt.cm.Reds(intensity * 0.8 + 0.2)) 
        return f'background-color: {color}'
    
    red_heatmap_df = pd.DataFrame([red_freq])
    red_heatmap_df.columns = [f"{i:02d}" for i in range(1, 34)]
    
    styled_reds = red_heatmap_df.style.applymap(color_red_cell).format("{:.0f}")
    st.dataframe(styled_reds, use_container_width=True, height=100)
    st.caption("颜色越深，代表近 50 期出现频次越高（热号）；越浅代表频次越低（冷号）。")

    st.subheader("蓝球近期热度分布")
    blue_freq = {i: 0 for i in range(1, 17)}
    for b in df["blue"].tolist()[-50:]:
        blue_freq[b] += 1
        
    def color_blue_cell(val):
        intensity = val / max(blue_freq.values())
        color = mcolors.to_hex(plt.cm.Blues(intensity * 0.8 + 0.2))
        return f'background-color: {color}'
        
    blue_heatmap_df = pd.DataFrame([blue_freq])
    blue_heatmap_df.columns = [f"{i:02d}" for i in range(1, 17)]
    
    styled_blues = blue_heatmap_df.style.applymap(color_blue_cell).format("{:.0f}")
    st.dataframe(styled_blues, use_container_width=True, height=100)
    st.caption("颜色越深，代表近 50 期出现频次越高。蓝球复式选 1-2 个热号 + 1 个冷号往往有奇效。")

st.caption("免责声明：计算基于历史统计结构，不保证中奖，理性购彩。")
