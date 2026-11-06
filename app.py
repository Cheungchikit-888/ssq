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
st.caption("数据级过滤废号 + 质数/区间/连号/重号多维度科学评分。")

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
        comp = 0.45*f_n[num] + 0.30*m_n[num] + 0.20*c_n[num] + 0.05*r_n[num]
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
    
    primes = {2,3,5,7,11,13,17,19,23,29,31}
    prime_count = sum(1 for x in reds if x in primes)
    
    last_draw = set(df["reds"].tolist()[-1])
    repeat = len(set(reds) & last_draw)

    checks = [
        ("和值", s, 80 <= s <= 120, f"和值{s} {'在' if 80<=s<=120 else '偏离'}80-120区间"),
        ("奇偶", f"{odd}:{6-odd}", 2 <= odd <= 4, f"奇偶{odd}:{6-odd}"),
        ("三区", f"{zones[0]}-{zones[1]}-{zones[2]}", min(zones) > 0, "三区" + ("不断档" if min(zones)>0 else "断档")),
        ("跨度", span, 15 <= span <= 32, f"跨度{span}"),
        ("连号", "有" if consecutive else "无", True, f"{'含' if consecutive else '无'}连号"),
        ("质数", prime_count, 2 <= prime_count <= 4, f"质数{prime_count}个"),
        ("重号", repeat, repeat <= 2, f"重号{repeat}个"),
    ]
    passed = sum(1 for _, _, ok, _ in checks if ok)
    return checks, passed, zones, consecutive, repeat

def get_group_score(reds, score_map):
    checks, passed, zones, consecutive, repeat = structure_analysis(reds)
    
    struct_score = 100 - (7 - passed) * 15
    if passed == 7: struct_score += 5
    
    if zones == [2, 2, 2]: struct_score += 5 
    if consecutive: struct_score += 5         
    
    pool_score = sum(score_map[r] for r in reds) / 6
    total_score = round(struct_score * 0.6 + pool_score * 0.4, 1)
    
    if repeat == 3:
        total_score -= 5
    elif repeat >= 4:
        total_score -= 15
        
    return min(max(total_score, 0), 99.9), checks

def get_rating(score):
    if score >= 85: return "S级 · 极优解 🌟"
    if score >= 78: return "A级 · 优等解 ⭐"
    if score >= 70: return "B级 · 良好解 ✅"
    return "C级 · 普通解"

# ==================== 单式最优解测算 ====================
@st.cache_data(ttl=3600)
def calculate_best_singles(score_list, n_picks=5):
    score_map = dict(score_list)
    random.seed(42) 
    
    candidates = set()
    while len(candidates) < 60000:
        reds = tuple(sorted(random.sample(range(1, 34), 6)))
        checks, passed, _, _, _ = structure_analysis(reds)
        if passed >= 6:  
            candidates.add(reds)
            
    scored_cands = []
    for cand in candidates:
        sc, _ = get_group_score(cand, score_map)
        scored_cands.append((sc, list(cand)))
            
    scored_cands.sort(key=lambda x: x[0], reverse=True)
    
    final_picks = []
    seen_reds = []
    blue_idx = 0
    
    for base_sc, cand in scored_cands:
        if len(final_picks) >= n_picks: break
        
        valid = True
        overlap_penalty = 0
        
        for selected in seen_reds:
            overlap_count = len(set(cand) & set(selected))
            if overlap_count >= 4:
                valid = False 
                break
            elif overlap_count == 3:
                overlap_penalty += 15
            elif overlap_count == 2:
                overlap_penalty += 5 
        
        if valid:
            final_sc = round(base_sc - overlap_penalty, 1)
            blue = int(blue_score_df.iloc[blue_idx % 16]["蓝球"])
            final_picks.append((final_sc, cand, blue, overlap_penalty))
            seen_reds.append(cand)
            blue_idx += 1
            
    return final_picks

# ==================== 界面 ====================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 最优单式", "📦 最优复式", "🔥 冷热透视", "📜 历史开奖"])

with tab1:
    st.subheader("测算最优单式")
    st.caption("已启用【高级规则过滤】：和值收窄 + 质数/三区/连号综合加分。")
    n_picks = st.slider("测算前几组", 1, 15, 5)
    
    if st.button("一键计算最优组合", type="primary", use_container_width=True):
        with st.spinner("正在测算最优组合..."):
            best_singles = calculate_best_singles(tuple(score_df[["号码", "综合分"]].values.tolist()), n_picks)
            
            for i, (gs, reds, blue, penalty) in enumerate(best_singles, 1):
                checks, _, _, _, _ = structure_analysis(reds)
                rating = get_rating(gs)
                
                title = f"第{i}名 ｜ {rating} ｜ 评分 {gs} ｜ 红球 {' '.join(f'{x:02d}' for x in reds)} ｜ 蓝球 {blue:02d}"
                if penalty > 0:
                    title += f" (分散扣分-{penalty})"
                
                with st.expander(title, expanded=(i==1)):
                    st.write("**多维度结构分析：**")
                    for name, val, ok, desc in checks:
                        icon = "✅" if ok else "⚠️"
                        st.write(f"- {icon} {name}：{desc}")

with tab2:
    st.subheader("测算最优复式")
    col1, col2 = st.columns(2)
    with col1:
        n_red = st.selectbox("选几个红球", [7, 8, 9, 10, 11, 12], index=0)
    with col2:
        m_blue = st.selectbox("选几个蓝球", [1, 2, 3, 4, 5, 16], index=0)
        
    if st.button("智能推荐最优复式", type="primary", use_container_width=True):
        best_reds = sorted(score_df.head(n_red)["号码"].tolist())
        best_blues = sorted(blue_score_df.head(m_blue)["蓝球"].tolist())
        
        top6 = best_reds[:6]
        gs, checks = get_group_score(top6, dict(score_df[["号码", "综合分"]].values.tolist()))
        
        bets = comb(n_red, 6) * m_blue
        amount = bets * PER_BET
        
        st.info(f"**推荐红球（评分前{n_red}）：** {' '.join(f'{x:02d}' for x in best_reds)}")
        st.info(f"**推荐蓝球（热度前{m_blue}）：** {' '.join(f'{x:02d}' for x in best_blues)}")
        
        st.markdown(f"### 方案评分：{gs} （以红球前6个测算）")
        st.metric("组合注数", f"{bets} 注")
        st.metric("投注金额", f"¥{amount}")

with tab3:
    st.subheader("🔥 红球冷热温透视")
    
    red_freq = {i: 0 for i in range(1, 34)}
    for draw in df["reds"].tolist()[-50:]:
        for r in draw: red_freq[r] += 1
        
    red_miss = {}
    for num in range(1, 34):
        cur = 0
        for d in reversed(df["reds"].tolist()):
            if num in d: break
            cur += 1
        red_miss[num] = cur

    def color_red_cell(val):
        intensity = val / max(red_freq.values())
        return f'background-color: {mcolors.to_hex(plt.cm.Reds(intensity * 0.8 + 0.2))}'
    
    red_heatmap_df = pd.DataFrame([red_freq])
    red_heatmap_df.columns = [f"{i:02d}" for i in range(1, 34)]
    st.dataframe(red_heatmap_df.style.map(color_red_cell).format("{:.0f}"), use_container_width=True, height=100)

    red_chart_df = pd.DataFrame({
        "号码": [f"{i:02d}" for i in range(1, 34)],
        "出现频次": [red_freq[i] for i in range(1, 34)]
    }).set_index("号码")
    st.bar_chart(red_chart_df, height=200)

    sorted_reds_hot = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    sorted_reds_cold = sorted(red_miss.items(), key=lambda x: x[1], reverse=True)[:5]
    
    st.write("📈 **近期最活跃热号（前5）：**")
    st.success(" | ".join([f"{n:02d}号 🔥(出{c}次)" for n, c in sorted_reds_hot]))
    
    st.write("🧊 **近期极冷号（遗漏最久前5）：**")
    st.info(" | ".join([f"{n:02d}号 ❄️(漏{c}期)" for n, c in sorted_reds_cold]))

    st.divider()

    st.subheader("🔥 蓝球冷热温透视")
    blue_freq = {i: 0 for i in range(1, 17)}
    for b in df["blue"].tolist()[-50:]:
        blue_freq[b] += 1
        
    blue_miss = {}
    for num in range(1, 17):
        cur = 0
        for d in reversed(df["blue"].tolist()):
            if d == num: break
            cur += 1
        blue_miss[num] = cur

    def color_blue_cell(val):
        intensity = val / max(blue_freq.values())
        return f'background-color: {mcolors.to_hex(plt.cm.Blues(intensity * 0.8 + 0.2))}'
        
    blue_heatmap_df = pd.DataFrame([blue_freq])
    blue_heatmap_df.columns = [f"{i:02d}" for i in range(1, 17)]
    st.dataframe(blue_heatmap_df.style.map(color_blue_cell).format("{:.0f}"), use_container_width=True, height=100)

    blue_chart_df = pd.DataFrame({
        "蓝球": [f"{i:02d}" for i in range(1, 17)],
        "出现频次": [blue_freq[i] for i in range(1, 17)]
    }).set_index("蓝球")
    st.bar_chart(blue_chart_df, height=200)

    sorted_blues_hot = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)[:3]
    sorted_blues_cold = sorted(blue_miss.items(), key=lambda x: x[1], reverse=True)[:3]
    
    st.write("📈 **近期最活跃蓝球（前3）：**")
    st.success(" | ".join([f"{n:02d}号 🔥(出{c}次)" for n, c in sorted_blues_hot]))
    
    st.write("🧊 **近期极冷蓝球（遗漏最久前3）：**")
    st.info(" | ".join([f"{n:02d}号 ❄️(漏{c}期)" for n, c in sorted_blues_cold]))
    st.caption("💡 蓝球实战建议：采用“2热+1冷”组合买复式，兼顾热号惯性和冷门爆发。")

with tab4:
    st.subheader("历史开奖记录")
    show_n = st.slider("显示最近期数", 10, 300, 50)
    hist_view = df.tail(show_n).iloc[::-1].copy()
    hist_view["红球"] = hist_view["reds"].apply(lambda x: " ".join(f"{i:02d}" for i in x))
    hist_view["蓝球"] = hist_view["blue"].apply(lambda x: f"{x:02d}")
    st.dataframe(hist_view[["issue", "红球", "蓝球"]].rename(columns={"issue": "期号"}), use_container_width=True, height=400)

st.caption("免责声明：计算基于历史统计结构，不保证中奖，理性购彩。")
