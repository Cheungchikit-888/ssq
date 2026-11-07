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
st.caption("数据级过滤废号 + 冷热温强制均衡 + 深度数据解析。")

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
            rows.append({"issue": item["code"], "date": item.get("date", ""), "reds": reds, "blue": int(item["blue"])})
        df = pd.DataFrame(rows).sort_values("issue").reset_index(drop=True)
        if len(df) > 50: return df, "中国福彩网（官方）"
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
        if len(df2) > 50: return df2, "500.com（第三方兜底）"
    except Exception:
        pass

    st.warning("实时数据源暂不可用，使用内置示例数据演示。")
    sample = [[1,5,12,20,25,33,7],[3,8,15,22,28,31,12],[2,9,14,19,26,30,5]]
    rows = [{"issue": f"DEMO{i+1:03d}", "date": "", "reds": r[:6], "blue": r[6]} for i, r in enumerate(sample)]
    return pd.DataFrame(rows), "内置示例（非真实数据）"

df, data_source = load_history()
st.success(f"数据源：{data_source} ｜ 最新：{df.iloc[-1]['issue']} ｜ 刷新：5分钟")

# ==================== 评分与冷热温计算 ====================
def compute_scores(df):
    reds_list = df["reds"].tolist()
    n = len(reds_list)
    windows = [10, 20, 30, 50]
    
    freq_scores = {}
    true_freq_50 = {} 
    
    for num in range(1, 34):
        vals = []
        for w in windows:
            recent = reds_list[-w:] if n >= w else reds_list
            cnt = sum(1 for d in recent if num in d)
            vals.append(cnt / len(recent))
        freq_scores[num] = np.mean(vals)
        
        recent_50 = reds_list[-50:] if n >= 50 else reds_list
        true_freq_50[num] = sum(1 for d in recent_50 if num in d)

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
        miss_scores[num] = min(cur / max(avg, 1), 3.0) 

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
        
        tf = true_freq_50[num]
        if tf >= 11: tag = "🔥热"
        elif tf <= 7: tag = "❄️冷"
        else: tag = "🌤温"
            
        rows.append({
            "号码": num, 
            "综合分": round(comp*100, 1), 
            "近50期频次": tf, 
            "当前遗漏": cur_miss_map[num],
            "冷热标签": tag
        })
    return pd.DataFrame(rows).sort_values("综合分", ascending=False).reset_index(drop=True), cur_miss_map

score_df, cur_miss_map = compute_scores(df)
red_tag_dict = dict(zip(score_df["号码"], score_df["冷热标签"]))

blue_counts = Counter(df["blue"].tolist()[-50:])
def get_blue_miss(b):
    cur = 0
    for d in reversed(df["blue"].tolist()):
        if d == b: break
        cur += 1
    return cur

blue_score_df = pd.DataFrame({"蓝球": range(1, 17), "热度": [blue_counts.get(i, 0) for i in range(1, 17)], "当前遗漏": [get_blue_miss(i) for i in range(1, 17)]})
blue_score_df["冷热标签"] = blue_score_df["热度"].apply(lambda x: "🔥热" if x >= 5 else ("❄️冷" if x <= 2 else "🌤温"))
blue_score_df = blue_score_df.sort_values("热度", ascending=False).reset_index(drop=True)

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
    if repeat == 3: total_score -= 5
    elif repeat >= 4: total_score -= 15
    return min(max(total_score, 0), 99.9), checks

def get_rating(score):
    if score >= 85: return "S级 · 极优解 🌟"
    if score >= 78: return "A级 · 优等解 ⭐"
    if score >= 70: return "B级 · 良好解 ✅"
    return "C级 · 普通解"

def generate_detailed_report(reds, blue, score_map):
    checks, passed, zones, consecutive, repeat = structure_analysis(reds)
    red_details = []
    primes = {2,3,5,7,11,13,17,19,23,29,31}
    for r in reds:
        freq = int(score_df[score_df["号码"] == r]["近50期频次"].values[0])
        miss = int(score_df[score_df["号码"] == r]["当前遗漏"].values[0])
        tag = red_tag_dict[r]
        prime_tag = "(质)" if r in primes else ""
        red_details.append(f"**{r:02d}**{prime_tag}: 评{score_map[r]}, 出{freq}次, 漏{miss}期 [{tag}]")
        
    b_info = blue_score_df[blue_score_df["蓝球"] == blue].iloc[0]
    b_tag = b_info["冷热标签"]
    blue_detail = f"**{blue:02d}**号: 出{b_info['热度']}次, 漏{b_info['当前遗漏']}期 [{b_tag}]"

    report = "### 🔍 数据体检报告\n"
    report += "**1. 成分解析：**\n"
    report += "  \n".join(red_details) + "\n"
    report += f"  \n**蓝球：** {blue_detail}\n\n"
    report += "**2. 形态剖析：**\n"
    for name, val, ok, desc in checks:
        icon = "✅" if ok else "⚠️"
        report += f"- {icon} **{name}**: {desc}\n"
    if zones == [2, 2, 2]: report += "- 🌟 **完美区间**: 2:2:2 分布\n"
    if consecutive: report += "- 🌟 **连号加成**: 包含连号\n"
    return report

# ==================== 单式最优解测算 ====================
@st.cache_data(ttl=3600)
def calculate_best_singles(score_list, n_picks=5):
    score_map = dict(score_list)
    random.seed(42) 
    candidates = set()
    
    attempts = 0
    while len(candidates) < 60000 and attempts < 150000:
        attempts += 1
        reds = tuple(sorted(random.sample(range(1, 34), 6)))
        checks, passed, _, _, _ = structure_analysis(reds)
        
        if passed >= 6:  
            hot_cnt = sum(1 for x in reds if red_tag_dict[x] == "🔥热")
            cold_cnt = sum(1 for x in reds if red_tag_dict[x] == "❄️冷")
            if hot_cnt <= 3 and 1 <= cold_cnt <= 2:
                candidates.add(reds)
            
    scored_cands = []
    for cand in candidates:
        sc, _ = get_group_score(cand, score_map)
        scored_cands.append((sc, list(cand)))
            
    scored_cands.sort(key=lambda x: x[0], reverse=True)
    
    final_picks = []
    seen_reds = []
    
    hot_blues = blue_score_df[blue_score_df["冷热标签"] == "🔥热"]["蓝球"].tolist()
    warm_blues = blue_score_df[blue_score_df["冷热标签"] == "🌤温"]["蓝球"].tolist()
    cold_blues = blue_score_df[blue_score_df["冷热标签"] == "❄️冷"]["蓝球"].tolist()
    
    blue_pool = []
    if n_picks <= 2:
        blue_pool = hot_blues.copy()
        if len(blue_pool) < n_picks: blue_pool.extend(warm_blues)
        if len(blue_pool) < n_picks: blue_pool.extend(cold_blues)
    else:
        for i in range(6):
            if i < 2 and len(hot_blues) > 0: blue_pool.append(hot_blues[i % len(hot_blues)])
            if 2 <= i < 4 and len(warm_blues) > 0: blue_pool.append(warm_blues[(i-2) % len(warm_blues)])
            if 4 <= i < 5 and len(cold_blues) > 0: blue_pool.append(cold_blues[(i-4) % len(cold_blues)])
            
    if not blue_pool: blue_pool = [1, 2, 3, 4, 5]
    blue_idx = 0
    
    for base_sc, cand in scored_cands:
        if len(final_picks) >= n_picks: break
        
        valid = True
        for selected in seen_reds:
            overlap_count = len(set(cand) & set(selected))
            if overlap_count >= 3:
                valid = False 
                break
        
        if valid:
            blue = int(blue_pool[blue_idx % len(blue_pool)])
            final_picks.append((base_sc, cand, blue))
            seen_reds.append(cand)
            blue_idx += 1
            
    return final_picks

# ==================== 界面 ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 最优单式", "📦 最优复式", "🔍 号码诊断", "🔥 冷热透视", "📜 历史开奖"])

with tab1:
    st.subheader("测算最优单式")
    st.caption("严格执行【2热+2温+2冷/3热+2温+1冷】均衡法则，规避极端组合。")
    n_picks = st.slider("测算前几组", 1, 15, 5)
    
    if st.button("一键计算最优组合", type="primary", use_container_width=True):
        with st.spinner("正在测算最优组合..."):
            best_singles = calculate_best_singles(tuple(score_df[["号码", "综合分"]].values.tolist()), n_picks)
            score_map = dict(score_df[["号码", "综合分"]].values.tolist())
            
            for i, (gs, reds, blue) in enumerate(best_singles, 1):
                rating = get_rating(gs)
                title = f"第{i}名 ｜ {rating} ｜ 评分 {gs} ｜ 红球 {' '.join(f'{x:02d}' for x in reds)} ｜ 蓝球 {blue:02d}"
                with st.expander(title, expanded=(i==1)):
                    st.markdown(generate_detailed_report(reds, blue, score_map))

with tab2:
    st.subheader("测算最优复式")
    col1, col2 = st.columns(2)
    with col1:
        n_red = st.selectbox("选几个红球", [7, 8, 9, 10, 11, 12], index=0)
    with col2:
        m_blue = st.selectbox("选几个蓝球", [1, 2, 3, 4, 5, 16], index=0)
        
    if st.button("智能推荐最优复式", type="primary", use_container_width=True):
        top_reds = score_df.head(n_red)
        cold_in_top = top_reds[top_reds["冷热标签"] == "❄️冷"]
        
        if len(cold_in_top) == 0:
            all_colds = score_df[score_df["冷热标签"] == "❄️冷"]
            if len(all_colds) > 0:
                best_cold = int(all_colds.iloc[0]["号码"])
                best_reds = sorted(score_df.head(n_red-1)["号码"].tolist() + [best_cold])
            else:
                best_reds = sorted(top_reds["号码"].tolist())
        else:
            best_reds = sorted(top_reds["号码"].tolist())
            
        best_blues = sorted(blue_score_df.head(m_blue)["蓝球"].tolist())
        
        top6 = best_reds[:6]
        score_map = dict(score_df[["号码", "综合分"]].values.tolist())
        gs, checks = get_group_score(top6, score_map)
        
        bets = comb(n_red, 6) * m_blue
        amount = bets * PER_BET
        
        st.info(f"**推荐红球：** {' '.join(f'{x:02d}' for x in best_reds)}")
        st.info(f"**推荐蓝球：** {' '.join(f'{x:02d}' for x in best_blues)}")
        
        st.markdown(f"### 方案评分：{gs} （以核心前6个测算）")
        st.metric("组合注数", f"{bets} 注")
        st.metric("投注金额", f"¥{amount}")
        
        st.divider()
        st.markdown("### 🔍 核心红球与蓝球数据解析")
        red_details = []
        for r in best_reds:
            freq = int(score_df[score_df["号码"] == r]["近50期频次"].values[0])
            miss = int(score_df[score_df["号码"] == r]["当前遗漏"].values[0])
            tag = red_tag_dict[r]
            red_details.append(f"**{r:02d}**: 评{score_map[r]}, 出{freq}次, 漏{miss}期 [{tag}]")
        st.markdown("  \n".join(red_details))
        
        blue_details = []
        for b in best_blues:
            b_info = blue_score_df[blue_score_df["蓝球"] == b].iloc[0]
            b_tag = b_info["冷热标签"]
            blue_details.append(f"**蓝球 {b:02d}**: 出{b_info['热度']}次, 漏{b_info['当前遗漏']}期 [{b_tag}]")
        st.markdown("  \n".join(blue_details))
        st.caption("提示：复式使用的是【稳健阵型】，核心区配热号，边缘区强制布防冷号。")

with tab3:
    st.subheader("🔍 我的号码深度诊断")
    st.caption("输入你的心水号码，用大数据和概率规则为你做一次全面体检。")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        user_reds = st.multiselect("选择 6 个红球", list(range(1, 34)), max_selections=6, help="最多只能选 6 个")
    with col2:
        user_blue = st.selectbox("选择 1 个蓝球", list(range(1, 17)))
        
    if st.button("开始诊断", type="primary", use_container_width=True):
        if len(user_reds) != 6:
            st.error("⚠️ 请务必选择刚好 6 个红球！")
        else:
            user_reds = sorted(user_reds)
            score_map = dict(score_df[["号码", "综合分"]].values.tolist())
            gs, checks = get_group_score(user_reds, score_map)
            rating = get_rating(gs)
            
            # 顶部展示结果
            st.success(f"**综合评级：{rating} ｜ 最终得分：{gs}**")
            st.markdown(f"**诊断号码：** 红球 {' '.join(f'{x:02d}' for x in user_reds)} ｜ 蓝球 {user_blue:02d}")
            
            # 特别警报：冷热均衡检查
            hot_cnt = sum(1 for x in user_reds if red_tag_dict[x] == "🔥热")
            warm_cnt = sum(1 for x in user_reds if red_tag_dict[x] == "🌤温")
            cold_cnt = sum(1 for x in user_reds if red_tag_dict[x] == "❄️冷")
            
            if hot_cnt >= 5:
                st.error("🚨 **极端警报**：你的红球包含 5 个或以上的热号！这极容易变成大众票导致奖金稀释，且容错率极低。")
            elif cold_cnt >= 3:
                st.warning("⚠️ **阵型警告**：你的红球包含 3 个或以上的冷号。防守过度，极其依赖爆冷。")
            elif hot_cnt == 0 or cold_cnt == 0:
                st.info("💡 **阵型提示**：你的号码缺少活跃的热号或防守的极冷号，冷热温形态略显平庸。")
            else:
                st.success("✅ **阵型完美**：你的红球冷热温搭配合理，攻守兼备！")
                
            st.divider()
            
            # 生成详细报告
            report = generate_detailed_report(user_reds, user_blue, score_map)
            st.markdown(report)

with tab4:
    st.subheader("🔥 红球冷热温透视")
    red_freq = {i: 0 for i in range(1, 34)}
    for draw in df["reds"].tolist()[-50:]:
        for r in draw: red_freq[r] += 1
    red_miss = cur_miss_map

    def color_red_cell(val):
        return f'background-color: {mcolors.to_hex(plt.cm.Reds((val / max(red_freq.values())) * 0.8 + 0.2))}'
    
    red_heatmap_df = pd.DataFrame([red_freq])
    red_heatmap_df.columns = [f"{i:02d}" for i in range(1, 34)]
    st.dataframe(red_heatmap_df.style.map(color_red_cell).format("{:.0f}"), use_container_width=True, height=100)

    red_chart_df = pd.DataFrame({"号码": [f"{i:02d}" for i in range(1, 34)], "出现频次": [red_freq[i] for i in range(1, 34)]}).set_index("号码")
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
    blue_miss = {n: get_blue_miss(n) for n in range(1,17)}

    def color_blue_cell(val):
        return f'background-color: {mcolors.to_hex(plt.cm.Blues((val / max(blue_freq.values())) * 0.8 + 0.2))}'
        
    blue_heatmap_df = pd.DataFrame([blue_freq])
    blue_heatmap_df.columns = [f"{i:02d}" for i in range(1, 17)]
    st.dataframe(blue_heatmap_df.style.map(color_blue_cell).format("{:.0f}"), use_container_width=True, height=100)

    blue_chart_df = pd.DataFrame({"蓝球": [f"{i:02d}" for i in range(1, 17)], "出现频次": [blue_freq[i] for i in range(1, 17)]}).set_index("蓝球")
    st.bar_chart(blue_chart_df, height=200)

with tab5:
    st.subheader("历史开奖记录")
    show_n = st.slider("显示最近期数", 10, 300, 50)
    hist_view = df.tail(show_n).iloc[::-1].copy()
    hist_view["红球"] = hist_view["reds"].apply(lambda x: " ".join(f"{i:02d}" for i in x))
    hist_view["蓝球"] = hist_view["blue"].apply(lambda x: f"{x:02d}")
    st.dataframe(hist_view[["issue", "红球", "蓝球"]].rename(columns={"issue": "期号"}), use_container_width=True, height=400)

st.caption("免责声明：计算基于历史统计结构，不保证中奖，理性购彩。")
