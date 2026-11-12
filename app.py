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
st.caption("动态数据引擎 + 大数定律弹性打分 + 单式网状撒网保本策略。")

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
latest_issue = str(df.iloc[-1]['issue'])
st.success(f"数据源：{data_source} ｜ 最新：{latest_issue} ｜ 刷新：5分钟")

# ==================== 核心引擎：评分与冷热温 ====================
def compute_scores(df):
    reds_list = df["reds"].tolist()
    n = len(reds_list)
    windows = [10, 20, 30, 50]
    
    freq_scores = {}
    true_freq_30 = {}
    
    for num in range(1, 34):
        vals = []
        for w in windows:
            recent = reds_list[-w:] if n >= w else reds_list
            cnt = sum(1 for d in recent if num in d)
            vals.append(cnt / len(recent))
        freq_scores[num] = np.mean(vals)
        
        recent_30 = reds_list[-30:] if n >= 30 else reds_list
        true_freq_30[num] = sum(1 for d in recent_30 if num in d)

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
    prev_draw = reds_list[-2] if n >= 2 else []
    
    reweight = {}
    for num in range(1, 34):
        w = 0.0
        if num in last_draw: w += 1.0
        if num in prev_draw: w += 0.8
        reweight[num] = w

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
        comp = 0.40*f_n[num] + 0.30*m_n[num] + 0.20*c_n[num] + 0.10*r_n[num]
        
        tf = true_freq_30[num]
        if tf >= 6: tag = "🔥热"
        elif tf <= 2: tag = "❄️冷"
        else: tag = "🌤温"
            
        rows.append({
            "号码": num, 
            "综合分": round(comp*100, 1), 
            "近30期频次": tf, 
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
last_blue_draw = df.iloc[-1]["blue"]

# ==================== 弹性结构分析 ====================
def calculate_ac(reds):
    diffs = set()
    for i in range(6):
        for j in range(i+1, 6):
            diffs.add(abs(reds[i] - reds[j]))
    return len(diffs) - 5

def structure_analysis(reds):
    s = sum(reds)
    odd = sum(1 for x in reds if x % 2 == 1)
    r0 = sum(1 for x in reds if x % 3 == 0)
    r1 = sum(1 for x in reds if x % 3 == 1)
    r2 = sum(1 for x in reds if x % 3 == 2)
    route_str = f"{r0}:{r1}:{r2}"

    zones = [0, 0, 0]
    for x in reds:
        if x <= 11: zones[0] += 1
        elif x <= 22: zones[1] += 1
        else: zones[2] += 1
    zone_str = ":".join(map(str, sorted(zones, reverse=True)))
    
    span = max(reds) - min(reds)
    consecutive = any(reds[i+1] - reds[i] == 1 for i in range(5))
    
    tails = [x % 10 for x in reds]
    tail_counts = Counter(tails)
    same_tails = sum(1 for count in tail_counts.values() if count >= 2)
    
    primes = {2,3,5,7,11,13,17,19,23,29,31}
    prime_count = sum(1 for x in reds if x in primes)
    last_draw = set(df["reds"].tolist()[-1])
    repeat = len(set(reds) & last_draw)
    ac_val = calculate_ac(reds)

    checks = [
        ("和值", s, 80 <= s <= 120, f"和值{s}"),
        ("奇偶", f"{odd}:{6-odd}", 2 <= odd <= 4, f"奇偶{odd}:{6-odd}"),
        ("012路", route_str, r0>0 and r1>0 and r2>0, f"012路{route_str}"),
        ("三区", f"{zones[0]}-{zones[1]}-{zones[2]}", min(zones) > 0, "三区不断档" if min(zones)>0 else "三区断档"),
        ("跨度", span, 15 <= span <= 32, f"跨度{span}"),
        ("AC值", ac_val, ac_val >= 6, f"离散度{ac_val}"),
        ("同尾", same_tails, True, f"同尾{same_tails}组"),
        ("质数", prime_count, 2 <= prime_count <= 4, f"质数{prime_count}个"),
        ("重号", repeat, repeat <= 2, f"重号{repeat}个"),
    ]
    passed = sum(1 for _, _, ok, _ in checks if ok)
    return checks, passed, zones, consecutive, repeat, zone_str, same_tails

def get_group_score(reds, score_map):
    checks, passed, zones, consecutive, repeat, zone_str, same_tails = structure_analysis(reds)
    struct_score = 100 - (9 - passed) * 10
    
    if consecutive: struct_score += 5
    if same_tails == 1: struct_score += 3
    elif same_tails >= 2: struct_score += 6
        
    pool_score = sum(score_map[r] for r in reds) / 6
    total_score = round(struct_score * 0.6 + pool_score * 0.4, 1)
    
    if repeat == 3: total_score -= 5
    elif repeat >= 4: total_score -= 15
    
    return min(max(total_score, 0), 99.9), checks, zone_str

def get_rating(score):
    if score >= 85: return "S级 · 极优解 🌟"
    if score >= 78: return "A级 · 优等解 ⭐"
    if score >= 70: return "B级 · 良好解 ✅"
    return "C级 · 普通解"

# ==================== 蓝球智能推荐（修复去重Bug） ====================
def get_blue_reason(blue):
    b_info = blue_score_df[blue_score_df["蓝球"] == blue].iloc[0]
    tag = b_info["冷热标签"]
    miss = b_info["当前遗漏"]
    
    if blue == last_blue_draw: return "重号防守（上期刚开）"
    neighbors = []
    if last_blue_draw > 1: neighbors.append(last_blue_draw - 1)
    if last_blue_draw < 16: neighbors.append(last_blue_draw + 1)
    if blue in neighbors: return f"邻号策略（紧贴上期开奖号{last_blue_draw:02d}）"
        
    if tag == "🔥热": return f"热号延续（出{b_info['热度']}次，保底首选）"
    elif tag == "❄️冷": return f"极冷防守（已遗漏{miss}期，防爆冷）"
    else: return f"温号过渡（遗漏{miss}期）"

def recommend_blues(m_blue):
    hottest = blue_score_df.iloc[0]["蓝球"]
    neighbors = []
    if last_blue_draw > 1: neighbors.append(last_blue_draw - 1)
    if last_blue_draw < 16: neighbors.append(last_blue_draw + 1)
    
    # 核心修复：使用列表去重，避免 hottest 和 neighbor 撞车
    full_pool = []
    
    # 1. 必带的最热号
    full_pool.append(hottest)
    
    # 2. 邻号（只要不在池子里，就加进去）
    for n in neighbors:
        if n not in full_pool:
            full_pool.append(n)
            break # 只拿一个不重复的邻号
            
    # 3. 极冷号
    coldest = blue_score_df.sort_values("当前遗漏", ascending=False).iloc[0]["蓝球"]
    if coldest not in full_pool:
        full_pool.append(coldest)
        
    # 4. 温号
    warm_df = blue_score_df[blue_score_df["冷热标签"] == "🌤温"]
    if len(warm_df) > 0:
        warm = warm_df.iloc[0]["蓝球"]
        if warm not in full_pool:
            full_pool.append(warm)
            
    # 5. 剩下按热度补齐
    for b in blue_score_df["蓝球"].tolist():
        if b not in full_pool:
            full_pool.append(b)
            
    return full_pool[:m_blue], "按热度、邻号、冷号优先级依次分配（已智能去重）"

def generate_detailed_report(reds, blue, score_map):
    checks, passed, zones, consecutive, repeat, zone_str, same_tails = structure_analysis(reds)
    red_details = []
    primes = {2,3,5,7,11,13,17,19,23,29,31}
    last_draw = set(df["reds"].tolist()[-1])
    prev_draw = set(df["reds"].tolist()[-2]) if len(df) >= 2 else set()
    
    for r in reds:
        freq = int(score_df[score_df["号码"] == r]["近30期频次"].values[0])
        miss = int(score_df[score_df["号码"] == r]["当前遗漏"].values[0])
        tag = red_tag_dict[r]
        prime_tag = "(质)" if r in primes else ""
        special_tag = " [上期重号]" if r in last_draw else (" [隔期码]" if r in prev_draw else "")
        red_details.append(f"**{r:02d}**{prime_tag}: 评{score_map[r]}, 出{freq}次, 漏{miss}期 [{tag}]{special_tag}")
        
    blue_detail = f"**{blue:02d}**号: {get_blue_reason(blue)}"

    report = "### 🔍 数据体检报告\n"
    report += "**1. 成分解析：**\n"
    report += "  \n".join(red_details) + "\n"
    report += f"  \n**蓝球：** {blue_detail}\n\n"
    report += "**2. 形态剖析：**\n"
    for name, val, ok, desc in checks:
        icon = "✅" if ok else "⚠️"
        report += f"- {icon} **{name}**: {desc}\n"
    if consecutive: report += "- 🌟 **连号加成**: 包含连号\n"
    if same_tails >= 1: report += f"- 🌟 **同尾加成**: 包含{same_tails}组同尾\n"
    return report

# ==================== 单式最优解测算（动态种子引擎版） ====================
@st.cache_data(ttl=3600)
def calculate_best_singles(score_list, n_picks, current_issue):
    try:
        seed_val = int(current_issue)
    except:
        seed_val = 42
    random.seed(seed_val)
    
    score_map = dict(score_list)
    candidates = set()
    
    attempts = 0
    while len(candidates) < 60000 and attempts < 150000:
        attempts += 1
        reds = tuple(sorted(random.sample(range(1, 34), 6)))
        if calculate_ac(reds) < 5: continue
        checks, passed, _, _, _, _, _ = structure_analysis(reds)
        if passed >= 6:  
            candidates.add(reds)
            
    scored_cands = []
    for cand in candidates:
        sc, _, z_str = get_group_score(cand, score_map)
        scored_cands.append((sc, list(cand), z_str))
            
    scored_cands.sort(key=lambda x: x[0], reverse=True)
    
    final_picks = []
    seen_reds = []
    # 单式获取一个包含所有16个蓝球的完整池子（已去重）
    b_pool, _ = recommend_blues(16) 
    blue_idx = 0
    
    for base_sc, cand, z_str in scored_cands:
        if len(final_picks) >= n_picks: break
        
        valid = True
        for selected in seen_reds:
            overlap_count = len(set(cand) & set(selected))
            if overlap_count >= 2:
                valid = False 
                break
        
        if valid:
            blue = b_pool[blue_idx % len(b_pool)]
            final_picks.append((base_sc, cand, blue, z_str, get_blue_reason(blue)))
            seen_reds.append(cand)
            blue_idx += 1
            
    return final_picks

# ==================== 界面 ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 最优单式", "📦 最优复式", "🔍 号码诊断", "🔥 冷热透视", "📜 历史开奖"])

with tab1:
    st.subheader("测算最优单式（网状撒网模式）")
    st.caption("已启用动态数据引擎：每一期开奖后，底层6万组样本将彻底洗牌重算。")
    n_picks = st.slider("测算前几组", 1, 10, 5)
    
    if st.button("一键计算最优组合", type="primary", use_container_width=True):
        with st.spinner("正在测算本期全新撒网组合..."):
            best_singles = calculate_best_singles(tuple(score_df[["号码", "综合分"]].values.tolist()), n_picks, latest_issue)
            score_map = dict(score_df[["号码", "综合分"]].values.tolist())
            
            for i, (gs, reds, blue, z_str, blue_reason) in enumerate(best_singles, 1):
                rating = get_rating(gs)
                title = f"第{i}注 ｜ {rating} ｜ 评分 {gs} ｜ 形态 {z_str} ｜ 红球 {' '.join(f'{x:02d}' for x in reds)} ｜ 蓝球 {blue:02d}"
                with st.expander(title, expanded=(i==1)):
                    st.markdown(f"💡 **蓝球分配逻辑：** {blue_reason}")
                    st.divider()
                    st.markdown(generate_detailed_report(reds, blue, score_map))

with tab2:
    st.subheader("测算最优复式")
    col1, col2 = st.columns(2)
    with col1:
        n_red = st.selectbox("选几个红球", [7, 8, 9, 10, 11, 12], index=0)
    with col2:
        m_blue = st.selectbox("选几个蓝球", [1, 2, 3, 4, 5, 16], index=0)
        
    if st.button("智能推荐最优复式", type="primary", use_container_width=True):
        best_reds = sorted(score_df.head(n_red)["号码"].tolist())
        best_blues, blue_logic = recommend_blues(m_blue)

        top6 = best_reds[:6]
        score_map = dict(score_df[["号码", "综合分"]].values.tolist())
        gs, checks, z_str = get_group_score(top6, score_map)
        
        bets = comb(n_red, 6) * m_blue
        amount = bets * PER_BET
        
        st.info(f"**推荐红球：** {' '.join(f'{x:02d}' for x in best_reds)}")
        st.info(f"**推荐蓝球：** {' '.join(f'{x:02d}' for x in best_blues)}")
        st.caption(f"💡 **选蓝逻辑**：{blue_logic}")
        
        st.markdown(f"### 方案评分：{gs} （形态 {z_str}）")
        st.metric("组合注数", f"{bets} 注")
        st.metric("投注金额", f"¥{amount}")
        
        st.divider()
        st.markdown("### 🔍 核心红球与蓝球数据解析")
        red_details = []
        for r in best_reds:
            freq = int(score_df[score_df["号码"] == r]["近30期频次"].values[0])
            miss = int(score_df[score_df["号码"] == r]["当前遗漏"].values[0])
            tag = red_tag_dict[r]
            special_tag = " [上期重号]" if r in set(df["reds"].tolist()[-1]) else (" [隔期码]" if len(df) >= 2 and r in set(df["reds"].tolist()[-2]) else "")
            red_details.append(f"**{r:02d}**: 评{score_map[r]}, 出{freq}次, 漏{miss}期 [{tag}]{special_tag}")
        st.markdown("  \n".join(red_details))
        
        blue_details = [f"**蓝球 {b:02d}**: {get_blue_reason(b)}" for b in best_blues]
        st.markdown("  \n".join(blue_details))

with tab3:
    st.subheader("🔍 我的号码深度诊断")
    st.caption("支持单式与复式体检，为你输入的号码进行冷热温与结构评估。")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        user_reds = st.multiselect("选择红球", list(range(1, 34)), default=[1, 7, 12, 18, 24, 30])
    with col2:
        user_blues = st.multiselect("选择蓝球", list(range(1, 17)), default=[7])
        
    if st.button("开始诊断", type="primary", use_container_width=True):
        if len(user_reds) < 6 or len(user_blues) < 1:
            st.error("⚠️ 红球至少需要 6 个，蓝球至少需要 1 个！")
        else:
            user_reds = sorted(user_reds)
            user_blues = sorted(user_blues)
            n_red, m_blue = len(user_reds), len(user_blues)
            
            bets = comb(n_red, 6) * m_blue
            amount = bets * PER_BET
            score_map = dict(score_df[["号码", "综合分"]].values.tolist())
            
            st.info(f"**当前配置：** {n_red}红 + {m_blue}蓝 ｜ **总注数：** {bets} 注 ｜ **总金额：** ¥{amount}")
            
            hot_cnt = sum(1 for x in user_reds if red_tag_dict[x] == "🔥热")
            warm_cnt = sum(1 for x in user_reds if red_tag_dict[x] == "🌤温")
            cold_cnt = sum(1 for x in user_reds if red_tag_dict[x] == "❄️冷")
            
            if n_red == 6 and m_blue == 1:
                gs, checks, z_str = get_group_score(user_reds, score_map)
                rating = get_rating(gs)
                st.success(f"**综合评级：{rating} ｜ 最终得分：{gs} ｜ 形态：{z_str}**")
                
                if hot_cnt >= 5: st.error("🚨 **极端警报**：红球热号过多，容易撞车。")
                elif cold_cnt >= 3: st.warning("⚠️ **阵型警告**：红球冷号过多，极其依赖爆冷。")
                else: st.success("✅ **阵型健康**：冷热温搭配合理。")
                    
                st.divider()
                st.markdown(generate_detailed_report(user_reds, user_blues[0], score_map))
            else:
                core_reds = sorted(score_df[score_df["号码"].isin(user_reds)].head(6)["号码"].tolist())
                core_score, checks, z_str = get_group_score(core_reds, score_map)
                
                breadth_score = core_score
                zones_covered = len(set([1 if x <= 11 else (2 if x <= 22 else 3) for x in user_reds]))
                if zones_covered == 3: breadth_score += 2
                if hot_cnt > 0 and warm_cnt > 0 and cold_cnt > 0: breadth_score += 3
                breadth_score = min(breadth_score, 99.9)
                
                st.success(f"**复式综合评级：{get_rating(breadth_score)} ｜ 综合得分：{round(breadth_score, 1)}**")
                st.divider()
                st.markdown("### 🔍 复式池成分剖析")
                red_details = []
                for r in user_reds:
                    freq = int(score_df[score_df["号码"] == r]["近30期频次"].values[0])
                    miss = int(score_df[score_df["号码"] == r]["当前遗漏"].values[0])
                    tag = red_tag_dict[r]
                    special_tag = " [上期重号]" if r in set(df["reds"].tolist()[-1]) else (" [隔期码]" if len(df) >= 2 and r in set(df["reds"].tolist()[-2]) else "")
                    red_details.append(f"**{r:02d}**: 评{score_map[r]}, 出{freq}次, 漏{miss}期 [{tag}]{special_tag}")
                st.markdown("  \n".join(red_details))
                
                blue_details = [f"**蓝球 {b:02d}**: {get_blue_reason(b)}" for b in user_blues]
                st.markdown("  \n".join(blue_details))

with tab4:
    st.subheader("🔥 红球冷热温透视（近30期滚动）")
    red_freq = {i: 0 for i in range(1, 34)}
    for draw in df["reds"].tolist()[-30:]:
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

    sorted_blues_hot = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)[:3]
    sorted_blues_cold = sorted(blue_miss.items(), key=lambda x: x[1], reverse=True)[:3]
    
    st.write("📈 **近期最活跃蓝球（前3）：**")
    st.success(" | ".join([f"{n:02d}号 🔥(出{c}次)" for n, c in sorted_blues_hot]))
    st.write("🧊 **近期极冷蓝球（遗漏最久前3）：**")
    st.info(" | ".join([f"{n:02d}号 ❄️(漏{c}期)" for n, c in sorted_blues_cold]))

with tab5:
    st.subheader("历史开奖记录")
    show_n = st.slider("显示最近期数", 10, 300, 50)
    hist_view = df.tail(show_n).iloc[::-1].copy()
    hist_view["红球"] = hist_view["reds"].apply(lambda x: " ".join(f"{i:02d}" for i in x))
    hist_view["蓝球"] = hist_view["blue"].apply(lambda x: f"{x:02d}")
    st.dataframe(hist_view[["issue", "红球", "蓝球"]].rename(columns={"issue": "期号"}), use_container_width=True, height=400)

st.caption("免责声明：计算基于历史统计结构，不保证中奖，理性购彩。")
