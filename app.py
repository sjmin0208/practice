import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="서울시 수질 분석 대시보드",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 커스텀 CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary:    #0d1117;
        --bg-secondary:  #161b22;
        --bg-card:       #1c2230;
        --bg-card-hover: #212836;
        --border:        #2a3547;
        --border-bright: #3d5a8a;
        --accent-1:      #3b82f6;
        --accent-2:      #60a5fa;
        --accent-3:      #93c5fd;
        --accent-glow:   rgba(59,130,246,0.25);
        --text-primary:  #e8edf5;
        --text-secondary:#8b9ab8;
        --text-muted:    #4e6080;
        --red:           #f87171;
        --green:         #34d399;
        --yellow:        #fbbf24;
        --font-main:     'Pretendard', 'Noto Sans KR', sans-serif;
        --font-mono:     'DM Mono', monospace;
    }

    html, body, [class*="css"] {
        font-family: var(--font-main);
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    .stApp {
        background-color: var(--bg-primary);
    }

    /* ── 사이드바 ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] .stMarkdown th,
    section[data-testid="stSidebar"] .stMarkdown td {
        color: var(--text-secondary) !important;
        border-color: var(--border) !important;
        font-size: 0.78rem !important;
    }

    /* ── 멀티셀렉트 / 셀렉트박스 다크 ── */
    .stMultiSelect > div > div,
    .stSelectbox > div > div {
        background-color: var(--bg-card) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
    }
    .stTextInput > div > div {
        background-color: var(--bg-card) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
    }

    /* ── 데이터프레임 ── */
    .stDataFrame { background-color: var(--bg-card) !important; }
    .stDataFrame thead th {
        background-color: var(--bg-secondary) !important;
        color: var(--accent-2) !important;
        font-weight: 700 !important;
        border-bottom: 1px solid var(--border-bright) !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }
    .stDataFrame tbody tr:hover { background-color: var(--bg-card-hover) !important; }

    /* ── KPI 카드 ── */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-1), transparent);
    }
    .kpi-card:hover {
        border-color: var(--border-bright);
        box-shadow: 0 0 24px var(--accent-glow);
    }
    .kpi-card .value {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
        font-family: var(--font-mono);
    }
    .kpi-card .label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 6px;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .kpi-card .unit {
        font-size: 0.65rem;
        color: var(--accent-1);
        margin-top: 3px;
        font-family: var(--font-mono);
    }

    /* ── 섹션 타이틀 ── */
    .section-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--text-secondary);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0 0 12px 0;
        margin: 28px 0 16px;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-title::before {
        content: '';
        display: inline-block;
        width: 3px;
        height: 14px;
        background: var(--accent-1);
        border-radius: 2px;
    }

    /* ── 정보 박스 ── */
    .info-box {
        background: rgba(59,130,246,0.06);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.83rem;
        color: var(--text-secondary);
        border: 1px solid rgba(59,130,246,0.2);
        margin-bottom: 20px;
        line-height: 1.7;
    }
    .info-box b { color: var(--accent-2); font-weight: 600; }
    .warn-box {
        background: rgba(251,191,36,0.06);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.83rem;
        color: #c9a84c;
        border: 1px solid rgba(251,191,36,0.2);
        margin-bottom: 20px;
        line-height: 1.7;
    }
    .warn-box b { color: var(--yellow); font-weight: 600; }

    /* ── 네비게이션 탭 버튼 ── */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 12px 8px;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-secondary);
        background: var(--bg-card);
        border: 1px solid var(--border);
        transition: all 0.2s ease;
        letter-spacing: 0.02em;
        font-family: var(--font-main);
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: var(--bg-card-hover) !important;
        border-color: var(--accent-1) !important;
        color: var(--accent-2) !important;
        box-shadow: 0 0 16px var(--accent-glow) !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:focus,
    div[data-testid="stHorizontalBlock"] .stButton > button:active {
        background: rgba(59,130,246,0.12) !important;
        color: var(--accent-2) !important;
        border-color: var(--accent-1) !important;
    }

    /* ── 정책 카드 ── */
    .policy-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px 18px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .policy-card::after {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 12px;
        background: radial-gradient(ellipse at top, rgba(59,130,246,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .policy-card .pval   {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--accent-2);
        font-family: var(--font-mono);
        letter-spacing: -0.02em;
    }
    .policy-card .ptitle { font-size: 0.88rem; color: var(--text-primary); margin-top: 8px; font-weight: 600; }
    .policy-card .pdesc  { font-size: 0.76rem; color: var(--text-secondary); margin-top: 6px; font-family: var(--font-mono); }
    .policy-card .pstat  {
        font-size: 0.72rem; color: var(--green); margin-top: 12px;
        background: rgba(52,211,153,0.08);
        border-radius: 20px;
        padding: 4px 12px; display: inline-block;
        border: 1px solid rgba(52,211,153,0.25);
    }

    /* ── 현재 화면 배지 ── */
    .active-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59,130,246,0.1);
        color: var(--accent-2);
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 6px;
        padding: 5px 14px;
        border: 1px solid rgba(59,130,246,0.25);
        margin-bottom: 6px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ── 다운로드 버튼 ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        letter-spacing: 0.02em !important;
    }
    .stDownloadButton > button:hover {
        box-shadow: 0 0 20px var(--accent-glow) !important;
    }

    /* ── 헤더 ── */
    .dash-header {
        padding: 8px 0 24px;
    }
    .dash-header h1 {
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        margin: 0;
    }
    .dash-header .subtitle {
        font-size: 0.8rem;
        color: var(--text-muted);
        font-family: var(--font-mono);
        letter-spacing: 0.06em;
        margin-top: 4px;
    }
    .dash-header .dot {
        display: inline-block;
        width: 7px; height: 7px;
        background: var(--green);
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ── 구분선 ── */
    hr { border-color: var(--border) !important; margin: 16px 0 !important; }

    /* ── 푸터 ── */
    .footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.72rem;
        padding: 20px 0 8px;
        border-top: 1px solid var(--border);
        margin-top: 16px;
        font-family: var(--font-mono);
        letter-spacing: 0.04em;
    }

    /* ── 사이드바 브랜드 ── */
    .sidebar-brand {
        padding: 4px 0 16px;
    }
    .sidebar-brand .logo-text {
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }
    .sidebar-brand .logo-sub {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-family: var(--font-mono);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* ── 태그 칩 ── */
    .tag-chip {
        display: inline-block;
        background: rgba(59,130,246,0.1);
        color: var(--accent-2);
        font-size: 0.68rem;
        font-family: var(--font-mono);
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(59,130,246,0.2);
        margin: 1px;
    }

    /* ── 순위 테이블 라벨 ── */
    .rank-label-top {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--red);
        margin-bottom: 6px;
    }
    .rank-label-bot {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--green);
        margin-bottom: 6px;
    }

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# 데이터 생성
# ══════════════════════════════════════════════════
@st.cache_data
def load_data():
    rng = np.random.default_rng(42)
    districts = [
        "강남구","강동구","강북구","강서구","관악구","광진구","구로구","금천구",
        "노원구","도봉구","동대문구","동작구","마포구","서대문구","서초구",
        "성동구","성북구","송파구","양천구","영등포구","용산구","은평구",
        "종로구","중구","중랑구"
    ]
    turbidity_mean = {
        "강남구":0.073,"강동구":0.049,"강북구":0.049,"강서구":0.059,"관악구":0.050,
        "광진구":0.060,"구로구":0.052,"금천구":0.049,"노원구":0.054,"도봉구":0.049,
        "동대문구":0.062,"동작구":0.050,"마포구":0.051,"서대문구":0.053,"서초구":0.068,
        "성동구":0.059,"성북구":0.049,"송파구":0.049,"양천구":0.053,"영등포구":0.052,
        "용산구":0.049,"은평구":0.053,"종로구":0.052,"중구":0.050,"중랑구":0.069,
    }
    chlorine_mean = {
        "강남구":0.31,"강동구":0.25,"강북구":0.22,"강서구":0.30,"관악구":0.28,
        "광진구":0.29,"구로구":0.30,"금천구":0.27,"노원구":0.28,"도봉구":0.26,
        "동대문구":0.28,"동작구":0.28,"마포구":0.27,"서대문구":0.27,"서초구":0.32,
        "성동구":0.29,"성북구":0.27,"송파구":0.28,"양천구":0.27,"영등포구":0.28,
        "용산구":0.29,"은평구":0.27,"종로구":0.28,"중구":0.28,"중랑구":0.31,
    }
    rows = []
    for district in districts:
        n = rng.integers(60, 130)
        for _ in range(n):
            hour = rng.choice([5, 6, 7, 8, 9, 10])
            temp_base = 8.2 + (hour - 5) * 0.15
            rows.append({
                "구명":     district,
                "측정시각": hour,
                "전기전도도": round(float(rng.normal(277.3, 17.8)), 1),
                "pH":         round(float(rng.normal(7.23, 0.37)), 2),
                "잔류염소":   round(float(rng.normal(chlorine_mean[district], 0.04)), 3),
                "탁도":       round(float(rng.normal(turbidity_mean[district], 0.008)), 4),
                "수온":       round(float(rng.normal(temp_base, 0.5)), 1),
                "수은농도":   round(float(rng.normal(0.00055, 0.00015)), 6),
            })
    df = pd.DataFrame(rows)
    df["탁도"]    = df["탁도"].clip(0.01, 0.78)
    df["잔류염소"] = df["잔류염소"].clip(0.05, 0.75)
    df["pH"]      = df["pH"].clip(5.8, 8.5)
    df["수은농도"] = df["수은농도"].clip(0.0001, 0.001)
    return df


@st.cache_data
def load_policy_data():
    before = {
        "구명":     ["노원구","강남구","송파구","구로구","서초구","중랑구","마포구","강서구","도봉구","성북구"],
        "탁도":     [0.078,0.108,0.138,0.121,0.067,0.127,0.075,0.094,0.109,0.114],
        "잔류염소": [0.22,0.26,0.28,0.30,0.21,0.35,0.24,0.27,0.23,0.25],
        "수은농도": [0.0008,0.0007,0.0009,0.0008,0.0006,0.0009,0.0007,0.0008,0.0007,0.0007],
    }
    after = {
        "구명":     ["노원구","강남구","송파구","구로구","서초구","중랑구","마포구","강서구","도봉구","성북구"],
        "탁도":     [0.046,0.064,0.045,0.067,0.042,0.042,0.066,0.069,0.059,0.066],
        "잔류염소": [0.32,0.35,0.28,0.33,0.25,0.38,0.30,0.32,0.27,0.29],
        "수은농도": [0.0003,0.0003,0.0002,0.0004,0.0003,0.0004,0.0003,0.0003,0.0002,0.0003],
    }
    df_b = pd.DataFrame(before); df_b["기간"] = "정책 이전(2023)"
    df_a = pd.DataFrame(after);  df_a["기간"] = "정책 이후(2025)"
    return pd.concat([df_b, df_a], ignore_index=True)


df        = load_data()
df_policy = load_policy_data()

# ── 다크 테마 플롯 공통 설정 ──
PLOT_BASE = dict(
    plot_bgcolor="#1c2230",
    paper_bgcolor="#1c2230",
    font=dict(family="Pretendard, Noto Sans KR, sans-serif", color="#8b9ab8", size=11),
    margin=dict(t=50, b=40, l=50, r=20),
)
GRID_COLOR  = "#2a3547"
ACCENT_BLUE = ["#1d4ed8","#2563eb","#3b82f6","#60a5fa","#93c5fd","#bfdbfe"]


# ══════════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="logo-text">💧 수질 분석</div>'
        '<div class="logo-sub">Seoul Water Quality · 2026</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown(
        '<span class="tag-chip">📅 2026.03.19</span> '
        '<span class="tag-chip">05~10시</span><br>'
        '<span class="tag-chip">25개 자치구</span> '
        '<span class="tag-chip">2,535건</span>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**데이터 필터**")
    sel_districts = st.multiselect(
        "자치구 선택", sorted(df["구명"].unique()), default=[]
    )
    sel_hours = st.multiselect(
        "측정 시각", [5, 6, 7, 8, 9, 10],
        default=[5, 6, 7, 8, 9, 10],
        format_func=lambda x: f"{x}시"
    )
    sel_metric = st.selectbox(
        "주요 지표",
        ["탁도", "잔류염소", "pH", "전기전도도", "수온", "수은농도"]
    )

    st.markdown("---")
    st.markdown("**먹는물 수질 기준** *(환경부)*")
    st.markdown("""
| 항목 | 기준 | 단위 |
|------|------|------|
| pH | 5.8 ~ 8.5 | — |
| 탁도 | ≤ 0.5 | NTU |
| 잔류염소 | 0.1 ~ 4.0 | mg/L |
| 수은 | ≤ 0.001 | mg/L |
""")


# ══════════════════════════════════════════════════
# 필터 적용
# ══════════════════════════════════════════════════
df_f = df.copy()
if sel_districts: df_f = df_f[df_f["구명"].isin(sel_districts)]
if sel_hours:     df_f = df_f[df_f["측정시각"].isin(sel_hours)]


# ══════════════════════════════════════════════════
# 헤더
# ══════════════════════════════════════════════════
st.markdown(
    '<div class="dash-header">'
    '<h1>서울특별시 수질 분석 대시보드</h1>'
    '<div class="subtitle"><span class="dot"></span>LIVE · Seoul Water Quality Intelligence Platform · 2026.03.19</div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-box">'
    '📌 <b>서울시 수질오염 현황 분석 및 정책 실효성 평가 보고서</b>(2026.03.19) 기반 대시보드 · '
    '분석 항목: pH, 탁도, 잔류염소, 전기전도도, 수온, 수은농도'
    '</div>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════
# KPI 카드
# ══════════════════════════════════════════════════
kpi_cols = st.columns(5)
kpis = [
    ("2,535",                              "총 측정 건수",  "건"),
    (f"{df_f['탁도'].mean():.3f}",         "평균 탁도",     "NTU"),
    (f"{df_f['잔류염소'].mean():.3f}",     "평균 잔류염소", "mg/L"),
    (f"{df_f['pH'].mean():.2f}",           "평균 pH",       "—"),
    (f"{df_f['수온'].mean():.1f}",         "평균 수온",     "℃"),
]
for col, (val, label, unit) in zip(kpi_cols, kpis):
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="value">{val}</div>'
        f'<div class="label">{label}</div>'
        f'<div class="unit">{unit}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# 네비게이션 버튼
# ══════════════════════════════════════════════════
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "지역별 분포"

nav_items = [
    ("💧  지역별 분포",     "지역별 분포"),
    ("🌊  시간별 변화",     "시간별 변화"),
    ("🏛  정책 전후 비교",  "정책 전후 비교"),
    ("🗂  원본 데이터",     "원본 데이터"),
]
nav_cols = st.columns(4)
for col, (label, key) in zip(nav_cols, nav_items):
    with col:
        if st.button(label, key=f"nav_{key}"):
            st.session_state.active_tab = key

active = st.session_state.active_tab

st.markdown(
    f'<div style="margin: 14px 0 0 2px;">'
    f'<span class="active-badge">▶ {active}</span>'
    f'</div>',
    unsafe_allow_html=True
)
st.markdown("---")


# ══════════════════════════════════════════════════
# 화면 1 : 지역별 분포
# ══════════════════════════════════════════════════
if active == "지역별 분포":

    dist_agg = (
        df_f.groupby("구명")[sel_metric]
        .agg(["mean", "std", "count"]).reset_index()
        .rename(columns={"mean": "평균", "std": "표준편차", "count": "건수"})
        .sort_values("평균", ascending=False)
    )
    q75 = dist_agg["평균"].quantile(0.75)
    q25 = dist_agg["평균"].quantile(0.25)
    bar_colors = [
        "#f87171" if v > q75 else
        "#34d399" if v < q25 else
        "#3b82f6"
        for v in dist_agg["평균"]
    ]

    col_a, col_b = st.columns([3, 1])

    with col_a:
        st.markdown(f'<div class="section-title">자치구별 {sel_metric} 평균 비교</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=dist_agg["구명"],
            y=dist_agg["평균"],
            marker_color=bar_colors,
            marker_line_color="rgba(0,0,0,0)",
            error_y=dict(type="data", array=dist_agg["표준편차"],
                         visible=True, color="#3d5a8a", thickness=1.5),
            hovertemplate="<b>%{x}</b><br>평균: %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(
            height=380,
            xaxis=dict(tickangle=-40, gridcolor=GRID_COLOR, title="", tickfont=dict(size=10)),
            yaxis=dict(gridcolor=GRID_COLOR, title=sel_metric, tickfont=dict(size=10)),
            **PLOT_BASE
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown(f'<div class="section-title">구간별 순위</div>', unsafe_allow_html=True)
        st.markdown('<div class="rank-label-top">▲ 상위 5 · 주의</div>', unsafe_allow_html=True)
        top5 = dist_agg.head(5)[["구명", "평균"]].copy()
        top5["평균"] = top5["평균"].round(4)
        st.dataframe(top5.set_index("구명"), use_container_width=True)

        st.markdown('<div class="rank-label-bot" style="margin-top:20px;">▼ 하위 5 · 양호</div>', unsafe_allow_html=True)
        bot5 = dist_agg.tail(5)[["구명", "평균"]].copy()
        bot5["평균"] = bot5["평균"].round(4)
        st.dataframe(bot5.set_index("구명"), use_container_width=True)

    st.markdown('<div class="section-title">항목별 분포 — Box Plot</div>', unsafe_allow_html=True)
    fig2 = make_subplots(rows=1, cols=4,
                         subplot_titles=["탁도 (NTU)", "잔류염소 (mg/L)", "pH", "수온 (℃)"])
    fill_colors = [
        "rgba(248,113,113,0.12)",
        "rgba(59,130,246,0.12)",
        "rgba(52,211,153,0.12)",
        "rgba(251,191,36,0.12)",
    ]
    line_colors = ["#f87171", "#3b82f6", "#34d399", "#fbbf24"]
    for i, (m, lc, fc) in enumerate(zip(["탁도", "잔류염소", "pH", "수온"], line_colors, fill_colors), 1):
        fig2.add_trace(
            go.Box(y=df_f[m], name=m, marker_color=lc,
                   line_color=lc, fillcolor=fc, showlegend=False,
                   boxpoints="outliers", marker_size=3),
            row=1, col=i
        )
    fig2.update_layout(height=300, **PLOT_BASE)
    fig2.update_xaxes(gridcolor=GRID_COLOR, showticklabels=False)
    fig2.update_yaxes(gridcolor=GRID_COLOR, tickfont=dict(size=10))
    for ann in fig2.layout.annotations:
        ann.font.color = "#8b9ab8"
        ann.font.size  = 11
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════
# 화면 2 : 시간별 변화
# ══════════════════════════════════════════════════
elif active == "시간별 변화":

    hourly = (
        df_f.groupby("측정시각")[["잔류염소", "탁도", "pH", "수온"]]
        .mean().reset_index()
    )
    x_vals = hourly["측정시각"].astype(str) + "시"

    st.markdown('<div class="section-title">시간대별 수질 항목 추이</div>', unsafe_allow_html=True)

    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=["잔류염소 (mg/L)", "탁도 (NTU)", "pH", "수온 (℃)"],
        vertical_spacing=0.20, horizontal_spacing=0.10
    )
    metric_colors = {
        "잔류염소": ("#3b82f6", "rgba(59,130,246,0.10)"),
        "탁도":     ("#f87171", "rgba(248,113,113,0.10)"),
        "pH":       ("#34d399", "rgba(52,211,153,0.10)"),
        "수온":     ("#fbbf24", "rgba(251,191,36,0.10)"),
    }
    for (m, r, c) in [("잔류염소", 1, 1), ("탁도", 1, 2), ("pH", 2, 1), ("수온", 2, 2)]:
        lc, fc = metric_colors[m]
        fig3.add_trace(go.Scatter(
            x=x_vals, y=hourly[m],
            mode="lines+markers", name=m,
            line=dict(color=lc, width=2.5),
            marker=dict(size=8, color=lc,
                        line=dict(color=PLOT_BASE["plot_bgcolor"], width=2)),
            fill="tozeroy",
            fillcolor=fc,
        ), row=r, col=c)

    fig3.update_layout(height=460, showlegend=False, **PLOT_BASE)
    fig3.update_xaxes(gridcolor=GRID_COLOR, tickfont=dict(size=10))
    fig3.update_yaxes(gridcolor=GRID_COLOR, tickfont=dict(size=10))
    for ann in fig3.layout.annotations:
        ann.font.color = "#8b9ab8"
        ann.font.size  = 11
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">시간대별 수치 요약표</div>', unsafe_allow_html=True)
    display_hourly = hourly.rename(columns={"측정시각": "시각"}).copy()
    display_hourly["시각"] = display_hourly["시각"].astype(str) + "시"
    st.dataframe(
        display_hourly.style.format(
            {"잔류염소": "{:.4f}", "탁도": "{:.4f}", "pH": "{:.3f}", "수온": "{:.2f}"}
        ).set_properties(**{"text-align": "center"}),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════
# 화면 3 : 정책 전후 비교
# ══════════════════════════════════════════════════
elif active == "정책 전후 비교":

    st.markdown(
        '<div class="warn-box">'
        '⚠️ 아래 데이터는 2024년 <b>상수도 수질 안전관리 강화 대책</b> 효과 시뮬레이션을 위한 '
        '<b>가상 데이터</b>입니다. 실제 효과 분석을 위해서는 실측 종단 데이터가 필요합니다.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Welch\'s t-검정 결과 요약</div>', unsafe_allow_html=True)

    for col, (title, change, values, stat) in zip(st.columns(3), [
        ("탁도 저감",     "↓ 45.1%", "0.1031 → 0.0566 NTU",    "t = 5.573"),
        ("잔류염소 강화", "↑ 44.7%", "0.2620 → 0.3790 mg/L",   "t = 4.971"),
        ("수은농도 감소", "↓ 60.0%", "0.00075 → 0.00030 mg/L", "t = 3.821"),
    ]):
        col.markdown(
            f'<div class="policy-card">'
            f'<div class="pval">{change}</div>'
            f'<div class="ptitle">{title}</div>'
            f'<div class="pdesc">{values}</div>'
            f'<div class="pstat">{stat} · p &lt; 0.01 ✓</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    pol_metric = st.selectbox("비교 항목 선택", ["탁도", "잔류염소", "수은농도"])

    before_df = df_policy[df_policy["기간"] == "정책 이전(2023)"]
    after_df  = df_policy[df_policy["기간"] == "정책 이후(2025)"]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">정책 전후 그룹 비교</div>', unsafe_allow_html=True)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            name="정책 이전 (2023)",
            x=before_df["구명"], y=before_df[pol_metric],
            marker_color="#3d5a8a",
            marker_line_color="rgba(0,0,0,0)",
        ))
        fig4.add_trace(go.Bar(
            name="정책 이후 (2025)",
            x=after_df["구명"], y=after_df[pol_metric],
            marker_color="#3b82f6",
            marker_line_color="rgba(0,0,0,0)",
        ))
        fig4.update_layout(
            barmode="group", height=360,
            xaxis=dict(tickangle=-30, gridcolor=GRID_COLOR, tickfont=dict(size=9)),
            yaxis=dict(gridcolor=GRID_COLOR, title=pol_metric, tickfont=dict(size=10)),
            legend=dict(orientation="h", y=1.12, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            **PLOT_BASE
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">개선 산점도</div>', unsafe_allow_html=True)
        merged = before_df[["구명", pol_metric]].merge(
            after_df[["구명", pol_metric]], on="구명", suffixes=("_이전", "_이후")
        )
        mn = merged[[f"{pol_metric}_이전", f"{pol_metric}_이후"]].min().min() * 0.93
        mx = merged[[f"{pol_metric}_이전", f"{pol_metric}_이후"]].max().max() * 1.07
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(dash="dot", color="#2a3547", width=1.5),
            name="변화 없음", showlegend=True
        ))
        fig5.add_trace(go.Scatter(
            x=merged[f"{pol_metric}_이전"],
            y=merged[f"{pol_metric}_이후"],
            mode="markers+text",
            text=merged["구명"],
            textposition="top center",
            textfont=dict(size=9, color="#8b9ab8"),
            marker=dict(size=11, color="#3b82f6",
                        line=dict(color=PLOT_BASE["plot_bgcolor"], width=2)),
            hovertemplate="<b>%{text}</b><br>이전: %{x:.5f}<br>이후: %{y:.5f}<extra></extra>",
            name=pol_metric,
        ))
        fig5.update_layout(
            height=360,
            xaxis=dict(title="정책 이전", gridcolor=GRID_COLOR, tickfont=dict(size=10)),
            yaxis=dict(title="정책 이후", gridcolor=GRID_COLOR, tickfont=dict(size=10)),
            legend=dict(orientation="h", y=1.12, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            **PLOT_BASE
        )
        st.caption("※ 대각선 아래 = 개선된 지점")
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════
# 화면 4 : 원본 데이터
# ══════════════════════════════════════════════════
elif active == "원본 데이터":

    col_info, _ = st.columns([1, 3])
    with col_info:
        st.markdown(
            f'<div class="kpi-card" style="max-width:180px;">'
            f'<div class="value">{len(df_f):,}</div>'
            f'<div class="label">현재 조회 건수</div>'
            f'<div class="unit">건</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    search = st.text_input("🔍 자치구명 검색", placeholder="예: 강남구")
    show_df = df_f[df_f["구명"].str.contains(search)] if search else df_f

    st.dataframe(
        show_df.sort_values(["구명", "측정시각"]).reset_index(drop=True),
        use_container_width=True,
        height=460,
    )

    st.download_button(
        label="⬇  CSV 다운로드",
        data=show_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="seoul_water_quality.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════
# 푸터
# ══════════════════════════════════════════════════
st.markdown(
    "<p class='footer'>서울특별시 환경정책과 · POLY Analyst · 2026.03.19 · Seoul Water Quality Intelligence</p>",
    unsafe_allow_html=True
)
