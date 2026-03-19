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

# ── 커스텀 CSS (파란색 계열 통일) ────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
        background-color: #f0f4f8;
    }
    .stApp { background-color: #f0f4f8; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d2b55 0%, #1565c0 100%);
    }
    section[data-testid="stSidebar"] * { color: #e3f2fd !important; }

    .kpi-card {
        background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
        border-radius: 14px; padding: 20px 16px; color: white;
        text-align: center; box-shadow: 0 4px 16px rgba(21,101,192,0.25);
        border: 1px solid rgba(255,255,255,0.15);
    }
    .kpi-card .value { font-size: 1.9rem; font-weight: 700; color: #bbdefb; }
    .kpi-card .label { font-size: 0.82rem; color: #90caf9; margin-top: 4px; }
    .kpi-card .unit  { font-size: 0.72rem; color: #64b5f6; margin-top: 2px; }

    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #0d47a1;
        border-left: 5px solid #1976d2; padding-left: 10px; margin: 28px 0 14px;
    }
    .info-box {
        background: #e3f2fd; border-radius: 8px; padding: 12px 16px;
        font-size: 0.88rem; color: #1565c0; border-left: 5px solid #1976d2;
        margin-bottom: 16px;
    }

    /* 네비게이션 버튼 */
    .stButton > button {
        width: 100%; border-radius: 10px; padding: 12px 8px;
        font-size: 0.95rem; font-weight: 600; transition: all 0.2s ease;
        border: 2px solid #90caf9; background: white; color: #1565c0;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1565c0, #42a5f5) !important;
        color: white !important; border-color: #1565c0 !important;
        box-shadow: 0 4px 14px rgba(21,101,192,0.35);
        transform: translateY(-2px);
    }
    .stButton > button:focus, .stButton > button:active {
        background: linear-gradient(135deg, #0d47a1, #1565c0) !important;
        color: white !important; border-color: #0d47a1 !important;
    }

    .policy-card {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%);
        border-radius: 14px; padding: 20px; color: white;
        text-align: center; box-shadow: 0 4px 16px rgba(13,71,161,0.3);
    }
    .policy-card .pval   { font-size: 1.8rem; font-weight: 700; color: #bbdefb; }
    .policy-card .ptitle { font-size: 0.9rem; color: #90caf9; margin-top: 4px; }
    .policy-card .pdesc  { font-size: 0.78rem; color: #64b5f6; margin-top: 6px; }
    .policy-card .pstat  { font-size: 0.72rem; color: #42a5f5; margin-top: 8px;
                            background: rgba(255,255,255,0.1); border-radius: 6px; padding: 4px 8px; }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #1565c0, #1976d2) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
    }
    .footer { text-align:center; color:#90a4ae; font-size:0.8rem; padding: 16px 0; }
</style>
""", unsafe_allow_html=True)


# ── 데이터 생성 ──────────────────────────────────────────────
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
            hour = rng.choice([5,6,7,8,9,10])
            temp_base = 8.2 + (hour - 5) * 0.15
            rows.append({
                "구명": district, "측정시각": hour,
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
PLOT_BASE = dict(plot_bgcolor="white", paper_bgcolor="white",
                 font=dict(family="Noto Sans KR", color="#1a237e"))
BLUES = ["#1565c0","#1976d2","#1e88e5","#42a5f5","#90caf9","#bbdefb"]


# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💧 서울시 수질 분석")
    st.markdown("**측정 기간**: 2026.03.19 05~10시")
    st.markdown("**데이터**: 25개 자치구 2,535건")
    st.markdown("---")
    st.markdown("#### 🔎 필터")
    sel_districts = st.multiselect("자치구 선택", sorted(df["구명"].unique()), default=[])
    sel_hours     = st.multiselect("측정 시각", [5,6,7,8,9,10], default=[5,6,7,8,9,10],
                                   format_func=lambda x: f"{x}시")
    sel_metric    = st.selectbox("주요 지표", ["탁도","잔류염소","pH","전기전도도","수온","수은농도"])
    st.markdown("---")
    st.markdown("#### 📋 수질 기준")
    st.markdown("""
| 항목 | 기준 |
|------|------|
| pH | 5.8 ~ 8.5 |
| 탁도 | ≤ 0.5 NTU |
| 잔류염소 | 0.1~4.0 mg/L |
| 수은 | ≤ 0.001 mg/L |
""")

# ── 필터 적용 ────────────────────────────────────────────────
df_f = df.copy()
if sel_districts: df_f = df_f[df_f["구명"].isin(sel_districts)]
if sel_hours:     df_f = df_f[df_f["측정시각"].isin(sel_hours)]

# ── 헤더 ─────────────────────────────────────────────────────
st.markdown("# 💧 서울특별시 수질 분석 대시보드")
st.markdown(
    '<div class="info-box">📌 서울시 수질오염 현황 분석 및 정책 실효성 평가 보고서(2026.03.19) 기반 대시보드입니다. '
    '정책 비교 데이터는 시뮬레이션 가상 데이터를 포함합니다.</div>',
    unsafe_allow_html=True
)

# ── KPI 카드 ─────────────────────────────────────────────────
for col, (val, label, unit) in zip(st.columns(5), [
    ("2,535건",                       "총 측정 건수",  ""),
    (f"{df_f['탁도'].mean():.3f}",    "평균 탁도",     "NTU"),
    (f"{df_f['잔류염소'].mean():.3f}", "평균 잔류염소", "mg/L"),
    (f"{df_f['pH'].mean():.2f}",      "평균 pH",       ""),
    (f"{df_f['수온'].mean():.1f}",    "평균 수온",     "℃"),
]):
    col.markdown(
        f'<div class="kpi-card"><div class="value">{val}</div>'
        f'<div class="label">{label}</div><div class="unit">{unit}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── 버튼 네비게이션 ──────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "지역별 분포"

nav_items = [
    ("📊 지역별 분포",     "지역별 분포"),
    ("⏱️ 시간별 변화",    "시간별 변화"),
    ("🏛️ 정책 전후 비교", "정책 전후 비교"),
    ("📋 원본 데이터",     "원본 데이터"),
]
for col, (label, key) in zip(st.columns(4), nav_items):
    if col.button(label, key=f"nav_{key}"):
        st.session_state.active_tab = key

active = st.session_state.active_tab
st.markdown(f"<div class='section-title'>현재 화면 : {active}</div>", unsafe_allow_html=True)
st.markdown("---")


# ════════════════════════════════════════════════════════════
# 화면 1 : 지역별 분포
# ════════════════════════════════════════════════════════════
if active == "지역별 분포":
    dist_agg = (
        df_f.groupby("구명")[sel_metric]
        .agg(["mean","std","count"]).reset_index()
        .rename(columns={"mean":"평균","std":"표준편차","count":"건수"})
        .sort_values("평균", ascending=False)
    )
    q75, q25 = dist_agg["평균"].quantile(0.75), dist_agg["평균"].quantile(0.25)
    bar_colors = ["#ef5350" if v > q75 else "#42a5f5" if v < q25 else "#1976d2"
                  for v in dist_agg["평균"]]

    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig = go.Figure(go.Bar(
            x=dist_agg["구명"], y=dist_agg["평균"], marker_color=bar_colors,
            error_y=dict(type="data", array=dist_agg["표준편차"], visible=True, color="#90caf9"),
            hovertemplate="<b>%{x}</b><br>평균: %{y:.4f}<extra></extra>"
        ))
        fig.update_layout(title=f"자치구별 {sel_metric} 평균", xaxis_tickangle=-45, height=420,
                          xaxis=dict(gridcolor="#e3f2fd"), yaxis=dict(gridcolor="#e3f2fd"), **PLOT_BASE)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">상위 5 🔴 (주의)</div>', unsafe_allow_html=True)
        st.dataframe(dist_agg.head(5)[["구명","평균"]].set_index("구명"), use_container_width=True)
        st.markdown('<div class="section-title">하위 5 🔵 (양호)</div>', unsafe_allow_html=True)
        st.dataframe(dist_agg.tail(5)[["구명","평균"]].set_index("구명"), use_container_width=True)

    st.markdown('<div class="section-title">항목별 분포 — Box Plot</div>', unsafe_allow_html=True)
    fig2 = make_subplots(rows=1, cols=4, subplot_titles=["탁도","잔류염소","pH","수온"])
    for i, (m, c) in enumerate(zip(["탁도","잔류염소","pH","수온"], BLUES), 1):
        fig2.add_trace(go.Box(y=df_f[m], name=m, marker_color=c, showlegend=False), row=1, col=i)
    fig2.update_layout(height=340, **PLOT_BASE)
    fig2.update_xaxes(gridcolor="#e3f2fd"); fig2.update_yaxes(gridcolor="#e3f2fd")
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# 화면 2 : 시간별 변화
# ════════════════════════════════════════════════════════════
elif active == "시간별 변화":
    hourly = (df_f.groupby("측정시각")[["잔류염소","탁도","pH","수온"]].mean().reset_index())
    x_vals = hourly["측정시각"].astype(str) + "시"

    fig3 = make_subplots(rows=2, cols=2,
                         subplot_titles=["잔류염소 (mg/L)","탁도 (NTU)","pH","수온 (℃)"])
    for (m, r, c), color in zip(
        [("잔류염소",1,1),("탁도",1,2),("pH",2,1),("수온",2,2)],
        ["#1565c0","#1976d2","#1e88e5","#42a5f5"]
    ):
        fig3.add_trace(go.Scatter(
            x=x_vals, y=hourly[m], mode="lines+markers", name=m,
            line=dict(color=color, width=2.5), marker=dict(size=9, color=color),
        ), row=r, col=c)

    fig3.update_layout(height=500, showlegend=False, **PLOT_BASE)
    fig3.update_xaxes(gridcolor="#e3f2fd"); fig3.update_yaxes(gridcolor="#e3f2fd")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">시간대별 수치 요약표</div>', unsafe_allow_html=True)
    st.dataframe(
        hourly.rename(columns={"측정시각":"시각"})
        .style.format({"잔류염소":"{:.4f}","탁도":"{:.4f}","pH":"{:.3f}","수온":"{:.2f}"}),
        use_container_width=True
    )


# ════════════════════════════════════════════════════════════
# 화면 3 : 정책 전후 비교
# ════════════════════════════════════════════════════════════
elif active == "정책 전후 비교":
    st.markdown(
        '<div class="info-box">⚠️ 아래 데이터는 정책 효과 시뮬레이션을 위한 <b>가상 데이터</b>입니다. '
        '실제 정책 효과 분석을 위해서는 실측 종단 데이터가 필요합니다.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Welch\'s t-검정 결과 (p &lt; 0.01)</div>',
                unsafe_allow_html=True)
    for col, (title, change, values, stat) in zip(st.columns(3), [
        ("탁도 감소",     "↓ 45.1%", "0.1031 → 0.0566 NTU",   "t = 5.573"),
        ("잔류염소 증가", "↑ 44.7%", "0.2620 → 0.3790 mg/L",  "t = 4.971"),
        ("수은농도 감소", "↓ 60.0%", "0.00075 → 0.00030 mg/L","t = 3.821"),
    ]):
        col.markdown(
            f'<div class="policy-card">'
            f'<div class="pval">{change}</div><div class="ptitle">{title}</div>'
            f'<div class="pdesc">{values}</div>'
            f'<div class="pstat">{stat} · p &lt; 0.01 ✅</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    pol_metric = st.selectbox("비교 항목 선택", ["탁도","잔류염소","수은농도"])
    before_df = df_policy[df_policy["기간"] == "정책 이전(2023)"]
    after_df  = df_policy[df_policy["기간"] == "정책 이후(2025)"]

    col_l, col_r = st.columns(2)
    with col_l:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name="정책 이전(2023)", x=before_df["구명"],
                              y=before_df[pol_metric], marker_color="#90caf9"))
        fig4.add_trace(go.Bar(name="정책 이후(2025)", x=after_df["구명"],
                              y=after_df[pol_metric],  marker_color="#1565c0"))
        fig4.update_layout(barmode="group", title=f"정책 전후 {pol_metric} 비교",
                           height=400, xaxis_tickangle=-30,
                           legend=dict(orientation="h", y=1.12),
                           xaxis=dict(gridcolor="#e3f2fd"), yaxis=dict(gridcolor="#e3f2fd"),
                           **PLOT_BASE)
        st.plotly_chart(fig4, use_container_width=True)

    with col_r:
        merged = before_df[["구명",pol_metric]].merge(
            after_df[["구명",pol_metric]], on="구명", suffixes=("_이전","_이후")
        )
        mn = merged[[f"{pol_metric}_이전",f"{pol_metric}_이후"]].min().min() * 0.95
        mx = merged[[f"{pol_metric}_이전",f"{pol_metric}_이후"]].max().max() * 1.05
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                  line=dict(dash="dash", color="#90caf9", width=1.5), name="변화 없음"))
        fig5.add_trace(go.Scatter(
            x=merged[f"{pol_metric}_이전"], y=merged[f"{pol_metric}_이후"],
            mode="markers+text", text=merged["구명"], textposition="top center",
            marker=dict(size=11, color="#1565c0", line=dict(color="white", width=1.5)),
            hovertemplate="<b>%{text}</b><br>이전: %{x:.5f}<br>이후: %{y:.5f}<extra></extra>",
            name=pol_metric
        ))
        fig5.update_layout(title=f"{pol_metric} 산점도 (대각선 아래 = 개선)",
                           height=400, xaxis_title="정책 이전", yaxis_title="정책 이후",
                           xaxis=dict(gridcolor="#e3f2fd"), yaxis=dict(gridcolor="#e3f2fd"),
                           **PLOT_BASE)
        st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════
# 화면 4 : 원본 데이터
# ════════════════════════════════════════════════════════════
elif active == "원본 데이터":
    st.markdown(
        f'<div class="kpi-card" style="max-width:200px">'
        f'<div class="value">{len(df_f):,}건</div>'
        f'<div class="label">조회 건수</div></div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    search = st.text_input("🔍 자치구 검색", "")
    show_df = df_f[df_f["구명"].str.contains(search)] if search else df_f
    st.dataframe(show_df.sort_values(["구명","측정시각"]).reset_index(drop=True),
                 use_container_width=True, height=500)
    st.download_button(
        label="⬇️ CSV 다운로드",
        data=show_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="seoul_water_quality.csv",
        mime="text/csv"
    )


# ── 푸터 ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p class='footer'>서울특별시 환경정책과 | POLY Analyst | 2026.03.19</p>",
    unsafe_allow_html=True
)
