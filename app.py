import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="서울시 수질 분석 대시보드",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 커스텀 CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #0f2942 0%, #1a4a7a 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
    .metric-card .label { font-size: 0.85rem; color: #90caf9; margin-top: 4px; }
    .metric-card .unit  { font-size: 0.75rem; color: #78909c; }

    .section-title {
        font-size: 1.2rem; font-weight: 700;
        color: #0d47a1; border-left: 4px solid #1976d2;
        padding-left: 10px; margin: 24px 0 12px;
    }

    .info-box {
        background: #e3f2fd; border-radius: 8px;
        padding: 12px 16px; font-size: 0.88rem; color: #1565c0;
        border-left: 4px solid #1976d2; margin-bottom: 16px;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #e8f4fd; border-radius: 8px 8px 0 0;
        font-weight: 500; color: #1565c0;
    }
    .stTabs [aria-selected="true"] { background: #1976d2 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)


# ── 데이터 생성 ──────────────────────────────────────────────
@st.cache_data
def load_data():
    """보고서 기반 데이터 생성"""
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
    hours = [5, 6, 7, 8, 9, 10]
    for district in districts:
        n = rng.integers(60, 130)
        for _ in range(n):
            hour = rng.choice(hours)
            temp_base = 8.2 + (hour - 5) * 0.15
            rows.append({
                "구명": district,
                "측정시각": hour,
                "전기전도도": round(float(rng.normal(277.3, 17.8)), 1),
                "pH": round(float(rng.normal(7.23, 0.37)), 2),
                "잔류염소": round(float(rng.normal(chlorine_mean[district], 0.04)), 3),
                "탁도": round(float(rng.normal(turbidity_mean[district], 0.008)), 4),
                "수온": round(float(rng.normal(temp_base, 0.5)), 1),
                "수은농도": round(float(rng.normal(0.00055, 0.00015)), 6),
            })
    df = pd.DataFrame(rows)
    df["탁도"] = df["탁도"].clip(0.01, 0.78)
    df["잔류염소"] = df["잔류염소"].clip(0.05, 0.75)
    df["pH"] = df["pH"].clip(5.8, 8.5)
    df["수은농도"] = df["수은농도"].clip(0.0001, 0.001)
    return df


@st.cache_data
def load_policy_data():
    """정책 전후 비교 데이터"""
    before = {
        "구명":     ["노원구","강남구","송파구","구로구","서초구","중랑구","마포구","강서구","도봉구","성북구"],
        "탁도":     [0.078,0.108,0.138,0.121,0.067,0.127,0.075,0.094,0.109,0.114],
        "잔류염소": [0.22, 0.26, 0.28, 0.30, 0.21, 0.35, 0.24, 0.27, 0.23, 0.25],
        "수은농도": [0.0008,0.0007,0.0009,0.0008,0.0006,0.0009,0.0007,0.0008,0.0007,0.0007],
    }
    after = {
        "구명":     ["노원구","강남구","송파구","구로구","서초구","중랑구","마포구","강서구","도봉구","성북구"],
        "탁도":     [0.046,0.064,0.045,0.067,0.042,0.042,0.066,0.069,0.059,0.066],
        "잔류염소": [0.32, 0.35, 0.28, 0.33, 0.25, 0.38, 0.30, 0.32, 0.27, 0.29],
        "수은농도": [0.0003,0.0003,0.0002,0.0004,0.0003,0.0004,0.0003,0.0003,0.0002,0.0003],
    }
    df_b = pd.DataFrame(before); df_b["기간"] = "정책 이전(2023)"
    df_a = pd.DataFrame(after);  df_a["기간"] = "정책 이후(2025)"
    return pd.concat([df_b, df_a], ignore_index=True)


df = load_data()
df_policy = load_policy_data()

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💧 서울시 수질 분석")
    st.markdown("**측정 기간**: 2026.03.19 05~10시")
    st.markdown("**데이터**: 25개 자치구")
    st.markdown("---")

    st.markdown("#### 🔎 필터")
    sel_districts = st.multiselect(
        "자치구 선택", sorted(df["구명"].unique()), default=[]
    )
    sel_hours = st.multiselect(
        "측정 시각", [5,6,7,8,9,10],
        default=[5,6,7,8,9,10],
        format_func=lambda x: f"{x}시"
    )
    sel_metric = st.selectbox(
        "주요 지표 선택",
        ["탁도", "잔류염소", "pH", "전기전도도", "수온", "수은농도"]
    )

    st.markdown("---")
    st.markdown("#### 📋 수질 기준")
    st.markdown("""
    | 항목 | 기준 |
    |------|------|
    | pH | 5.8 ~ 8.5 |
    | 탁도 | ≤ 0.5 NTU |
    | 잔류염소 | 0.1 ~ 4.0 mg/L |
    | 수은 | ≤ 0.001 mg/L |
    """)


# ── 데이터 필터링 ────────────────────────────────────────────
df_f = df.copy()
if sel_districts:
    df_f = df_f[df_f["구명"].isin(sel_districts)]
if sel_hours:
    df_f = df_f[df_f["측정시각"].isin(sel_hours)]

# ── 헤더 ─────────────────────────────────────────────────────
st.markdown("# 💧 서울특별시 수질 분석 대시보드")
st.markdown(
    '<div class="info-box">📌 본 대시보드는 서울시 수질오염 현황 분석 및 정책 실효성 평가 보고서(2026.03.19) 기반으로 작성되었습니다. '
    '정책 비교 데이터는 시뮬레이션 가상 데이터를 포함합니다.</div>',
    unsafe_allow_html=True
)

# ── KPI 카드 ─────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    ("2,535건", "총 측정 건수", ""),
    (f"{df_f['탁도'].mean():.3f}", "평균 탁도", "NTU"),
    (f"{df_f['잔류염소'].mean():.3f}", "평균 잔류염소", "mg/L"),
    (f"{df_f['pH'].mean():.2f}", "평균 pH", ""),
    (f"{df_f['수온'].mean():.1f}", "평균 수온", "℃"),
]
for col, (val, label, unit) in zip([c1,c2,c3,c4,c5], kpis):
    col.markdown(
        f'<div class="metric-card"><div class="value">{val}</div>'
        f'<div class="label">{label}</div><div class="unit">{unit}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── 탭 ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 지역별 분포", "⏱️ 시간별 변화", "🏛️ 정책 전후 비교", "📋 원본 데이터"
])

# ── TAB 1: 지역별 분포 ────────────────────────────────────────
with tab1:
    st.markdown(f'<div class="section-title">자치구별 {sel_metric} 분포</div>', unsafe_allow_html=True)

    dist_agg = (
        df_f.groupby("구명")[sel_metric]
        .agg(["mean","std","count"])
        .reset_index()
        .rename(columns={"mean":"평균","std":"표준편차","count":"건수"})
        .sort_values("평균", ascending=False)
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        colors = ["#ef5350" if v > dist_agg["평균"].quantile(0.75) else
                  "#42a5f5" if v < dist_agg["평균"].quantile(0.25) else "#66bb6a"
                  for v in dist_agg["평균"]]
        fig = go.Figure(go.Bar(
            x=dist_agg["구명"], y=dist_agg["평균"],
            marker_color=colors,
            error_y=dict(type="data", array=dist_agg["표준편차"], visible=True),
            hovertemplate="<b>%{x}</b><br>평균: %{y:.4f}<extra></extra>"
        ))
        fig.update_layout(
            title=f"자치구별 {sel_metric} 평균",
            xaxis_tickangle=-45, height=420,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#e0e0e0"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown(f'<div class="section-title">상위 5 / 하위 5</div>', unsafe_allow_html=True)
        st.markdown("**🔴 상위 5 (주의)**")
        st.dataframe(dist_agg.head(5)[["구명","평균"]].set_index("구명"), use_container_width=True)
        st.markdown("**🔵 하위 5 (양호)**")
        st.dataframe(dist_agg.tail(5)[["구명","평균"]].set_index("구명"), use_container_width=True)

    st.markdown('<div class="section-title">항목별 분포 (Box Plot)</div>', unsafe_allow_html=True)
    metrics = ["탁도","잔류염소","pH","수온"]
    fig2 = make_subplots(rows=1, cols=4, subplot_titles=metrics)
    for i, m in enumerate(metrics, 1):
        fig2.add_trace(
            go.Box(y=df_f[m], name=m, marker_color="#1976d2", showlegend=False),
            row=1, col=i
        )
    fig2.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)


# ── TAB 2: 시간별 변화 ────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">시간대별 수질 항목 변화</div>', unsafe_allow_html=True)

    hourly = (
        df_f.groupby("측정시각")[["잔류염소","탁도","pH","수온"]]
        .mean().reset_index()
    )

    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=["잔류염소 (mg/L)","탁도 (NTU)","pH","수온 (℃)"]
    )
    cfg = [
        ("잔류염소","#1976d2",1,1), ("탁도","#e53935",1,2),
        ("pH","#43a047",2,1), ("수온","#fb8c00",2,2),
    ]
    for col_name, color, r, c in cfg:
        fig3.add_trace(
            go.Scatter(
                x=hourly["측정시각"].astype(str) + "시",
                y=hourly[col_name], mode="lines+markers",
                name=col_name, line=dict(color=color, width=2.5),
                marker=dict(size=8),
            ), row=r, col=c
        )
    fig3.update_layout(height=500, showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    fig3.update_xaxes(gridcolor="#f0f0f0")
    fig3.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">시간대별 수치 요약표</div>', unsafe_allow_html=True)
    st.dataframe(
        hourly.rename(columns={"측정시각":"시각"})
        .style.format({"잔류염소":"{:.4f}","탁도":"{:.4f}","pH":"{:.3f}","수온":"{:.2f}"}),
        use_container_width=True
    )


# ── TAB 3: 정책 전후 비교 ─────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">2024 수질 안전관리 강화 대책 효과 분석</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">⚠️ 아래 데이터는 정책 효과 시뮬레이션을 위한 <b>가상 데이터</b>입니다. '
        '실제 정책 효과 분석을 위해서는 실측 종단 데이터가 필요합니다.</div>',
        unsafe_allow_html=True
    )

    # t-검정 결과 요약
    col1, col2, col3 = st.columns(3)
    ttest_results = [
        (col1, "탁도 감소", "↓ 45.1%", "0.1031 → 0.0566 NTU", "t=5.573, p<0.01"),
        (col2, "잔류염소 증가", "↑ 44.7%", "0.2620 → 0.3790 mg/L", "t=4.971, p<0.01"),
        (col3, "수은농도 감소", "↓ 60.0%", "0.00075 → 0.00030 mg/L", "t=3.821, p<0.01"),
    ]
    for col, title, change, values, stat in ttest_results:
        color = "#1b5e20" if "↓" in change or "↑" in change else "#b71c1c"
        col.markdown(
            f'<div class="metric-card" style="background: linear-gradient(135deg,#1b4332,#2d6a4f)">'
            f'<div class="value" style="color:#69db7c">{change}</div>'
            f'<div class="label">{title}</div>'
            f'<div class="unit">{values}</div>'
            f'<div class="unit" style="margin-top:6px;font-size:0.7rem;color:#95d5b2">{stat}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    pol_metric = st.selectbox("비교 항목", ["탁도","잔류염소","수은농도"], key="pol")

    before_df = df_policy[df_policy["기간"]=="정책 이전(2023)"]
    after_df  = df_policy[df_policy["기간"]=="정책 이후(2025)"]

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        name="정책 이전(2023)", x=before_df["구명"], y=before_df[pol_metric],
        marker_color="#ef9a9a"
    ))
    fig4.add_trace(go.Bar(
        name="정책 이후(2025)", x=after_df["구명"], y=after_df[pol_metric],
        marker_color="#81c784"
    ))
    fig4.update_layout(
        barmode="group", title=f"정책 전후 {pol_metric} 비교",
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        xaxis_tickangle=-30, legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig4, use_container_width=True)

    # scatter
    merged = before_df[["구명",pol_metric]].merge(
        after_df[["구명",pol_metric]], on="구명", suffixes=("_이전","_이후")
    )
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=merged[f"{pol_metric}_이전"], y=merged[f"{pol_metric}_이후"],
        mode="markers+text", text=merged["구명"],
        textposition="top center", marker=dict(size=10, color="#1976d2"),
        hovertemplate="<b>%{text}</b><br>이전:%{x:.5f} → 이후:%{y:.5f}<extra></extra>"
    ))
    mn = min(merged[f"{pol_metric}_이전"].min(), merged[f"{pol_metric}_이후"].min())
    mx = max(merged[f"{pol_metric}_이전"].max(), merged[f"{pol_metric}_이후"].max())
    fig5.add_trace(go.Scatter(
        x=[mn,mx], y=[mn,mx], mode="lines",
        line=dict(dash="dash", color="gray"), name="변화 없음"
    ))
    fig5.update_layout(
        title=f"{pol_metric} 정책 전후 산점도 (대각선 아래 = 개선)",
        height=400, plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="정책 이전", yaxis_title="정책 이후"
    )
    st.plotly_chart(fig5, use_container_width=True)


# ── TAB 4: 원본 데이터 ────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">필터링된 원본 데이터</div>', unsafe_allow_html=True)

    col_s, col_d = st.columns([1,3])
    with col_s:
        st.metric("조회 건수", f"{len(df_f):,}건")

    search = st.text_input("🔍 자치구 검색", "")
    show_df = df_f[df_f["구명"].str.contains(search)] if search else df_f

    st.dataframe(
        show_df.sort_values(["구명","측정시각"]).reset_index(drop=True),
        use_container_width=True, height=480
    )

    csv = show_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="⬇️ CSV 다운로드",
        data=csv,
        file_name="seoul_water_quality.csv",
        mime="text/csv"
    )

# ── 푸터 ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#90a4ae; font-size:0.8rem;'>"
    "서울특별시 환경정책과 | POLY Analyst | 2026.03.19</p>",
    unsafe_allow_html=True
)
