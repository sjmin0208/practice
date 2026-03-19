import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="Seoul Water Quality Insight",
    page_icon="💧",
    layout="wide",
)

# ── 커스텀 스타일 (Modern Glass Design) ─────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;600;800&display=swap');

    * { font-family: 'Pretendard', sans-serif; }
    
    .stApp { background-color: #f8fafc; }

    /* 사이드바 스타일링 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* KPI 카드 디자인 */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
    }

    /* 섹션 타이틀 */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* 커스텀 버튼 (Navigation) */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background: white;
        color: #475569;
        font-weight: 600;
        height: 3rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        border-color: #3b82f6;
        color: #3b82f6;
        background: #eff6ff;
    }
    
    /* 강조 상자 */
    .highlight-box {
        padding: 1rem;
        border-radius: 12px;
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        color: #0369a1;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── 데이터 로드 함수 (기존 로직 유지) ───────────────────────
@st.cache_data
def get_data():
    rng = np.random.default_rng(42)
    districts = ["강남구","강동구","강북구","강서구","관악구","광진구","구로구","금천구","노원구","도봉구","동대문구","동작구","마포구","서대문구","서초구","성동구","성북구","송파구","양천구","영등포구","용산구","은평구","종로구","중구","중랑구"]
    rows = []
    for district in districts:
        n = rng.integers(60, 130)
        for _ in range(n):
            hour = rng.choice([5, 6, 7, 8, 9, 10])
            rows.append({
                "구명": district, "측정시각": hour,
                "전기전도도": round(float(rng.normal(277.3, 17.8)), 1),
                "pH": round(float(rng.normal(7.23, 0.37)), 2),
                "잔류염소": round(float(rng.normal(0.28, 0.04)), 3),
                "탁도": round(float(rng.normal(0.05, 0.008)), 4),
                "수온": round(float(rng.normal(8.5, 0.5)), 1),
                "수은농도": round(float(rng.normal(0.0005, 0.0001)), 6)
            })
    return pd.DataFrame(rows)

df = get_data()

# ── 사이드바 ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water-drop.png", width=80)
    st.title("Water Quality")
    st.markdown("---")
    
    sel_districts = st.multiselect("📍 분석 지역", sorted(df["구명"].unique()))
    sel_metric = st.selectbox("📊 분석 지표", ["탁도", "잔류염소", "pH", "전기전도도", "수온"])
    
    st.markdown("---")
    st.info("💡 **수질 기준 안내**\n- pH: 5.8 ~ 8.5\n- 탁도: 0.5 NTU 이하\n- 잔류염소: 4.0 mg/L 이하")

# ── 메인 대시보드 ───────────────────────────────────────────
st.title("🏙️ 서울시 수질 분석 리포트")
st.markdown(f'<div class="highlight-box">실시간 측정 데이터 기반 서울시 자치구별 수질 현황을 분석합니다. (기준일: 2026.03.19
