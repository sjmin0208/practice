import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats

# 페이지 설정
st.set_page_config(page_title="서울시 수질정책 평가 대시보드", layout="wide")

@st.cache_data
def load_data():
    # 보고서 제3장/제4장의 데이터를 기반으로 한 샘플 데이터 프레임 생성 [cite: 55, 61]
    policy_data = pd.DataFrame({
        'Period': ['Before', 'Before', 'After', 'After'] * 5,
        'District': ['Gangnam', 'Nowon', 'Gangnam', 'Nowon'] * 5,
        'Turbidity': [0.103, 0.078, 0.056, 0.046] * 5,  # [cite: 61, 63]
        'Chlorine': [0.26, 0.22, 0.35, 0.32] * 5,      # [cite: 61, 63]
        'Mercury': [0.0007, 0.0008, 0.0003, 0.0003] * 5 # [cite: 61, 63]
    })
    return policy_data

df = load_data()

# 헤더 영역
st.title("📊 서울특별시 수질분석 및 정책 실효성 평가")
st.markdown("2026년 3월 19일 기준 실측 데이터 및 정책 시뮬레이션 분석 [cite: 6, 10]")

# 사이드바 - 분석 필터
st.sidebar.header("Filter Settings")
selected_district = st.sidebar.multiselect("자치구 선택", options=['Gangnam', 'Nowon', 'Seocho', 'Jungrang'], default=['Gangnam', 'Nowon'])

# 메인 탭 구성
tab1, tab2, tab3 = st.tabs(["수질 현황 요약", "지역별 분포 분석", "정책 효과 검증(t-test)"])

with tab1:
    st.header("📍 실측 데이터 기술통계 [cite: 67]")
    col1, col2, col3, col4 = st.columns(4)
    # 보고서 제5장 통계치 반영 [cite: 67]
    col1.metric("평균 pH", "7.23", "0.37 (std)")
    col2.metric("평균 탁도", "0.055 NTU", "-45.1% (vs Previous)")
    col3.metric("평균 잔류염소", "0.280 mg/L", "+44.7% (vs Previous)")
    col4.metric("평균 수온", "8.79 ℃", "0.9 ℃ (Range)")

with tab2:
    st.header("🗺️ 자치구별 수질 지표 분포 [cite: 88]")
    # 지역별 탁도 분포 차트 (제7장 데이터 반영) [cite: 85, 88]
    fig_bar = px.bar(df, x='District', y='Turbidity', color='Period', barmode='group',
                     title="자치구별 정책 전후 탁도 비교")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.info("강남구, 서초구, 중랑구 등 상업/공업 밀집 지역의 탁도가 상대적으로 높게 관측됨 [cite: 85, 86]")

with tab3:
    st.header("🧪 정책 전후 Welch's t-test 결과 [cite: 101, 102]")
    
    target_col = st.selectbox("검증 항목 선택", ["Turbidity", "Chlorine", "Mercury"])
    
    before = df[df['Period'] == 'Before'][target_col]
    after = df[df['Period'] == 'After'][target_col]
    
    t_stat, p_val = stats.ttest_ind(before, after, equal_var=False)
    
    col_stat1, col_stat2 = st.columns(2)
    col_stat1.write(f"**T-Statistic:** {t_stat:.4f}")
    col_stat2.write(f"**P-Value:** {p_val:.4e}")
    
    if p_val < 0.05:
        st.success(f"✅ 통계적으로 유의미한 차이가 발견되었습니다. (p < 0.05) [cite: 115]")
    else:
        st.warning("⚠️ 정책 전후 차이가 통계적으로 유의하지 않습니다.")

    # 정책 제언 섹션 (제9장) [cite: 117]
    st.divider()
    st.subheader("💡 데이터 기반 정책 제언")
    st.markdown("""
    * **고탁도 지역 관리:** 강남/서초/중랑구 플러싱 주기 단축 권고 [cite: 123]
    * **소독 강화:** 원거리 지역(강북/강동) 재염소 처리시설 검토 [cite: 125]
    * **중금속 모니터링:** 공업지역(중랑/구로/금천) 측정 빈도 상향 [cite: 127]
    """)
