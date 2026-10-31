import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# 페이지 기본 설정
st.set_page_config(
    page_title="고객 행동 데이터 기초통계분석",
    page_icon="📊",
    layout="wide"
)

# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    # 파일명이 다를 경우 실제 데이터 파일 경로로 변경하세요.
    df = pd.read_csv('processed_data_korean_v3.csv')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 대시보드 제목 및 설명
st.title("📊 고객 행동 데이터 기초통계분석 대시보드")
st.markdown("---")

# 사이드바 설정
st.sidebar.header("⚙️ 분석 설정")

# 분석 대상 수치형 컬럼 선택 (식별용 '번호' 제외)
numeric_cols = [c for c in df.columns if c not in ['번호']]

# 메인 KPI 요약
st.subheader("📌 핵심 데이터 요약 (KPI)")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("총 데이터 수 (행)", f"{len(df):,}개")
col2.metric("총 변수 수 (열)", f"{len(df.columns)}개")
col3.metric("평균 체류시간", f"{df['체류시간_초'].mean():.1f} 초")
col4.metric("평균 장바구니 금액", f"{df['장바구니_금액'].mean():,.0f} 원")
col5.metric("전환율", f"{(df['전환_여부'].mean() * 100):.2f}%")

st.markdown("---")

# 탭 구성
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 데이터 미리보기 & 요약통계", 
    "📈 변수별 분포 분석", 
    "🔗 상관관계 분석", 
    "🎯 이탈/전환 그룹 비교"
])

# TAB 1: 데이터 미리보기 & 기술통계
with tab1:
    st.markdown("### 1. 데이터 샘플 (상위 10개 행)")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 2. 기술통계량 (Descriptive Statistics)")
    stats_df = df[numeric_cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats_df.columns = ['개수', '평균', '표준편차', '최소값', '25%', '중앙값(50%)', '75%', '최대값']
    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

# TAB 2: 단일 변수 분포 분석 (Plotly)
with tab2:
    st.markdown("### 변수별 데이터 분포 확인")
    selected_var = st.selectbox("분석할 변수를 선택하세요:", numeric_cols, index=0)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        # 히스토그램 (Plotly)
        fig_hist = px.histogram(
            df, 
            x=selected_var, 
            nbins=30, 
            title=f"[{selected_var}] 히스토그램",
            color_discrete_sequence=['#3366CC']
        )
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_right:
        # 상자 그림 (Boxplot)
        fig_box = px.box(
            df, 
            y=selected_var, 
            title=f"[{selected_var}] 상자 그림 (Box Plot)",
            color_discrete_sequence=['#109618']
        )
        st.plotly_chart(fig_box, use_container_width=True)

# TAB 3: 상관관계 분석
with tab3:
    st.markdown("### 수치형 변수 간 상관관계 히트맵")
    corr = df[numeric_cols].corr()
    
    # Plotly Correlation Heatmap
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="변수 간 Pearson 상관계수 Matrix"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# TAB 4: 이탈/전환 타겟 변수별 비교
with tab4:
    st.markdown("### 전환 및 이탈 여부에 따른 주요 변수 비교")
    
    group_target = st.radio("비교 기준 타겟 변수 선택:", ['전환_여부', '이탈_여부'], horizontal=True)
    compare_var = st.selectbox("비교할 대상 변수 선택:", [c for c in numeric_cols if c not in ['전환_여부', '이탈_여부']], index=0)
    
    fig_group = px.box(
        df, 
        x=group_target, 
        y=compare_var, 
        color=group_target,
        title=f"{group_target}에 따른 {compare_var} 차이 분석",
        color_discrete_map={0: '#FF9900', 1: '#109618'}
    )
    st.plotly_chart(fig_group, use_container_width=True)
