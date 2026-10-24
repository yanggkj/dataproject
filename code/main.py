import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# Streamlit 페이지 설정
st.set_page_config(
    page_title="매체접점 인지·호감·구매 기초통계분석",
    page_icon="📊",
    layout="wide"
)

st.title("📊 매체접점 연도별 인지·호감·구매 기초통계분석 앱 (Plotly)")
st.markdown("업로드된 **`매체접점_연도별_재구조화.csv`** 데이터를 바탕으로 Plotly 인터랙티브 시각화를 수행합니다.")

# 경로 자동 탐색을 통한 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_clean_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, "../data/매체접점_연도별_재구조화.csv"),
        os.path.join(current_dir, "data/매체접점_연도별_재구조화.csv"),
        os.path.join(current_dir, "매체접점_연도별_재구조화.csv"),
        "data/매체접점_연도별_재구조화.csv",
        "매체접점_연도별_재구조화.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, encoding='utf-8')
                break
            except:
                try:
                    df = pd.read_csv(path, encoding='cp949')
                    break
                except:
                    continue
                    
    if df is None:
        raise FileNotFoundError("지정한 경로에서 파일을 찾을 수 없습니다.")
    
    # 구분/접점(매체)이 결측치인 행 제거
    cleaned_df = df.dropna(subset=['구분', '접점(매체)']).copy()
    cleaned_df['연도'] = cleaned_df['연도'].astype(int)
    
    return cleaned_df

try:
    df = load_and_clean_data()
except Exception as e:
    st.error(f"데이터를 로드하는 중 오류가 발생했습니다: {e}")
    st.stop()

# 사이드바 구성
st.sidebar.header("📌 메뉴")
menu = st.sidebar.radio(
    "이동할 메뉴를 선택하세요:",
    [
        "1. 질문 요건 검토 & 데이터 개요",
        "2. 기초 통계 분석 (Descriptive)",
        "3. 상관관계 및 회귀 분석",
        "4. 연도별/매체별 분석"
    ]
)

# ---------------------------------------------------------
# TAB 1: 질문 요건 검토 & 데이터 개요
# ---------------------------------------------------------
if menu == "1. 질문 요건 검토 & 데이터 개요":
    st.header("1. 데이터 질문 요건 검토 결과")
    
    c1, c2 = st.columns(2)
    with c1:
        st.success("✅ **Q1. 데이터의 범위가 명확한가?**")
        st.write(f"""
        - **범위**: 2017년~2019년 주요 7개 구분(34개 매체 접점)의 마케팅 수치 데이터.
        - **행/열 요건**: 전체 데이터 **{df.shape[0]}개 행**, **{df.shape[1]}개 열** (행 50개 이상, 열 6개 이상 충족)
        """)
        
        st.success("✅ **Q2. 데이터를 수치화할 수 있는가?**")
        st.write("""
        - **수치형 열 (4개)**: `연도`, `인지(%)`, `호감(%)`, `구매(%)` (정량 수치화 완료)
        """)

    with c2:
        st.success("✅ **Q3. 회귀, 분류, 군집화, 연관 분석이 가능한가?**")
        st.write("""
        - **회귀 분석**: 인지/호감이 구매에 미치는 영향 분석 가능
        - **군집화/분류**: 매체별 성과 그룹핑 가능
        - **시계열**: 연도별 트렌드 비교 가능
        """)

    st.markdown("---")
    st.subheader("📋 전처리된 데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------------------------
# TAB 2: 기초 통계 분석
# ---------------------------------------------------------
elif menu == "2. 기초 통계 분석 (Descriptive)":
    st.header("2. 기초 통계 분석")
    
    st.subheader("📌 주요 수치형 데이터 요약 통계량")
    stats_df = df[['인지(%)', '호감(%)', '구매(%)']].describe().T
    stats_df['median'] = df[['인지(%)', '호감(%)', '구매(%)']].median()
    st.dataframe(stats_df[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max']], use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 지표별 분포 (Plotly Interactive Histogram & Boxplot)")
    
    col_choice = st.selectbox("분석할 지표 선택:", ['인지(%)', '호감(%)', '구매(%)'])
    
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(df, x=col_choice, nbins=20, title=f"{col_choice} 히스토그램", marginal="rug", color_discrete_sequence=['teal'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c2:
        fig_box = px.box(df, y=col_choice, title=f"{col_choice} 박스플롯", points="all", color_discrete_sequence=['lightgreen'])
        st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: 상관관계 및 회귀 분석
# ---------------------------------------------------------
elif menu == "3. 상관관계 및 회귀 분석":
    st.header("3. 상관관계 및 간단한 회귀 분석")
    
    st.subheader("🔗 지표 간 상관계수 히트맵")
    corr = df[['인지(%)', '호감(%)', '구매(%)']].corr()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(corr, use_container_width=True)
    with col2:
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title="상관계수 히트맵")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    st.markdown("---")
    st.subheader("📈 선형 회귀 분석 (Linear Regression)")
    
    x_var = st.selectbox("독립변수 (X) 선택:", ['인지(%)', '호감(%)'])
    
    sub_df = df[[x_var, '구매(%)', '접점(매체)', '연도']].dropna()
    X = sub_df[[x_var]]
    y = sub_df['구매(%)']
    
    model = LinearRegression()
    model.fit(X, y)
    sub_df['회귀예측값'] = model.predict(X)
    r2 = r2_score(y, sub_df['회귀예측값'])
    
    st.write(f"- **회귀 방정식**: `구매(%)` = {model.coef_[0]:.4f} * `{x_var}` + ({model.intercept_:.4f})")
    st.write(f"- **결정계수 ($R^2$)**: {r2:.4f}")
    
    # Plotly Scatter + Trendline
    fig_reg = px.scatter(
        sub_df, x=x_var, y='구매(%)', hover_data=['접점(매체)', '연도'],
        trendline="ols", title=f"{x_var}에 따른 구매(%) 회귀 분석"
    )
    fig_reg.update_traces(marker=dict(size=8, color='purple'))
    st.plotly_chart(fig_reg, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: 연도별/매체별 분석
# ---------------------------------------------------------
elif menu == "4. 연도별/매체별 분석":
    st.header("4. 연도별 및 매체 구분별 비교 분석")
    
    st.subheader("📅 연도별 평균 지표 비교")
    yearly_avg = df.groupby('연도')[['인지(%)', '호감(%)', '구매(%)']].mean().reset_index()
    yearly_melted = yearly_avg.melt(id_vars='연도', value_vars=['인지(%)', '호감(%)', '구매(%)'], var_name='지표', value_name='평균비율(%)')
    
    fig_year = px.bar(yearly_melted, x='연도', y='평균비율(%)', color='지표', barmode='group', title="연도별 평균 인지 / 호감 / 구매 (%)")
    st.plotly_chart(fig_year, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🏷️ 매체 구분별 평균 지표")
    category_avg = df.groupby('구분')[['인지(%)', '호감(%)', '구매(%)']].mean().reset_index()
    category_melted = category_avg.melt(id_vars='구분', value_vars=['인지(%)', '호감(%)', '구매(%)'], var_name='지표', value_name='평균비율(%)')
    
    fig_cat = px.bar(category_melted, y='구분', x='평균비율(%)', color='지표', orientation='h', barmode='group', title="매체 구분별 평균 인지 / 호감 / 구매 (%)")
    fig_cat.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_cat, use_container_width=True)
