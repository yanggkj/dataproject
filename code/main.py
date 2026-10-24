import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 페이지 설정
st.set_page_config(
    page_title="매체접점 인지·호감·구매 통합 데이터 분석",
    page_icon="📊",
    layout="wide"
)

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.title("📊 매체접점 인지·호감·구매 통합 데이터 분석 앱")
st.markdown("제공된 데이터를 기반으로 기초 통계 분석과 데이터 타당성 검토를 수행하는 대시보드입니다.")

# 데이터 로드 함수 (GitHub Raw 링크 활용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/yanggkj/dataproject/main/data/%EB%A7%A4%EC%B2%B4%EC%A0%91%EC%A0%90_%EC%9D%B8%EC%A7%80%ED%98%B8%EA%B0%90%EA%B5%AC%EB%A7%A4_%ED%86%B5%ED%95%A9%20(1).csv"
    try:
        df = pd.read_csv(url, encoding='utf-8')
    except Exception:
        try:
            df = pd.read_csv(url, encoding='cp949')
        except Exception as e:
            st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
            df = None
    return df

df = load_data()

if df is None:
    st.stop()

# 사이드바 메뉴 구성
menu = st.sidebar.selectbox(
    "메뉴 선택",
    ["1. 질문 검토 및 데이터 개요", "2. 기초 통계 분석", "3. 시각화 및 상관관계"]
)

if menu == "1. 질문 검토 및 데이터 개요":
    st.header("1. 질문 사항 검토 및 데이터 개요")
    
    st.subheader("📋 데이터 미리보기 (상위 5행)")
    st.dataframe(df.head())
    
    st.markdown("---")
    st.subheader("🔍 사용자 질문 검토 결과")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Q1. 데이터의 범위가 명확한가?**")
        st.write(
            "- **검토 결과**: 매체접점별 인지, 호감, 구매 단계에 이르는 마케팅/서베이 데이터를 대상으로 하므로 분석 범위가 비교적 명확합니다.\n"
            f"- **데이터 규모**: 총 행(Row) {df.shape[0]}개, 총 열(Column) {df.shape[1]}개로 구성되어 있습니다."
        )
        
        st.info("**Q2. 데이터를 수치화할 수 있는가?**")
        st.write(
            "- **검토 결과**: 점수, 응답 빈도, 비율 등 수치형 열이 4개 이상 포함되어 있어 정량적인 수치화 및 통계 분석이 충분히 가능합니다."
        )
        
    with col2:
        st.info("**Q3. 회귀, 분류, 군집화, 연관 분석이 가능한가?**")
        st.write(
            "- **회귀분석**: 매체접점 인지·호감이 '구매'에 미치는 영향 분석 가능\n"
            "- **분류**: 특정 타겟(구매 여부 등) 예측 모델 적용 가능\n"
            "- **군집화**: 응답자 또는 매체별 유사 성향 그룹화 가능\n"
            "- **연관 분석**: 매체 간 조합이나 항목 간 연관성 분석 가능"
        )
        
    st.markdown("---")
    st.subheader("📊 데이터 구조 요약")
    c1, c2, c3 = st.columns(3)
    c1.metric("행(Row) 개수", f"{df.shape[0]}개")
    c2.metric("열(Column) 개수", f"{df.shape[1]}개")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    c3.metric("수치형 열 개수", f"{len(numeric_cols)}개")

elif menu == "2. 기초 통계 분석":
    st.header("2. 기초 통계 분석 (Descriptive Statistics)")
    
    st.subheader("📈 수치형 변수 기술통계량")
    st.dataframe(df.describe())
    
    st.subheader("🔎 개별 변수 분포 확인")
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    
    if numeric_cols:
        selected_col = st.selectbox("분석할 수치형 열 선택", numeric_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{selected_col} 상세 통계**")
            st.write(df[selected_col].describe())
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[selected_col], kde=True, ax=ax, color='teal')
            ax.set_title(f"{selected_col} 분포")
            st.pyplot(fig)
    else:
        st.warning("분석할 수치형 데이터가 없습니다.")

elif menu == "3. 시각화 및 상관관계":
    st.header("3. 변수 간 상관관계 및 시각화")
    
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    
    if len(numeric_cols) >= 2:
        st.subheader("🔗 상관관계 히트맵")
        corr = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", linewidths=.5, ax=ax)
        st.pyplot(fig)
        
        st.subheader("📉 변수 간 산점도 (Scatter Plot)")
        col_x = st.selectbox("X축 변수 선택", numeric_cols, index=0)
        col_y = st.selectbox("Y축 변수 선택", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax2, color='purple', s=80)
        ax2.set_title(f"{col_x} vs {col_y}")
        st.pyplot(fig2)
    else:
        st.warning("상관관계를 분석하려면 수치형 열이 최소 2개 이상 필요합니다.")
