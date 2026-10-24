import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Streamlit 페이지 설정
st.set_page_config(
    page_title="매체접점 인지·호감·구매 기초통계분석",
    page_icon="📊",
    layout="wide"
)

# Matplotlib 한글 폰트 깨짐 방지 설정
plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.title("📊 매체접점 연도별 인지·호감·구매 기초통계분석 앱")
st.markdown("업로드된 **`data/매체접점_연도별_재구조화.csv`** 데이터를 바탕으로 기초 통계량 및 분석을 수행합니다.")

# 1. 데이터 로드 및 전처리 (상위 폴더의 data 경로 지정)
@st.cache_data
def load_and_clean_data():
    file_path = "../data/매체접점_연도별_재구조화.csv"
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='cp949')
    
    # 구분/접점(매체)이 결측치인 행 제거
    cleaned_df = df.dropna(subset=['구분', '접점(매체)']).copy()
    
    # 연도 컬럼 정수형 변환
    cleaned_df['연도'] = cleaned_df['연도'].astype(int)
    
    return cleaned_df

try:
    df = load_and_clean_data()
except Exception as e:
    st.error(f"데이터를 로드하는 중 오류가 발생했습니다: {e}")
    st.info("💡 `code` 폴더 기준으로 상위 폴더에 있는 `data` 폴더와 파일명을 다시 확인해 주세요.")
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
        - **행/열 요건**: 
            - 전체 데이터: **{df.shape[0]}개 행**, **{df.shape[1]}개 열** (요건: 행 50개 이상, 열 6개 이상 충족)
        """)
        
        st.success("✅ **Q2. 데이터를 수치화할 수 있는가?**")
        st.write("""
        - **수치형 열 (4개)**: `연도`, `인지(%)`, `호감(%)`, `구매(%)`
        - 비율(%) 데이터로 정리되어 표준 정량 분석이 바로 가능합니다.
        """)

    with c2:
        st.success("✅ **Q3. 회귀, 분류, 군집화, 연관 분석이 가능한가?**")
        st.write("""
        - **회귀 분석**: `인지(%)`, `호감(%)`이 `구매(%)`에 미치는 영향 규명 가능
        - **분류/군집화**: 인지 대비 구매 전환율에 따른 고성과/저성과 매체 그룹핑 가능
        - **시계열/트렌드**: 2017~2019년 연도별 흐름 비교 분석 가능
        """)

    st.markdown("---")
    st.subheader("📋 전처리된 데이터 미리보기 (상위 10개 행)")
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
    st.subheader("📊 지표별 분포 (Histogram & Boxplot)")
    
    col_choice = st.selectbox("분석할 지표 선택:", ['인지(%)', '호감(%)', '구매(%)'])
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[col_choice].dropna(), kde=True, ax=ax[0], color='skyblue')
    ax[0].set_title(f"{col_choice} 히스토그램")
    
    sns.boxplot(y=df[col_choice].dropna(), ax=ax[1], color='lightgreen')
    ax[1].set_title(f"{col_choice} 박스플롯")
    
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 3: 상관관계 및 회귀 분석
# ---------------------------------------------------------
elif menu == "3. 상관관계 및 회귀 분석":
    st.header("3. 상관관계 및 간단한 회귀 분석")
    
    st.subheader("🔗 지표 간 상관계수 (Correlation)")
    corr = df[['인지(%)', '호감(%)', '구매(%)']].corr()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(corr, use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".3f", ax=ax)
        ax.set_title("상관계수 히트맵")
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("📈 선형 회귀 분석 (Linear Regression)")
    
    x_var = st.selectbox("독립변수 (X) 선택:", ['인지(%)', '호감(%)'])
    
    sub_df = df[[x_var, '구매(%)']].dropna()
    X = sub_df[[x_var]]
    y = sub_df['구매(%)']
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    st.write(f"- **회귀 방정식**: `구매(%)` = {model.coef_[0]:.4f} * `{x_var}` + ({model.intercept_:.4f})")
    st.write(f"- **결정계수 ($R^2$)**: {r2:.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.regplot(data=sub_df, x=x_var, y='구매(%)', ax=ax, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    ax.set_title(f"{x_var}에 따른 구매(%) 회귀선")
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 4: 연도별/매체별 분석
# ---------------------------------------------------------
elif menu == "4. 연도별/매체별 분석":
    st.header("4. 연도별 및 매체 구분별 비교 분석")
    
    st.subheader("📅 연도별 평균 지표 비교")
    yearly_avg = df.groupby('연도')[['인지(%)', '호감(%)', '구매(%)']].mean()
    st.dataframe(yearly_avg, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    yearly_avg.plot(kind='bar', ax=ax)
    ax.set_title("연도별 평균 인지 / 호감 / 구매 (%)")
    ax.set_ylabel("비율 (%)")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("🏷️ 매체 구분별 평균 지표")
    category_avg = df.groupby('구분')[['인지(%)', '호감(%)', '구매(%)']].mean().sort_values(by='구매(%)', ascending=False)
    st.dataframe(category_avg, use_container_width=True)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    category_avg.plot(kind='barh', ax=ax2)
    ax2.set_title("매체 구분별 평균 인지 / 호감 / 구매 (%)")
    ax2.set_xlabel("비율 (%)")
    ax2.invert_yaxis()
    st.pyplot(fig2)
