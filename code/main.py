import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="차원 축소 및 프로파일 히트맵 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

st.title("📊 차원 축소(PCA/t-SNE) 및 프로파일 히트맵 분석 대시보드")
st.markdown("수치형 및 범주형 속성을 결합하여 고차원 데이터를 2차원으로 축소하고, 사용자 행동별 특성을 프로파일 히트맵으로 시각화합니다.")
st.markdown("---")

# ---------------------------------------------------------
# 데이터 로드 (캐싱 적용)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    for path in ['data/processed_retail_user_behavior_korean.csv', 'processed_retail_user_behavior_korean.csv']:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    return None

df = load_data()
if df is None:
    st.error("❌ 데이터 파일을 찾을 수 없습니다 (`processed_retail_user_behavior_korean.csv`).")
    st.stop()

# 서버 메모리 보호 및 속도를 위한 가벼운 샘플링 (2,000건)
sample_n = min(2000, len(df))
df_sample = df.sample(n=sample_n, random_state=42).copy().reset_index(drop=True)

# ---------------------------------------------------------
# 전처리 및 차원 축소 (PCA & t-SNE) 수행
# ---------------------------------------------------------
num_cols = ['체류시간_초', '세션_길이', '상호작용_수', '가격', '이벤트_인덱스']
cat_cols = ['카테고리', '기기_유형', '유입_경로', '사용자_동작']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_processed = preprocessor.fit_transform(df_sample[num_cols + cat_cols])

# 1. PCA (주성분 분석)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_processed)
df_sample['PCA1'] = X_pca[:, 0]
df_sample['PCA2'] = X_pca[:, 1]

# 2. t-SNE (데이터 샘플 수에 맞춰 perplexity 자동 조절로 에러 방지)
n_samples = X_processed.shape[0]
dynamic_perplexity = min(30, max(5, n_samples // 3))

tsne = TSNE(n_components=2, perplexity=dynamic_perplexity, random_state=42, n_iter=300)
X_tsne = tsne.fit_transform(X_processed)
df_sample['t-SNE1'] = X_tsne[:, 0]
df_sample['t-SNE2'] = X_tsne[:, 1]

# 3. K-Means 군집 라벨 생성 (시각화 색상 구분용)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=5)
df_sample['군집'] = kmeans.fit_predict(X_processed).astype(str)

# ---------------------------------------------------------
# 1. 차원 축소 시각화 탭 (PCA / t-SNE / 산점도)
# ---------------------------------------------------------
st.subheader("1. 🌐 차원 축소 및 군집 산점도 (PCA & t-SNE)")

tab_pca, tab_tsne, tab_scatter = st.tabs([
    "PCA (주성분 분석)", 
    "t-SNE (비선형 매니폴드)", 
    "기본 산점도 (체류시간 vs 상호작용)"
])

with tab_pca:
    st.markdown("#### PCA 2차원 투영 시각화")
    fig_pca = px.scatter(
        df_sample, x='PCA1', y='PCA2', color='군집', symbol='기기_유형',
        title="PCA 차원 축소 결과 (고차원 속성을 2개 주성분으로 압축)",
        opacity=0.8
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    st.info("💡 **PCA 설명:** 데이터의 분산을 최대한 보존하면서 여러 변수들을 직교하는 2개의 주성분(PCA1, PCA2)으로 압축하여 전체적인 군집 분포를 보여줍니다.")

with tab_tsne:
    st.markdown(f"#### t-SNE 2차원 군집화 시각화 (Perplexity: {dynamic_perplexity})")
    fig_tsne = px.scatter(
        df_sample, x='t-SNE1', y='t-SNE2', color='군집', symbol='기기_유형',
        title="t-SNE 차원 축소 결과 (비선형 군집 구조 시각화)",
        opacity=0.8
    )
    st.plotly_chart(fig_tsne, use_container_width=True)
    st.info("💡 **t-SNE 설명:** 고차원 공간에서의 데이터 간 거리를 보존하며 2차원에 비선형적으로 배치하여, 밀집된 사용자 그룹과 이탈 패턴을 뚜렷하게 군집화합니다.")

with tab_scatter:
    st.markdown("#### 기본 사용자 행동 산점도")
    fig_sc = px.scatter(
        df_sample, x='상호작용_수', y='체류시간_초', color='사용자_동작', symbol='기기_유형',
        title="체류시간 vs 상호작용 수 산점도 (사용자 동작별 분포)",
        opacity=0.8
    )
    st.plotly_chart(fig_sc, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 2. 프로파일 히트맵 (Profile Heatmap)
# ---------------------------------------------------------
st.subheader("2. 🔥 프로파일 히트맵 (Profile Heatmap)")
st.markdown("사용자 동작별, 카테고리별 주요 수치 지표의 평균값 비교를 통해 그룹 특성을 파악합니다.")

col_h1, col_h2 = st.columns(2)

with col_h1:
    st.markdown("#### [동작별] 수치형 평균 프로파일 히트맵")
    profile_action = df.groupby('사용자_동작')[num_cols].mean()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(profile_action, annot=True, fmt=".1f", cmap="Blues", cbar=True, ax=ax)
    ax.set_title("사용자 동작에 따른 수치형 지표 평균")
    st.pyplot(fig)

with col_h2:
    st.markdown("#### [기기 유형별] 주요 행동 지표 프로파일 히트맵")
    profile_device = df.groupby('기기_유형')[num_cols].mean()
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(profile_device, annot=True, fmt=".1f", cmap="Oranges", cbar=True, ax=ax2)
    ax2.set_title("기기 유형에 따른 수치형 지표 평균")
    st.pyplot(fig2)

st.markdown("---")
st.success("✅ t-SNE perplexity 동적 조절 코드가 반영되어 에러 없이 정상 구동됩니다.")
