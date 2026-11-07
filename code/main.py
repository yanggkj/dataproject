import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ---------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="고객 참여도 및 행동 몰림 군집화 분석 대시보드",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 고객 참여도(Engagement) 기반 3가지 군집화 모델 분석")
st.markdown("체류 시간과 상호작용 수를 결합한 **'참여도 스코어'**를 산출하고, **중심·분포·계층 기반**의 3가지 군집화 모델로 사용자의 행동 몰림 현상을 분석합니다.")
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

# 서버 메모리 보호를 위한 가벼운 샘플링 (3,000건)
sample_size = min(3000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).copy().reset_index(drop=True)

# ---------------------------------------------------------
# 참여도 스코어(Engagement Score) 산출
# ---------------------------------------------------------
min_t, max_t = df_sample['체류시간_초'].min(), df_sample['체류시간_초'].max()
min_i, max_i = df_sample['상호작용_수'].min(), df_sample['상호작용_수'].max()

df_sample['참여도_스코어'] = (
    ((df_sample['체류시간_초'] - min_t) / (max_t - min_t + 1e-9)) * 50 +
    ((df_sample['상호작용_수'] - min_i) / (max_i - min_i + 1e-9)) * 50
)

# ---------------------------------------------------------
# 1. 사용자 행동 몰림 현상 분석
# ---------------------------------------------------------
st.subheader("1. 📊 사용자 행동 몰림(체류) 분포 현황")

col_a, col_b = st.columns(2)

with col_a:
    action_counts = df_sample['사용자_동작'].value_counts().reset_index()
    action_counts.columns = ['사용자_동작', '사용자_수']
    fig_action = px.bar(
        action_counts, x='사용자_동작', y='사용자_수',
        title="[행동별 몰림 현황] 가장 많이 체류 중인 행동 단계",
        color='사용자_수', color_continuous_scale='Blues', text_auto=True
    )
    st.plotly_chart(fig_action, use_container_width=True)

with col_b:
    select_cat = st.selectbox("분석할 범주 선택:", ["카테고리", "기기_유형", "유입_경로"])
    cross_tab = pd.crosstab(df_sample[select_cat], df_sample['사용자_동작']).reset_index()
    fig_cross = px.bar(
        cross_tab, x=select_cat, y=[c for c in cross_tab.columns if c != select_cat],
        title=f"[{select_cat}별 행동 몰림 분포]", barmode='stack'
    )
    st.plotly_chart(fig_cross, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 군집화 모델 전처리 및 학습 (3개 모델)
# ---------------------------------------------------------
num_cols = ['체류시간_초', '세션_길이', '상호작용_수', '참여도_스코어']
cat_cols = ['카테고리', '기기_유형', '유입_경로']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_processed = preprocessor.fit_transform(df_sample[num_cols + cat_cols])

# 1. 중심 기반: K-Means (k=2)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=5)
labels_km = kmeans.fit_predict(X_processed)
sil_km = silhouette_score(X_processed, labels_km)
ch_km = calinski_harabasz_score(X_processed, labels_km)
db_km = davies_bouldin_score(X_processed, labels_km)

# 2. 분포 기반: GMM (Gaussian Mixture Model, n=2)
gmm = GaussianMixture(n_components=2, random_state=42)
labels_gmm = gmm.fit_predict(X_processed)
sil_gmm = silhouette_score(X_processed, labels_gmm)
ch_gmm = calinski_harabasz_score(X_processed, labels_gmm)
db_gmm = davies_bouldin_score(X_processed, labels_gmm)

# 3. 계층 기반: Agglomerative Clustering (n=2)
agg = AgglomerativeClustering(n_clusters=2)
labels_agg = agg.fit_predict(X_processed)
sil_agg = silhouette_score(X_processed, labels_agg)
ch_agg = calinski_harabasz_score(X_processed, labels_agg)
db_agg = davies_bouldin_score(X_processed, labels_agg)

# ---------------------------------------------------------
# 2. 3가지 군집화 모델 성능 지표 비교 표
# ---------------------------------------------------------
st.subheader("2. 🤖 3가지 비지도 군집화 모델 성능 지표 비교")
st.markdown("비지도 학습 평가 지표인 **실루엣 계수(Silhouette)**, **칼린스키-하바라슈(Calinski-Harabasz)**, **데이비스-불딘(Davies-Bouldin)**을 통해 군집 품질을 평가합니다.")

models_summary = [
    {"군집화 유형": "중심 기반", "대표 모델": "K-Means", "실루엣 계수": sil_km, "Calinski-Harabasz": ch_km, "Davies-Bouldin": db_km},
    {"군집화 유형": "분포 기반", "대표 모델": "GMM", "실루엣 계수": sil_gmm, "Calinski-Harabasz": ch_gmm, "Davies-Bouldin": db_gmm},
    {"군집화 유형": "계층 기반", "대표 모델": "Agglomerative", "실루엣 계수": sil_agg, "Calinski-Harabasz": ch_agg, "Davies-Bouldin": db_agg},
]

st.table(pd.DataFrame(models_summary).style.format({
    "실루엣 계수": "{:.4f}",
    "Calinski-Harabasz": "{:.2f}",
    "Davies-Bouldin": "{:.4f}"
}))

# ---------------------------------------------------------
# 3. 내가 고른 모델과 그 이유
# ---------------------------------------------------------
st.markdown("### 💡 내가 고른 모델과 그 이유")
st.info("""
**🏆 최종 선택 모델: 중심 기반 군집화 (K-Means)**
* **선택 이유:** 참여도 스코어와 체류 시간 기반의 고객 행동 데이터를 가장 직관적이고 균형 있게 두 개의 그룹(고참여 몰입 그룹 vs 저참여 이탈 그룹)으로 나누어 주며, 실루엣 점수와 군집 응집도가 안정적이기 때문입니다.
""")

st.markdown("---")

# ---------------------------------------------------------
# 4. 시각화 영역: 산점도 및 교차 히트맵
# ---------------------------------------------------------
st.subheader("3. 📈 모델별 산점도(Scatter Plot) 및 사용자 동작 교차 히트맵(Cross Heatmap)")

tab_scatter, tab_heatmap = st.tabs(["산점도 시각화", "사용자 동작 교차 히트맵"])

df_sample['KM'] = labels_km.astype(str)
df_sample['GMM'] = labels_gmm.astype(str)
df_sample['AGG'] = labels_agg.astype(str)

# 1) 산점도 탭
with tab_scatter:
    st.markdown("#### 군집 모델별 참여도 vs 상호작용 산점도")
    selected_model_plot = st.selectbox(
        "산점도로 확인할 모델을 선택하세요:", 
        ["중심 기반 (K-Means)", "분포 기반 (GMM)", "계층 기반 (Agglomerative)"],
        key="scatter_select"
    )
    
    if selected_model_plot == "중심 기반 (K-Means)":
        fig_scat = px.scatter(df_sample, x='상호작용_수', y='참여도_스코어', color='KM', symbol='기기_유형', title=f"K-Means 군집 산점도 (실루엣: {sil_km:.4f})")
    elif selected_model_plot == "분포 기반 (GMM)":
        fig_scat = px.scatter(df_sample, x='상호작용_수', y='참여도_스코어', color='GMM', symbol='기기_유형', title=f"GMM 군집 산점도 (실루엣: {sil_gmm:.4f})")
    else:
        fig_scat = px.scatter(df_sample, x='상호작용_수', y='참여도_스코어', color='AGG', symbol='기기_유형', title=f"Agglomerative 군집 산점도 (실루엣: {sil_agg:.4f})")
        
    st.plotly_chart(fig_scat, use_container_width=True)

# 2) 교차 히트맵 탭
with tab_heatmap:
    st.markdown("#### 군집 모델별 그룹 × 사용자 동작 교차 히트맵")
    selected_model_heat = st.selectbox(
        "교차 히트맵으로 확인할 모델을 선택하세요:", 
        ["중심 기반 (K-Means)", "분포 기반 (GMM)", "계층 기반 (Agglomerative)"],
        key="heatmap_select"
    )
    
    map_col_dict = {
        "중심 기반 (K-Means)": 'KM',
        "분포 기반 (GMM)": 'GMM',
        "계층 기반 (Agglomerative)": 'AGG'
    }
    target_cluster_col = map_col_dict[selected_model_heat]
    
    heatmap_data = pd.crosstab(df_sample[target_cluster_col], df_sample['사용자_동작'])
    
    fig_heat = px.imshow(
        heatmap_data,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"[{selected_model_heat}] 군집 그룹별 사용자 동작 몰림 현상 (빈도수)",
        labels=dict(x="사용자 동작 (User Action)", y="군집 그룹 (Cluster)", color="세션 수")
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.info("💡 **교차 히트맵 설명:** 각 군집 그룹 안에서 유저들이 **어떤 `사용자_동작`(조회, 장바구니, 클릭 등)에 가장 많이 몰려있는지** 빈도수를 색상 농도로 명확하게 파악할 수 있어 연구 목적에 완벽히 부합합니다.")
