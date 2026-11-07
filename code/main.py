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

st.title("🎯 모델별 최적 군집 개수 탐색 및 평가 대시보드")
st.markdown("원래 데이터 파일을 기반으로 3가지 비지도 군집화 모델의 최적화 곡선과 정확한 수치를 비교합니다.")
st.markdown("---")

# ---------------------------------------------------------
# 데이터 로드 (원래 파일명 기준)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv('processed_retail_user_behavior_korean.csv')
    except FileNotFoundError:
        return None

df = load_data()
if df is None:
    st.error("❌ `processed_retail_user_behavior_korean.csv` 파일을 찾을 수 없습니다. 파일을 확인해주세요.")
    st.stop()

# 서버 메모리 보호를 위한 샘플링 (3,000건)
sample_size = min(3000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).copy().reset_index(drop=True)

# ---------------------------------------------------------
# 참여도 스코어(Engagement Score) 산출
# ---------------------------------------------------------
if '참여도_스코어' not in df_sample.columns:
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

# 전처리
num_cols = ['체류시간_초', '세션_길이', '상호작용_수', '참여도_스코어']
cat_cols = ['카테고리', '기기_유형', '유입_경로']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)
X_processed = preprocessor.fit_transform(df_sample[num_cols + cat_cols])

# 기본 K=2 학습
kmeans = KMeans(n_clusters=2, random_state=42, n_init=5).fit(X_processed)
gmm = GaussianMixture(n_components=2, random_state=42).fit(X_processed)
agg = AgglomerativeClustering(n_clusters=2).fit(X_processed)

labels_km = kmeans.labels_
labels_gmm = gmm.predict(X_processed)
labels_agg = agg.labels_

# 성능 지표
sil_km, ch_km, db_km = silhouette_score(X_processed, labels_km), calinski_harabasz_score(X_processed, labels_km), davies_bouldin_score(X_processed, labels_km)
sil_gmm, ch_gmm, db_gmm = silhouette_score(X_processed, labels_gmm), calinski_harabasz_score(X_processed, labels_gmm), davies_bouldin_score(X_processed, labels_gmm)
sil_agg, ch_agg, db_agg = silhouette_score(X_processed, labels_agg), calinski_harabasz_score(X_processed, labels_agg), davies_bouldin_score(X_processed, labels_agg)

# ---------------------------------------------------------
# 2. 성능 지표 비교 표
# ---------------------------------------------------------
st.subheader("2. 🤖 3가지 비지도 군집화 모델 성능 지표 비교 (K=2 기준)")
models_summary = [
    {"군집화 유형": "중심 기반", "대표 모델": "K-Means", "실루엣 계수": sil_km, "칼린스키-하바라슈 지수": ch_km, "데이비스-불딘 지수": db_km},
    {"군집화 유형": "분포 기반", "대표 모델": "GMM", "실루엣 계수": sil_gmm, "칼린스키-하바라슈 지수": ch_gmm, "데이비스-불딘 지수": db_gmm},
    {"군집화 유형": "계층 기반", "대표 모델": "Agglomerative", "실루엣 계수": sil_agg, "칼린스키-하바라슈 지수": ch_agg, "데이비스-불딘 지수": db_agg},
]
st.table(pd.DataFrame(models_summary).style.format({
    "실루엣 계수": "{:.4f}",
    "칼린스키-하바라슈 지수": "{:.2f}",
    "데이비스-불딘 지수": "{:.4f}"
}))

# ---------------------------------------------------------
# 3. 모델별 최적 군집 개수 탐색 곡선 및 정확한 수치
# ---------------------------------------------------------
st.subheader("3. 📉 모델별 최적 군집 개수 탐색 곡선 및 상세 수치 ($K=2 \\sim 8$)")

k_range = list(range(2, 9))
inertia_list, bic_list, sil_list = [], [], []

for k in k_range:
    km_t = KMeans(n_clusters=k, random_state=42, n_init=5).fit(X_processed)
    inertia_list.append(km_t.inertia_)
    
    gmm_t = GaussianMixture(n_components=k, random_state=42).fit(X_processed)
    bic_list.append(gmm_t.bic(X_processed))
    
    agg_t = AgglomerativeClustering(n_clusters=k).fit(X_processed)
    sil_list.append(silhouette_score(X_processed, agg_t.labels_))

eval_df = pd.DataFrame({
    "군집수(K)": k_range,
    "K-Means (Inertia)": inertia_list,
    "GMM (BIC)": bic_list,
    "Agglomerative (실루엣)": sil_list
})

st.markdown("#### 📋 군집 개수별 평가 지표 상세 수치")
st.dataframe(eval_df.style.format({
    "K-Means (Inertia)": "{:,.2f}",
    "GMM (BIC)": "{:,.2f}",
    "Agglomerative (실루엣)": "{:.4f}"
}), use_container_width=True)

tab_el, tab_bic, tab_sil = st.tabs(["K-Means 엘보우 곡선", "GMM BIC 곡선", "계층 기반 실루엣 곡선"])

with tab_el:
    fig_el = px.line(eval_df, x="군집수(K)", y="K-Means (Inertia)", markers=True, title="[K-Means] 이너시아 변화 (Elbow Method)")
    st.plotly_chart(fig_el, use_container_width=True)

with tab_bic:
    fig_bic = px.line(eval_df, x="군집수(K)", y="GMM (BIC)", markers=True, title="[GMM] BIC 정보기준 곡선 (낮을수록 좋음)")
    st.plotly_chart(fig_bic, use_container_width=True)

with tab_sil:
    fig_sil = px.line(eval_df, x="군집수(K)", y="Agglomerative (실루엣)", markers=True, title="[Agglomerative] 실루엣 계수 곡선 (높을수록 좋음)")
    st.plotly_chart(fig_sil, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 4. 시각화 영역: 산점도 및 교차 히트맵
# ---------------------------------------------------------
st.subheader("4. 📈 모델별 산점도 및 사용자 동작 교차 히트맵")
tab_scatter, tab_heatmap = st.tabs(["산점도 시각화", "사용자 동작 교차 히트맵"])

df_sample['KM'] = labels_km.astype(str)
df_sample['GMM'] = labels_gmm.astype(str)
df_sample['AGG'] = labels_agg.astype(str)

with tab_scatter:
    selected_model_plot = st.selectbox("산점도 모델 선택:", ["중심 기반 (K-Means)", "분포 기반 (GMM)", "계층 기반 (Agglomerative)"], key="scat_sel")
    if selected_model_plot == "중심 기반 (K-Means)":
        fig_scat = px.scatter(df_sample, x='상호작용_수', y='참여도_스코어', color='KM', symbol='기기_유형', title="K-Means 산점도")
    elif selected_model_plot == "분포 기반 (GMM)":
        fig_scat = px.scatter(df_sample, x='상호작용_수', y='참여도_스코어', color='GMM', symbol='기기_유형', title="GMM 산점도")
    else:
        fig_scat = px.scatter(df_sample, x='상호작용_수', y='참여도_스코어', color='AGG', symbol='기기_유형', title="Agglomerative 산점도")
    st.plotly_chart(fig_scat, use_container_width=True)

with tab_heatmap:
    selected_model_heat = st.selectbox("교차 히트맵 모델 선택:", ["중심 기반 (K-Means)", "분포 기반 (GMM)", "계층 기반 (Agglomerative)"], key="heat_sel")
    map_col_dict = {"중심 기반 (K-Means)": 'KM', "분포 기반 (GMM)": 'GMM', "계층 기반 (Agglomerative)": 'AGG'}
    target_cluster_col = map_col_dict[selected_model_heat]
    
    heatmap_data = pd.crosstab(df_sample[target_cluster_col], df_sample['사용자_동작'])
    fig_heat = px.imshow(heatmap_data, text_auto=True, color_continuous_scale="Blues", title=f"[{selected_model_heat}] 그룹별 사용자 동작 몰림 빈도수")
    st.plotly_chart(fig_heat, use_container_width=True)
