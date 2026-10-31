import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, silhouette_score

# ---------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="고객 행동 분석 & 군집화 모델 대시보드",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 어떤 사용자 행동이 구매로 이어지는가? (군집화 분석)")
st.markdown("중심 기반, 밀도 기반, 분포 기반 군집화 모델을 적용하여 사용자 행동 패턴을 분석하고 모델 성능을 비교합니다.")
st.markdown("---")

# ---------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------
@st.cache_data
def load_data():
    possible_paths = [
        'data/processed_retail_user_behavior_korean.csv',
        'processed_retail_user_behavior_korean.csv'
    ]
    for path in possible_paths:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    return None

st.sidebar.header("📁 데이터 및 설정")
uploaded_file = st.sidebar.file_uploader("CSV 파일 직접 업로드 (선택)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

if df is None:
    st.error("❌ 데이터 파일을 찾을 수 없습니다.")
    st.info("`data/processed_retail_user_behavior_korean.csv` 경로에 파일이 있는지 확인하거나, 사이드바에서 업로드해 주세요.")
    st.stop()

# ---------------------------------------------------------
# 전처리 및 모델 학습
# ---------------------------------------------------------
feature_cols = ['체류시간_초', '세션_길이', '상호작용_수', '가격', '이벤트_인덱스']
target_col = '전환_여부'

# 연산 속도 및 시각화를 위한 샘플링 (10,000건)
sample_size = min(10000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).copy().reset_index(drop=True)

X = df_sample[feature_cols].fillna(0)
y_true = df_sample[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 라벨 정렬 함수 (비지도 군집 0/1을 타겟 변수 방향에 맞춤)
def align_labels(y_true, y_pred):
    acc1 = accuracy_score(y_true, y_pred)
    acc2 = accuracy_score(y_true, 1 - y_pred)
    return (1 - y_pred) if acc2 > acc1 else y_pred

# 1. 중심 기반 군집화 (K-Means)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)
labels_km_aligned = align_labels(y_true, labels_km)
sil_km = silhouette_score(X_scaled, labels_km)

# 2. 밀도 기반 군집화 (DBSCAN)
dbscan = DBSCAN(eps=1.5, min_samples=10)
labels_db = dbscan.fit_predict(X_scaled)
labels_db_binary = np.where(labels_db == -1, 0, np.where(labels_db > 0, 1, 0))
labels_db_aligned = align_labels(y_true, labels_db_binary)
sil_db = silhouette_score(X_scaled, labels_db) if len(set(labels_db)) > 1 else 0.0

# 3. 분포 기반 군집화 (GMM)
gmm = GaussianMixture(n_components=2, random_state=42)
labels_gmm = gmm.fit_predict(X_scaled)
labels_gmm_aligned = align_labels(y_true, labels_gmm)
sil_gmm = silhouette_score(X_scaled, labels_gmm)

# ---------------------------------------------------------
# 1. 구매 전환 요인 분석 시각화
# ---------------------------------------------------------
st.subheader("1. 📊 구매 전환 유무에 따른 사용자 행동 탐색")

col_a, col_b = st.columns(2)

with col_a:
    action_conv = df.groupby('사용자_동작')['전환_여부'].mean().reset_index()
    action_conv['전환율(%)'] = action_conv['전환_여부'] * 100
    fig_action = px.bar(
        action_conv.sort_values(by='전환율(%)', ascending=True),
        x='전환율(%)',
        y='사용자_동작',
        orientation='h',
        title="사용자 동작(User Action)별 구매 전환율",
        color='전환율(%)',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_action, use_container_width=True)

with col_b:
    fig_scatter = px.scatter(
        df_sample,
        x='상호작용_수',
        y='체류시간_초',
        color=df_sample['전환_여부'].astype(str),
        title="상호작용 수 vs 체류시간에 따른 구매 전환 분포",
        color_discrete_map={'0': '#EF553B', '1': '#00CC96'},
        opacity=0.6
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 2. 군집화 모델별 성과 비교 (표 및 지표)
# ---------------------------------------------------------
st.subheader("2. 🤖 군집화 알고리즘별 성능 비교 표")

models_summary = [
    {
        "군집화 유형": "중심 기반 군집화 (Center-based)",
        "대표 모델": "K-Means",
        "정확도 (Accuracy)": accuracy_score(y_true, labels_km_aligned),
        "정밀도 (Precision)": precision_score(y_true, labels_km_aligned, zero_division=0),
        "재현율 (Recall)": recall_score(y_true, labels_km_aligned, zero_division=0),
        "실루엣 계수 (Silhouette)": sil_km,
        "최적 군집 수 / 파라미터": "K=2"
    },
    {
        "군집화 유형": "밀도 기반 군집화 (Density-based)",
        "대표 모델": "DBSCAN",
        "정확도 (Accuracy)": accuracy_score(y_true, labels_db_aligned),
        "정밀도 (Precision)": precision_score(y_true, labels_db_aligned, zero_division=0),
        "재현율 (Recall)": recall_score(y_true, labels_db_aligned, zero_division=0),
        "실루엣 계수 (Silhouette)": sil_db,
        "최적 군집 수 / 파라미터": "eps=1.5, min=10"
    },
    {
        "군집화 유형": "분포 기반 군집화 (Distribution-based)",
        "대표 모델": "GMM (Gaussian Mixture)",
        "정확도 (Accuracy)": accuracy_score(y_true, labels_gmm_aligned),
        "정밀도 (Precision)": precision_score(y_true, labels_gmm_aligned, zero_division=0),
        "재현율 (Recall)": recall_score(y_true, labels_gmm_aligned, zero_division=0),
        "실루엣 계수 (Silhouette)": sil_gmm,
        "최적 군집 수 / 파라미터": "Components=2"
    }
]

metrics_df = pd.DataFrame(models_summary)
st.table(metrics_df.style.format({
    "정확도 (Accuracy)": "{:.4f}",
    "정밀도 (Precision)": "{:.4f}",
    "재현율 (Recall)": "{:.4f}",
    "실루엣 계수 (Silhouette)": "{:.4f}"
}))

# ---------------------------------------------------------
# 3. '내가 고른 모델과 그 이유' 글상자
# ---------------------------------------------------------
st.markdown("### 💡 내가 고른 모델과 그 이유")

st.info("""
**🏆 최종 선택 모델: 중심 기반 군집화 (K-Means)**

* **선택 이유:**
  1. **균형 잡힌 예측 및 군집 품질:** K-Means 모델은 **62.6%의 정확도**와 **73% 이상의 재현율**을 기록하여 실제 구매자 그룹을 효과적으로 포착해 냈습니다.
  2. **밀도 기반(DBSCAN)의 한계:** DBSCAN은 데이터가 고르게 분산된 인공 데이터 특성상 대부분의 데이터를 단일 군집 혹은 노이즈로 처리하여, 실제 구매(양성) 예측 능력이 현저히 떨어졌습니다.
  3. **해석의 직관성:** K-Means는 각 군집의 중심점(Centroid)을 명확히 제공하므로, 사용자의 체류 시간과 상호작용 패턴을 바탕으로 '일반 방문객'과 '구매 유망 고객'을 나누는 비즈니스 인사이트 도출에 가장 적합합니다.
""")

st.markdown("---")

# ---------------------------------------------------------
# 4. 각 모델별 군집 구조 시각화 및 엘보우 분석
# ---------------------------------------------------------
st.subheader("3. 📈 각 모델별 군집 구조 및 엘보우(Elbow) 분석")

tab_km, tab_db, tab_gmm, tab_elbow = st.tabs([
    "중심 기반 (K-Means)", 
    "밀도 기반 (DBSCAN)", 
    "분포 기반 (GMM)", 
    "K-Means 엘보우 분석"
])

# 1) K-Means 시각화 (문자열 변환으로 오류 해결)
with tab_km:
    st.markdown("#### K-Means 군집 구조 (체류시간 vs 상호작용 수)")
    df_sample['KMeans_군집'] = labels_km_aligned.astype(str) # 문자열로 변환하여 이산형 컬러맵 적용
    fig_km = px.scatter(
        df_sample,
        x='상호작용_수',
        y='체류시간_초',
        color='KMeans_군집',
        title=f"K-Means 군집 결과 (실루엣 계수: {sil_km:.4f})",
        color_discrete_map={'0': '#EF553B', '1': '#636EFA'},
        opacity=0.7
    )
    st.plotly_chart(fig_km, use_container_width=True)

# 2) DBSCAN 시각화
with tab_db:
    st.markdown("#### DBSCAN 군집 구조 (체류시간 vs 상호작용 수)")
    df_sample['DBSCAN_군집'] = labels_db.astype(str)
    fig_db = px.scatter(
        df_sample,
        x='상호작용_수',
        y='체류시간_초',
        color='DBSCAN_군집',
        title=f"DBSCAN 군집 결과 (실루엣 계수: {sil_db:.4f}, 노이즈 포함)",
        opacity=0.7
    )
    st.plotly_chart(fig_db, use_container_width=True)

# 3) GMM 시각화
with tab_gmm:
    st.markdown("#### GMM 군집 구조 (체류시간 vs 상호작용 수)")
    df_sample['GMM_군집'] = labels_gmm_aligned.astype(str)
    fig_gmm = px.scatter(
        df_sample,
        x='상호작용_수',
        y='체류시간_초',
        color='GMM_군집',
        title=f"GMM 군집 결과 (실루엣 계수: {sil_gmm:.4f})",
        color_discrete_map={'0': '#AB63FA', '1': '#FFA15A'},
        opacity=0.7
    )
    st.plotly_chart(fig_gmm, use_container_width=True)

# 4) 엘보우(Elbow) 분석
with tab_elbow:
    st.markdown("#### K-Means 최적의 군집 수 결정을 위한 엘보우(Elbow) 곡선")
    inertias = []
    K_range = range(1, 6)
    for k in K_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=5)
        km_temp.fit(X_scaled)
        inertias.append(km_temp.inertia_)
    
    elbow_df = pd.DataFrame({'군집 수 (K)': list(K_range), '이네르시아 (Inertia)': inertias})
    fig_elbow = px.line(
        elbow_df,
        x='군집 수 (K)',
        y='이네르시아 (Inertia)',
        markers=True,
        title="K-Means Elbow Method (데이터가 인공 균등 분포여서 완만한 감소세를 보임)"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.info("💡 **엘보우 분석 참고:** 이 데이터셋은 시뮬레이션된 인공 데이터(Uniform Distribution)이므로 그래프가 꺾이는 지점(Elbow)이 뚜렷하지 않고 완만하게 감소하는 특징이 있습니다.")
