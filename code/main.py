import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, silhouette_score

# ---------------------------------------------------------
# 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="사용자 행동 체류 및 몰림 현황 분석 대시보드",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 어느 행동에 사용자가 몰려있는가? (행동 체류 & 로지스틱 회귀 분석)")
st.markdown("카테고리, 기기 유형 등 범주형 특성을 반영하여 **사용자가 가장 많이 체류/집중하는 행동 단계**를 파악하고, **다항 로지스틱 회귀** 및 **3개 군집화 모델**을 종합 비교합니다.")
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

st.sidebar.header("📁 데이터 설정")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드 (선택)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

if df is None:
    st.error("❌ 데이터 파일을 찾을 수 없습니다.")
    st.info("`data/processed_retail_user_behavior_korean.csv` 경로에 파일이 있는지 확인하거나, 사이드바에서 업로드해 주세요.")
    st.stop()

# ---------------------------------------------------------
# 1. 사용자가 몰려 있는 행동(체류) 분석 시각화
# ---------------------------------------------------------
st.subheader("1. 📊 사용자 행동 몰림(체류) 분포 현황")

col_a, col_b = st.columns(2)

with col_a:
    action_counts = df['사용자_동작'].value_counts().reset_index()
    action_counts.columns = ['사용자_동작', '사용자_수']
    fig_action = px.bar(
        action_counts,
        x='사용자_동작',
        y='사용자_수',
        title="[행동별 몰림 현황] 가장 많은 사용자가 체류 중인 행동 단계",
        color='사용자_수',
        color_continuous_scale='Blues',
        text_auto=True
    )
    st.plotly_chart(fig_action, use_container_width=True)

with col_b:
    select_cat = st.selectbox("분석할 범주 선택:", ["카테고리", "기기_유형", "유입_경로", "채널"])
    cross_tab = pd.crosstab(df[select_cat], df['사용자_동작']).reset_index()
    fig_cross = px.bar(
        cross_tab,
        x=select_cat,
        y=[c for c in cross_tab.columns if c != select_cat],
        title=f"[{select_cat}별 행동 몰림 분포] 어떤 속성에서 특정 행동이 많이 발생하는가?",
        barmode='stack'
    )
    st.plotly_chart(fig_cross, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 모델링 전처리 (10,000건 샘플링)
# ---------------------------------------------------------
num_cols = ['체류시간_초', '세션_길이', '상호작용_수', '가격', '이벤트_인덱스']
cat_cols = ['카테고리', '브랜드', '채널', '기기_유형', '지역', '유입_경로']
target_col = '전환_여부'

sample_size = min(10000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).copy().reset_index(drop=True)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_processed = preprocessor.fit_transform(df_sample[num_cols + cat_cols])
y_true = df_sample[target_col]

def align_labels(y_true, y_pred):
    acc1 = accuracy_score(y_true, y_pred)
    acc2 = accuracy_score(y_true, 1 - y_pred)
    return (1 - y_pred) if acc2 > acc1 else y_pred

# 1. 지도학습: 다항/이진 로지스틱 회귀 (Logistic Regression)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_processed, y_true)
labels_lr = log_reg.predict(X_processed)
sil_lr = silhouette_score(X_processed, labels_lr) if len(set(labels_lr)) > 1 else 0.0

# 2. 중심 기반 (K-Means)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km = align_labels(y_true, kmeans.fit_predict(X_processed))
sil_km = silhouette_score(X_processed, labels_km)

# 3. 밀도 기반 (DBSCAN)
dbscan = DBSCAN(eps=2.0, min_samples=15)
labels_db_raw = dbscan.fit_predict(X_processed)
labels_db = align_labels(y_true, np.where(labels_db_raw == -1, 0, np.where(labels_db_raw > 0, 1, 0)))
sil_db = silhouette_score(X_processed, labels_db_raw) if len(set(labels_db_raw)) > 1 else 0.0

# 4. 분포 기반 (GMM)
gmm = GaussianMixture(n_components=2, random_state=42)
labels_gmm = align_labels(y_true, gmm.fit_predict(X_processed))
sil_gmm = silhouette_score(X_processed, labels_gmm)

# ---------------------------------------------------------
# 2. 모델별 개별 지표 카드 및 종합 비교 표
# ---------------------------------------------------------
st.subheader("2. 🤖 지도학습(로지스틱 회귀) & 군집화 모델 지표 비교")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown("### 📍 로지스틱 회귀")
    st.metric("정확도 (Accuracy)", f"{accuracy_score(y_true, labels_lr):.4f}")
    st.metric("정밀도 (Precision)", f"{precision_score(y_true, labels_lr, zero_division=0):.4f}")
    st.metric("재현율 (Recall)", f"{recall_score(y_true, labels_lr, zero_division=0):.4f}")
    st.metric("실루엣 계수", f"{sil_lr:.4f}")

with col_m2:
    st.markdown("### 📍 중심 기반 (K-Means)")
    st.metric("정확도 (Accuracy)", f"{accuracy_score(y_true, labels_km):.4f}")
    st.metric("정밀도 (Precision)", f"{precision_score(y_true, labels_km, zero_division=0):.4f}")
    st.metric("재현율 (Recall)", f"{recall_score(y_true, labels_km, zero_division=0):.4f}")
    st.metric("실루엣 계수", f"{sil_km:.4f}")

with col_m3:
    st.markdown("### 📍 밀도 기반 (DBSCAN)")
    st.metric("정확도 (Accuracy)", f"{accuracy_score(y_true, labels_db):.4f}")
    st.metric("정밀도 (Precision)", f"{precision_score(y_true, labels_db, zero_division=0):.4f}")
    st.metric("재현율 (Recall)", f"{recall_score(y_true, labels_db, zero_division=0):.4f}")
    st.metric("실루엣 계수", f"{sil_db:.4f}")

with col_m4:
    st.markdown("### 📍 분포 기반 (GMM)")
    st.metric("정확도 (Accuracy)", f"{accuracy_score(y_true, labels_gmm):.4f}")
    st.metric("정밀도 (Precision)", f"{precision_score(y_true, labels_gmm, zero_division=0):.4f}")
    st.metric("재현율 (Recall)", f"{recall_score(y_true, labels_gmm, zero_division=0):.4f}")
    st.metric("실루엣 계수", f"{sil_gmm:.4f}")

st.markdown("#### 📋 모델별 종합 성능 비교 표")
models_summary = [
    {
        "모델 유형": "지도학습 (Supervised)",
        "대표 모델": "로지스틱 회귀 (Logistic Regression)",
        "정확도": accuracy_score(y_true, labels_lr),
        "정밀도": precision_score(y_true, labels_lr, zero_division=0),
        "재현율": recall_score(y_true, labels_lr, zero_division=0),
        "실루엣 계수": sil_lr,
        "특성": "기기·카테고리 영향력(가중치) 직접 해석 가능"
    },
    {
        "모델 유형": "중심 기반 (Center-based)",
        "대표 모델": "K-Means",
        "정확도": accuracy_score(y_true, labels_km),
        "정밀도": precision_score(y_true, labels_km, zero_division=0),
        "재현율": recall_score(y_true, labels_km, zero_division=0),
        "실루엣 계수": sil_km,
        "특성": "중심점 기반 행동 체류 집단 구형 분리"
    },
    {
        "모델 유형": "밀도 기반 (Density-based)",
        "대표 모델": "DBSCAN",
        "정확도": accuracy_score(y_true, labels_db),
        "정밀도": precision_score(y_true, labels_db, zero_division=0),
        "재현율": recall_score(y_true, labels_db, zero_division=0),
        "실루엣 계수": sil_db,
        "특성": "고밀도 체류 영역 및 이상 행동 감지"
    },
    {
        "모델 유형": "분포 기반 (Distribution-based)",
        "대표 모델": "GMM",
        "정확도": accuracy_score(y_true, labels_gmm),
        "정밀도": precision_score(y_true, labels_gmm, zero_division=0),
        "재현율": recall_score(y_true, labels_gmm, zero_division=0),
        "실루엣 계수": sil_gmm,
        "특성": "확률 분포 기반 유연한 행동 패턴 군집"
    }
]

metrics_df = pd.DataFrame(models_summary)
st.table(metrics_df.style.format({
    "정확도": "{:.4f}",
    "정밀도": "{:.4f}",
    "재현율": "{:.4f}",
    "실루엣 계수": "{:.4f}"
}))

# ---------------------------------------------------------
# 3. '내가 고른 모델과 그 이유' 글상자
# ---------------------------------------------------------
st.markdown("### 💡 내가 고른 모델과 그 이유")

st.info("""
**🏆 최종 선택 모델: 로지스틱 회귀 (Logistic Regression)**

* **선택 이유:**
  1. **범주형 변수(기기, 카테고리)의 영향력 직접 해석:** 로지스틱 회귀는 기기 유형이나 상품 카테고리와 같은 범주형 변수가 사용자의 특정 행동 체류 및 전환 확률에 미치는 영향(회귀 계수 및 오즈비)을 직관적인 수치로 알려줍니다.
  2. **높은 분류 정확도 및 안정성:** 군집화 알고리즘에 비해 지도학습 특성을 활용해 사용자가 어떤 상태에 반응하는지 가장 뛰어난 정확도 성능을 보입니다.
  3. **비지도 군집화(K-Means)와의 보완:** 사용자 분류 자체에는 K-Means가 직관적이지만, "어떤 기기나 카테고리가 해당 행동에 몰리게 하는가?"라는 질문에는 로지스틱 회귀가 가장 명확한 수치적 근거를 제시합니다.
""")

st.markdown("---")

# ---------------------------------------------------------
# 4. 각 모델별 군집 및 분류 구조 시각화
# ---------------------------------------------------------
st.subheader("3. 📈 모델별 예측 및 군집 구조 시각화")

tab_lr, tab_km, tab_db, tab_gmm = st.tabs([
    "로지스틱 회귀 예측", 
    "중심 기반 (K-Means)", 
    "밀도 기반 (DBSCAN)", 
    "분포 기반 (GMM)"
])

df_sample['로지스틱_예측'] = labels_lr.astype(str)
df_sample['KMeans_군집'] = labels_km.astype(str)
df_sample['DBSCAN_군집'] = labels_db_raw.astype(str)
df_sample['GMM_군집'] = labels_gmm.astype(str)

with tab_lr:
    st.markdown("#### [로지스틱 회귀] 기기 유형별 체류시간 및 상호작용 분류 결과")
    fig_lr = px.scatter(
        df_sample,
        x='상호작용_수',
        y='체류시간_초',
        color='로지스틱_예측',
        symbol='기기_유형',
        title="로지스틱 회귀 분류 결과 (범주형 특성 결합)",
        color_discrete_map={'0': '#EF553B', '1': '#00CC96'},
        opacity=0.7
    )
    st.plotly_chart(fig_lr, use_container_width=True)

with tab_km:
    st.markdown("#### [K-Means] 체류시간 vs 상호작용 수 군집 구조")
    fig_km = px.scatter(
        df_sample,
        x='상호작용_수',
        y='체류시간_초',
        color='KMeans_군집',
        symbol='기기_유형',
        title=f"K-Means 군집 결과 (실루엣: {sil_km:.4f})",
        color_discrete_map={'0': '#EF553B', '1': '#636EFA'},
        opacity=0.7
    )
    st.plotly_chart(fig_km, use_container_width=True)

with tab_db:
    st.markdown("#### [DBSCAN] 밀도 기반 군집 및 노이즈 구조")
    fig_db = px.scatter(
        df_sample,
        x='상호작용_수',
        y='체류시간_초',
        color='DBSCAN_군집',
        title=f"DBSCAN 군집 결과 (실루엣: {sil_db:.4f})",
        opacity=0.7
    )
    st.plotly_chart(fig_db, use_container_width=True)

with tab_gmm:
    st.markdown("#### [GMM] 확률 분포 기반 군집 구조")
    fig_gmm = px.scatter(
        df_sample,
        x='상호작용_수',
        y='체류시간_초',
        color='GMM_군집',
        symbol='기기_유형',
        title=f"GMM 군집 결과 (실루엣: {sil_gmm:.4f})",
        color_discrete_map={'0': '#AB63FA', '1': '#FFA15A'},
        opacity=0.7
    )
    st.plotly_chart(fig_gmm, use_container_width=True)
