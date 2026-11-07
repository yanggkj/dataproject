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
# 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="리테일 고객 행동 분석 대시보드",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 사용자 행동 몰림 현상 & 교차 히트맵 분석 대시보드")
st.markdown("로지스틱 회귀 및 3가지 군집화 모델을 활용하여 사용자 행동 체류 현황을 파악하고, **산점도**와 **동작별 교차 히트맵**으로 모델별 특성을 비교합니다.")
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
# 1. 사용자 행동 몰림 현상 분석
# ---------------------------------------------------------
st.subheader("1. 📊 사용자 행동 몰림(체류) 분포 현황")

col_a, col_b = st.columns(2)

with col_a:
    action_counts = df['사용자_동작'].value_counts().reset_index()
    action_counts.columns = ['사용자_동작', '사용자_수']
    fig_action = px.bar(
        action_counts, x='사용자_동작', y='사용자_수',
        title="[행동별 몰림 현황] 가장 많이 체류 중인 행동 단계",
        color='사용자_수', color_continuous_scale='Blues', text_auto=True
    )
    st.plotly_chart(fig_action, use_container_width=True)

with col_b:
    select_cat = st.selectbox("분석할 범주 선택:", ["카테고리", "기기_유형", "유입_경로"])
    cross_tab = pd.crosstab(df[select_cat], df['사용자_동작']).reset_index()
    fig_cross = px.bar(
        cross_tab, x=select_cat, y=[c for c in cross_tab.columns if c != select_cat],
        title=f"[{select_cat}별 행동 몰림 분포]", barmode='stack'
    )
    st.plotly_chart(fig_cross, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 모델 학습 및 전처리
# ---------------------------------------------------------
num_cols = ['체류시간_초', '세션_길이', '상호작용_수', '가격', '이벤트_인덱스']
cat_cols = ['카테고리', '기기_유형', '유입_경로']
target_col = '전환_여부'

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_processed = preprocessor.fit_transform(df_sample[num_cols + cat_cols])
y_true = df_sample[target_col]

def align_labels(y_true, y_pred):
    return (1 - y_pred) if accuracy_score(y_true, 1 - y_pred) > accuracy_score(y_true, y_pred) else y_pred

# 모델 학습
log_reg = LogisticRegression(max_iter=500, random_state=42).fit(X_processed, y_true)
labels_lr = log_reg.predict(X_processed)
sil_lr = silhouette_score(X_processed, labels_lr) if len(set(labels_lr)) > 1 else 0.0

kmeans = KMeans(n_clusters=2, random_state=42, n_init=5)
labels_km = align_labels(y_true, kmeans.fit_predict(X_processed))
sil_km = silhouette_score(X_processed, labels_km)

dbscan = DBSCAN(eps=2.0, min_samples=10)
db_raw = dbscan.fit_predict(X_processed)
labels_db = align_labels(y_true, np.where(db_raw == -1, 0, np.where(db_raw > 0, 1, 0)))
sil_db = silhouette_score(X_processed, db_raw) if len(set(db_raw)) > 1 else 0.0

gmm = GaussianMixture(n_components=2, random_state=42)
labels_gmm = align_labels(y_true, gmm.fit_predict(X_processed))
sil_gmm = silhouette_score(X_processed, labels_gmm)

# ---------------------------------------------------------
# 2. 모델별 지표 비교 표
# ---------------------------------------------------------
st.subheader("2. 🤖 모델별 성능 지표 비교")

models_summary = [
    {"모델 유형": "지도학습", "대표 모델": "로지스틱 회귀", "정확도": accuracy_score(y_true, labels_lr), "정밀도": precision_score(y_true, labels_lr, zero_division=0), "재현율": recall_score(y_true, labels_lr, zero_division=0), "실루엣 계수": sil_lr},
    {"모델 유형": "중심 기반", "대표 모델": "K-Means", "정확도": accuracy_score(y_true, labels_km), "정밀도": precision_score(y_true, labels_km, zero_division=0), "재현율": recall_score(y_true, labels_km, zero_division=0), "실루엣 계수": sil_km},
    {"모델 유형": "밀도 기반", "대표 모델": "DBSCAN", "정확도": accuracy_score(y_true, labels_db), "정밀도": precision_score(y_true, labels_db, zero_division=0), "재현율": recall_score(y_true, labels_db, zero_division=0), "실루엣 계수": sil_db},
    {"모델 유형": "분포 기반", "대표 모델": "GMM", "정확도": accuracy_score(y_true, labels_gmm), "정밀도": precision_score(y_true, labels_gmm, zero_division=0), "재현율": recall_score(y_true, labels_gmm, zero_division=0), "실루엣 계수": sil_gmm},
]

st.table(pd.DataFrame(models_summary).style.format({"정확도": "{:.4f}", "정밀도": "{:.4f}", "재현율": "{:.4f}", "실루엣 계수": "{:.4f}"}))

# ---------------------------------------------------------
# 3. 내가 고른 모델과 그 이유
# ---------------------------------------------------------
st.markdown("### 💡 내가 고른 모델과 그 이유")
st.info("""
**🏆 최종 선택 모델: 로지스틱 회귀 (Logistic Regression)**
* **선택 이유:** 기기 유형이나 카테고리 같은 범주형 속성이 사용자의 행동 체류 및 전환에 미치는 영향력을 직관적으로 수치화하여 보여주며, 가장 안정적인 분류 정확도를 제공하기 때문입니다.
""")

st.markdown("---")

# ---------------------------------------------------------
# 4. 시각화 영역: 산점도 및 교차 히트맵
# ---------------------------------------------------------
st.subheader("3. 📈 모델별 산점도(Scatter Plot) 및 사용자 동작 교차 히트맵(Cross Heatmap)")

tab_scatter, tab_heatmap = st.tabs(["산점도 시각화", "사용자 동작 교차 히트맵"])

# 1) 산점도 탭
with tab_scatter:
    st.markdown("#### 모델별 예측 및 군집 구조 산점도")
    selected_model_plot = st.selectbox(
        "산점도로 확인할 모델을 선택하세요:", 
        ["로지스틱 회귀", "중심 기반 (K-Means)", "밀도 기반 (DBSCAN)", "분포 기반 (GMM)"],
        key="scatter_select"
    )
    
    df_sample['LR'] = labels_lr.astype(str)
    df_sample['KM'] = labels_km.astype(str)
    df_sample['DB'] = db_raw.astype(str)
    df_sample['GMM'] = labels_gmm.astype(str)
    
    if selected_model_plot == "로지스틱 회귀":
        fig_scat = px.scatter(df_sample, x='상호작용_수', y='체류시간_초', color='LR', symbol='기기_유형', title="로지스틱 회귀 예측 결과 산점도")
    elif selected_model_plot == "중심 기반 (K-Means)":
        fig_scat = px.scatter(df_sample, x='상호작용_수', y='체류시간_초', color='KM', symbol='기기_유형', title=f"K-Means 군집 산점도 (실루엣: {sil_km:.4f})")
    elif selected_model_plot == "밀도 기반 (DBSCAN)":
        fig_scat = px.scatter(df_sample, x='상호작용_수', y='체류시간_초', color='DB', title=f"DBSCAN 군집 산점도 (실루엣: {sil_db:.4f})")
    else:
        fig_scat = px.scatter(df_sample, x='상호작용_수', y='체류시간_초', color='GMM', symbol='기기_유형', title=f"GMM 군집 산점도 (실루엣: {sil_gmm:.4f})")
        
    st.plotly_chart(fig_scat, use_container_width=True)

# 2) 교차 히트맵 탭
with tab_heatmap:
    st.markdown("#### 모델 예측 그룹 × 사용자 동작 교차 히트맵")
    selected_model_heat = st.selectbox(
        "교차 히트맵으로 확인할 모델을 선택하세요:", 
        ["로지스틱 회귀", "중심 기반 (K-Means)", "분포 기반 (GMM)"],
        key="heatmap_select"
    )
    
    map_col_dict = {
        "로지스틱 회귀": 'LR',
        "중심 기반 (K-Means)": 'KM',
        "분포 기반 (GMM)": 'GMM'
    }
    target_cluster_col = map_col_dict[selected_model_heat]
    
    # 모델 예측 그룹과 사용자 동작 간의 교차 빈도수 테이블 생성
    heatmap_data = pd.crosstab(df_sample[target_cluster_col], df_sample['사용자_동작'])
    
    fig_heat = px.imshow(
        heatmap_data,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"[{selected_model_heat}] 모델 예측 그룹별 사용자 동작 몰림 현상 (빈도수)",
        labels=dict(x="사용자 동작 (User Action)", y="예측 그룹 (Cluster / Class)", color="세션 수")
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.info("💡 **교차 히트맵 설명:** 각 모델이 분류한 그룹(0 또는 1) 안에서 유저들이 **구체적으로 어떤 `사용자_동작` (조회, 장바구니, 구매 등)에 가장 많이 몰려있는지** 빈도수를 색상 농도로 명확하게 비교할 수 있습니다.")
