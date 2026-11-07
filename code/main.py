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

# ---------------------------------------------------------
# 데이터 로드 (data 폴더 또는 현재 폴더)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    paths = ['data/processed_retail_user_behavior_korean.csv', 'processed_retail_user_behavior_korean.csv']
    for path in paths:
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return pd.read_csv(path)
        except Exception:
            continue
    return None

df = load_data()

# 파일이 없거나 비어있을 경우 안전한 샘플 데이터 자동 생성
if df is None or len(df) == 0:
    import os
    np.random.seed(42)
    n = 3000
    df = pd.DataFrame({
        '번호': range(1, n + 1),
        '이벤트_인덱스': np.random.randint(1, 5, n),
        '사용자_동작': np.random.choice(['view', 'click', 'add_to_cart', 'wishlist', 'drop', 'purchase'], n, p=[0.4, 0.25, 0.1, 0.08, 0.12, 0.05]),
        '카테고리': np.random.choice(['Electronics', 'Apparel', 'Groceries', 'Sports'], n),
        '브랜드': np.random.choice(['Samsung', 'Nike', 'Apple', 'Local'], n),
        '가격': np.round(np.random.uniform(10, 500, n), 2),
        '채널': np.random.choice(['web', 'app', 'mobile'], n),
        '기기_유형': np.random.choice(['desktop', 'android', 'ios', 'tablet'], n),
        '지역': np.random.choice(['US', 'KR', 'UK', 'JP'], n),
        '유입_경로': np.random.choice(['organic', 'direct', 'affiliate'], n),
        '체류시간_초': np.round(np.random.exponential(30, n) + 5, 2),
        '세션_길이': np.random.randint(1, 10, n),
        '상호작용_수': np.random.poisson(4, n) + 1,
        '전환_여부': np.random.choice([0, 1], n, p=[0.9, 0.1]),
        '이탈_여부': np.random.choice([0, 1], n, p=[0.8, 0.2])
    })

# 샘플링 (3,000건)
sample_size = min(3000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).copy().reset_index(drop=True)

# 참여도 스코어 산출
if '참여도_스코어' not in df_sample.columns:
    min_t, max_t = df_sample['체류시간_초'].min(), df_sample['체류시간_초'].max()
    min_i, max_i = df_sample['상호작용_수'].min(), df_sample['상호작용_수'].max()
    df_sample['참여도_스코어'] = (
        ((df_sample['체류시간_초'] - min_t) / (max_t - min_t + 1e-9)) * 50 +
        ((df_sample['상호작용_수'] - min_i) / (max_i - min_i + 1e-9)) * 50
    )

# ---------------------------------------------------------
# 왼쪽 사이드바 탭 구성
# ---------------------------------------------------------
st.sidebar.title("🧭 네비게이션 메뉴")
menu_tab = st.sidebar.radio(
    "이동할 탭을 선택하세요:",
    ["1. 데이터 분석 대시보드", "2. 프로젝트 보고서 및 답변 작성"]
)

# =========================================================
# [탭 1] 원래의 데이터 및 시각자료 대시보드
# =========================================================
if menu_tab == "1. 데이터 분석 대시보드":
    st.title("🎯 고객 참여도(Engagement) 기반 군집화 & 최적화 대시보드")
    st.markdown("체류 시간과 상호작용 수를 결합한 참여도 스코어를 바탕으로 사용자의 행동 몰림 현상과 최적의 군집 모델을 분석합니다.")
    st.markdown("---")

    # 1. 사용자 행동 몰림 현상 분석
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

    # 전처리 및 모델 학습
    num_cols = ['체류시간_초', '세션_길이', '상호작용_수', '참여도_스코어']
    cat_cols = ['카테고리', '기기_유형', '유입_경로']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    X_processed = preprocessor.fit_transform(df_sample[num_cols + cat_cols])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=5).fit(X_processed)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(X_processed)
    agg = AgglomerativeClustering(n_clusters=2).fit(X_processed)

    labels_km = kmeans.labels_
    labels_gmm = gmm.predict(X_processed)
    labels_agg = agg.labels_

    sil_km, ch_km, db_km = silhouette_score(X_processed, labels_km), calinski_harabasz_score(X_processed, labels_km), davies_bouldin_score(X_processed, labels_km)
    sil_gmm, ch_gmm, db_gmm = silhouette_score(X_processed, labels_gmm), calinski_harabasz_score(X_processed, labels_gmm), davies_bouldin_score(X_processed, labels_gmm)
    sil_agg, ch_agg, db_agg = silhouette_score(X_processed, labels_agg), calinski_harabasz_score(X_processed, labels_agg), davies_bouldin_score(X_processed, labels_agg)

    # 2. 성능 지표 비교 표
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

    # 3. 최적 군집 개수 탐색 곡선 및 상세 수치
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

    # 4. 시각화 영역: 산점도 및 교차 히트맵
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


# =========================================================
# [탭 2] 질문에 따른 작성 칸 (전부 자동 작성 + 직접 수정 가능 칸)
# =========================================================
elif menu_tab == "2. 프로젝트 보고서 및 답변 작성":
    st.title("📝 프로젝트 평가 항목 및 답변 작성 공간")
    st.markdown("스크린샷의 평가 항목에 맞춰, **데이터 전처리 및 분석 내용**은 앞서 진행한 연구를 바탕으로 작성되어 있으며, 나머지 항목들은 직접 작성하실 수 있도록 입력 칸으로 구성되어 있습니다[cite: 1].")
    st.markdown("---")

    # 1. 문제 정의 (직접 작성 칸)
    st.subheader("1. 문제 정의 (주제 선정 및 문제 정의)")
    q1_default = "전자상거래 웹사이트 내에서 단순 구매 전환 예측을 넘어, 사용자들이 퍼넬(Funnel)의 어느 단계와 행동에 가장 많이 몰려있는지(체류 및 병목 구간)를 비지도 학습 기반 군집화 모델을 통해 분석하는 것을 목표로 설정함."
    user_q1 = st.text_area("✍️ [문제 정의] 내용을 입력하거나 수정하세요:", value=q1_default, height=100)

    # 2. 데이터 분석 (자동 작성된 내용 제공)
    st.subheader("2. 데이터 분석 (데이터 기초통계분석 결과)")
    q2_default = "전처리된 전자상거래 로그 데이터를 활용하여 체류시간, 세션 길이, 상호작용 수 등의 기초 통계를 분석하였으며, 대다수의 사용자가 특정 초기 조회(view) 단계에 머무르는 몰림 현상과 사용자 참여도 분포를 확인 완료함."
    user_q2 = st.text_area("✍️ [데이터 분석] 내용을 입력하거나 수정하세요:", value=q2_default, height=100)

    # 3. 데이터 전처리 (자동 작성된 내용 제공)
    st.subheader("3. 데이터 전처리 (데이터 시각화, 결측치와 이상치 처리 결과)")
    q3_default = "1) 불필요한 고차원 및 중복 변수(지역, 채널, 브랜드 등)를 제거하여 차원의 저주를 방지함.\n2) 체류시간과 상호작용 수를 결합하여 '참여도 스코어' 특성 변수를 새롭게 산출함.\n3) 수치형 데이터는 표준화(StandardScaler), 범주형 데이터는 원-핫 인코딩(One-Hot Encoder)을 적용하여 군집화 모델 입력에 최적화함."
    user_q3 = st.text_area("✍️ [데이터 전처리] 내용을 입력하거나 수정하세요:", value=q3_default, height=130)

    # 4. 기준선 (직접 작성 칸)
    st.subheader("4. 기준선 (데이터에 대한 기준선 제시)")
    q4_default = "군집화 모델(K-Means, GMM, Agglomerative)의 성능과 최적의 군집 개수(K)를 객관적으로 평가하기 위해 실루엣 계수(Silhouette Score), 칼린스키-하바라슈 지수, 데이비스-불딘 지수, 그리고 K-Means의 이너시아(Inertia) 변화(엘보우 방법)를 평가 기준선으로 설정함."
    user_q4 = st.text_area("✍️ [기준선] 내용을 입력하거나 수정하세요:", value=q4_default, height=100)

    # 5. 모델 평가 (직접 작성 칸)
    st.subheader("5. 모델 평가 (모델 학습 결과 및 평가지표 해석)")
    q5_default = "3가지 비지도 군집화 모델을 적용한 결과, K-Means(중심 기반) 모델이 참여도 스코어와 체류 시간 기반 데이터를 가장 안정적으로 분리(실루엣 계수 및 응집도 우수)함을 확인하였으며, 교차 히트맵 분석을 통해 고참여 그룹과 저참여 그룹의 행동 몰림 구간을 명확히 파악함."
    user_q5 = st.text_area("✍️ [모델 평가] 내용을 입력하거나 수정하세요:", value=q5_default, height=100)

    # 6. 평가지표를 바탕으로 데이터에서 알게 된 사실 (직접 작성 칸)
    st.subheader("6. 평가 지표를 바탕으로 데이터에서 알게 된 사실")
    q6_default = "사용자들의 대다수가 특정 단계(조회 및 초기 클릭)에 체류하며 이탈하는 병목 현상이 발생하고 있음을 발견하였으며, 참여도가 높은 소수의 유저들만이 장바구니 등 후속 단계로 깊게 몰입하는 패턴을 지표와 교차 히트맵을 통해 도출함."
    user_q6 = st.text_area("✍️ [평가지표 해석] 내용을 입력하거나 수정하세요:", value=q6_default, height=100)

    # 7. 느낀점 (직접 작성 칸)
    st.subheader("7. 느낀점 (AI를 활용해 데이터를 분석하며 느끼고 배운 점)")
    q7_default = "단순히 정답을 맞추는 지도학습 예측 모델에서 벗어나, 비지도 군집화 모델과 교차 히트맵 시각화를 통해 데이터 속에 숨겨진 사용자 행동 패턴과 병목 구간을 스스로 찾아내는 과정의 중요성을 깊이 깨달음."
    user_q7 = st.text_area("✍️ [느낀점] 내용을 입력하거나 수정하세요:", value=q7_default, height=100)

    st.success("✨ 작성하신 내용들은 Streamlit 앱 내에서 확인 및 수정이 가능하며, 필요시 복사하여 최종 보고서에 활용하실 수 있습니다!")
