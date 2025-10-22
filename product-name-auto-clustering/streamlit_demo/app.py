import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import datetime

# -----------------------------
# 0. 기본 설정 (중립화)
# -----------------------------
st.set_page_config(page_title="상품명 자동 분류기", layout="wide")
st.title("🧩 상품명 자동 분류 시스템")
st.caption("연구/PoC 데모 | front/back embedding 기반 의미기반 자동군집")

# -----------------------------
# 1. 엑셀 업로드
# -----------------------------
uploaded_file = st.file_uploader("상품명 데이터 엑셀 파일을 업로드하세요", type=["xlsx"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.success(f"{data.shape[0]}개의 행이 업로드되었습니다.")
else:
    st.stop()

# -----------------------------
# 2. 전처리 함수 정의 (그대로 유지, 도메인 표현 중립화)
# -----------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"[\[\]\(\)]", " ", text)
    text = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# 특수 공백 정규화
_WS_HARD = r"[\u00A0\u202F\u2007\u2000-\u2006\u2008-\u200A\u200B-\u200D\u2060]"
def _normalize_spaces(s: str) -> str:
    s = re.sub(_WS_HARD, " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# 퍼지 패턴 : 특수문자·띄어쓰기 변화에 상관없이 유연하게 매칭하기 위한 정규식 패턴을 자동으로 만들어주기
def _build_fuzzy_pattern(cat: str) -> str:
    base = re.sub(_WS_HARD, "", cat)
    base = re.sub(r"[\s\W_]+", "", base)
    parts = [re.escape(ch) for ch in base]
    if not parts:
        return r"$^"
    return r"(?:[\s\W_]*)".join(parts)

# L_category/사이트이름 제거하여 핵심 토큰만 남기기
def clean_and_trim_text(row):
    text = str(row["pname"])
    category = str(row["L_category"])
    site_name = str(row.get("사이트이름", ""))
    text = re.sub(r"[\[\]\(\)]", " ", text)
    text_norm = _normalize_spaces(text)
    if category and category != "nan":
        cat_pat = _build_fuzzy_pattern(_normalize_spaces(category))
        text_norm = re.sub(cat_pat, " ", text_norm, flags=re.IGNORECASE)
    if site_name and site_name != "nan":
        site_pat = _build_fuzzy_pattern(_normalize_spaces(site_name))
        text_norm = re.sub(site_pat, " ", text_norm, flags=re.IGNORECASE)
    text_norm = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", text_norm)
    return _normalize_spaces(text_norm).lower()

# -----------------------------
# 3. 전처리 실행
# -----------------------------
data["상품명_clean"] = data["pname"].astype(str).apply(clean_text)
data["상품명_trim"] = data.apply(clean_and_trim_text, axis=1)

site = st.selectbox("사이트 선택", sorted(data["site_code"].unique()))
cats = st.multiselect("L_category 선택", sorted(data["L_category"].unique()))
df_test = data[(data["site_code"] == site) & (data["L_category"].isin(cats))].copy()
st.write(f"선택된 데이터: {df_test.shape[0]}개")

# -----------------------------
# 4. 1단계: 앞단 중심 임베딩 + 클러스터링
# -----------------------------
@st.cache_resource(show_spinner=True)
def get_model():
    # 한국어 문장 임베딩 (중립 모델명)
    return SentenceTransformer("jhgan/ko-sroberta-multitask")

model = get_model()

def front_only_embedding(text, front_ratio=0.7, min_tokens=2, decay_rate=0.25):
    tokens = re.split(r"\s+", str(text).strip())
    n = len(tokens)
    if n == 0:
        return np.zeros(model.get_sentence_embedding_dimension())
    split_idx = max(min_tokens, int(n * front_ratio))
    front_tokens = tokens[:split_idx]
    back_tokens = tokens[split_idx:]
    # 앞단 토큰 가중 평균
    front_emb = model.encode(front_tokens, show_progress_bar=False)
    front_weights = np.array([np.exp(-decay_rate * i) for i in range(len(front_tokens))])
    front_weights /= front_weights.sum()
    front_vec = np.average(front_emb, axis=0, weights=front_weights)
    # 뒷단은 미세 가중치
    back_vec = (np.mean(model.encode(back_tokens, show_progress_bar=False), axis=0) * 0.01
                if back_tokens else np.zeros(model.get_sentence_embedding_dimension()))
    return front_vec + back_vec

def extract_core_phrase(text):
    text = str(text)
    text = re.sub(r"\bv\d+|\bver\d+|\b\d{2,4}(\.\d+)?\b", "", text)
    text = re.sub(r"(교재|포함|미포함|only|온리|ver|버전)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    core = " ".join(tokens[:3]) if len(tokens) > 3 else text
    return core.strip()

if st.button("1단계 클러스터링 실행"):
    cluster_results = []
    for cat in sorted(df_test["L_category"].unique()):
        subset = df_test[df_test["L_category"] == cat].copy()
        if len(subset) < 2:
            continue
        subset["embedding_front"] = subset["상품명_trim"].apply(front_only_embedding)
        X_front = np.vstack(subset["embedding_front"].values)
        cluster_front = AgglomerativeClustering(
            n_clusters=None, distance_threshold=9, linkage='ward'
        )
        subset["cluster_lv1"] = cluster_front.fit_predict(X_front)
        cluster_results.append(subset)

    if cluster_results:
        df_lv1 = pd.concat(cluster_results, ignore_index=True)
        df_lv1["대표핵심어"] = df_lv1["상품명_trim"].apply(extract_core_phrase)

        rep_lv1 = (
            df_lv1.groupby(["사이트이름", "L_category", "cluster_lv1"])
            .agg(대표핵심어=("대표핵심어", lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]))
            .reset_index()
        )
        rep_lv1["대표_lv1명"] = rep_lv1.apply(
            lambda row: f"{row['사이트이름']}_{row['L_category']}_{row['대표핵심어']}", axis=1
        )

        df_lv1 = df_lv1.merge(
            rep_lv1[["사이트이름", "L_category", "cluster_lv1", "대표_lv1명"]],
            on=["사이트이름", "L_category", "cluster_lv1"],
            how="left"
        )

        # ✅ 1단계 결과 저장 (2단계 및 다운로드에서 활용)
        st.session_state["df_lv1"] = df_lv1

        # 표
        st.dataframe(
            df_lv1[["사이트이름", "L_category", "대표_lv1명", "pname", "cluster_lv1"]]
            .sort_values(["L_category", "cluster_lv1"])
            .reset_index(drop=True)
        )

        # -----------------------------
        # 📊 L_category > 대표 그룹별 묶인 상품 시각화
        # -----------------------------
        st.markdown("### 📂 L_category별 대표 그룹 시각화")
        for cat, cat_group in df_lv1.groupby("L_category"):
            st.markdown(f"#### 🗂️ {cat}")
            for rep_name, subset in cat_group.groupby("대표_lv1명"):
                with st.expander(f"📦 {rep_name} ({len(subset)}개 상품)"):
                    for s in subset["pname"].tolist():
                        st.write(f" - {s}")
            st.markdown("---")
    else:
        st.warning("선택된 데이터에서 클러스터를 형성할 충분한 샘플이 없습니다.")

# -----------------------------
# 5. 2단계: 뒷단 중심 세분화 (선택)
# -----------------------------
st.markdown("---")
st.subheader("2단계 세분화 (선택)")

def back_only_embedding(text, back_ratio=0.5, min_tokens=2, decay_rate=0.25):
    tokens = re.split(r"\s+", str(text).strip())
    n = len(tokens)
    if n == 0:
        return np.zeros(model.get_sentence_embedding_dimension())
    split_idx = max(n - max(min_tokens, int(n * back_ratio)), 0)
    front_tokens = tokens[:split_idx]
    back_tokens = tokens[split_idx:]
    back_emb = model.encode(back_tokens, show_progress_bar=False)
    back_weights = np.array([np.exp(-decay_rate * i) for i in range(len(back_tokens))])[::-1]
    back_weights /= back_weights.sum()
    back_vec = np.average(back_emb, axis=0, weights=back_weights)
    front_vec = (np.mean(model.encode(front_tokens, show_progress_bar=False), axis=0) * 0.01
                 if front_tokens else np.zeros(model.get_sentence_embedding_dimension()))
    return back_vec + front_vec

if st.button("2단계 세분화 실행"):
    if "df_lv1" not in st.session_state:
        st.warning(" 먼저 1단계를 실행해주세요.")
        st.stop()

    df_lv1 = st.session_state["df_lv1"].copy()
    st.info("2단계 세분화를 실행 중입니다. 잠시만 기다려주세요...")

    cluster_results = []
    for rep in sorted(df_lv1["cluster_lv1"].unique()):
        subset = df_lv1[df_lv1["cluster_lv1"] == rep].copy()
        if len(subset) < 3:
            subset["cluster_lv2"] = 0
            cluster_results.append(subset)
            continue
        subset["embedding_back"] = subset["상품명_trim"].apply(back_only_embedding)
        X_back = np.vstack(subset["embedding_back"].values)
        cluster_back = AgglomerativeClustering(
            n_clusters=None, distance_threshold=5, linkage='ward'
        )
        subset["cluster_lv2"] = cluster_back.fit_predict(X_back)
        cluster_results.append(subset)

    df_lv2 = pd.concat(cluster_results, ignore_index=True) if cluster_results else df_lv1.assign(cluster_lv2=0)

    # ✅ 2단계 결과 저장 (다운로드 탭에서 사용)
    st.session_state["df_lv2"] = df_lv2

    # 표
    st.dataframe(
        df_lv2[["사이트이름", "L_category", "대표_lv1명", "pname", "cluster_lv1", "cluster_lv2"]]
        .sort_values(["L_category", "cluster_lv1", "cluster_lv2"])
        .reset_index(drop=True)
    )

    # 📊 세분화 결과 시각화 (정돈된 계층 구조)
    st.markdown("### 📂 L_category > 1차 분류 > 세분화 결과 시각화")
    for cat, cat_group in df_lv2.groupby("L_category"):
        st.markdown(f"<h4>📁 {cat}</h4>", unsafe_allow_html=True)
        for lv1_name, lv1_group in cat_group.groupby("대표_lv1명"):
            # 1차 분류명
            st.markdown(
                f"<div style='margin-left:10px; color:#555; font-size:13px; font-style:italic;'>"
                f"1차 분류명: {lv1_name}</div>",
                unsafe_allow_html=True
            )
            # 실제 상품명만 나열
            product_list = lv1_group["pname"].unique().tolist()
            for s in product_list:
                st.markdown(
                    f"<div style='margin-left:35px; font-size:14px;'>• {s}</div>",
                    unsafe_allow_html=True
                )
            st.markdown("<hr style='margin:6px 0; border:0.5px solid #eee;'>", unsafe_allow_html=True)

# ==========================================================
# 📥 결과 다운로드 (1단계 / 2단계 탭 + 날짜 태깅)
# ==========================================================
today = datetime.date.today().strftime("%Y-%m-%d")

st.markdown("---")
st.subheader("📂 결과 다운로드")

tab1, tab2 = st.tabs(["1️. 1단계 결과", "2️. 2단계 결과"])

# 1️⃣ 1단계 결과
with tab1:
    st.markdown("### 1단계 결과 다운로드")
    st.caption("대표_lv1 기준으로 묶인 상품명 자동 분류 결과입니다.")
    df_lv1_final = st.session_state.get("df_lv1")
    if df_lv1_final is not None and not df_lv1_final.empty:
        # (A) 분석/리포팅용 간단 형태
        engine_export_lv1 = df_lv1_final[["사이트이름", "L_category", "대표_lv1명", "pname"]].copy()
        st.download_button(
            label=f"리포팅용(간단) CSV 다운로드 ({today})",
            data=engine_export_lv1.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"auto_cluster_lv1_report_{today}.csv",
            mime="text/csv"
        )
        # (B) 현재 기준 저장용 (임베딩/중간컬럼 포함)
        st.download_button(
            label=f" 기준 저장용(중간컬럼 포함) CSV 다운로드 ({today})",
            data=df_lv1_final.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"auto_cluster_lv1_full_{today}.csv",
            mime="text/csv"
        )
    else:
        st.warning("먼저 1단계를 실행해주세요.")

# 2️⃣ 2단계 결과
with tab2:
    st.markdown("### 2단계 세분화 결과 다운로드")
    st.caption("대표_lv1 내부에서 세분화된 상품명 자동 분류 결과입니다.")
    df_lv2_final = st.session_state.get("df_lv2")
    if df_lv2_final is not None and not df_lv2_final.empty:
        # (A) 리포팅용 간단 형태 (cluster_lv2 포함)
        engine_export_lv2 = df_lv2_final[["사이트이름", "L_category", "대표_lv1명", "cluster_lv2", "pname"]].copy()
        st.download_button(
            label=f"리포팅용(간단) CSV 다운로드 ({today})",
            data=engine_export_lv2.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"auto_cluster_lv2_report_{today}.csv",
            mime="text/csv"
        )
        # (B) 현재 기준 저장용 (임베딩/중간컬럼 포함)
        st.download_button(
            label=f"기준 저장용(중간컬럼 포함) CSV 다운로드 ({today})",
            data=df_lv2_final.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"auto_cluster_lv2_full_{today}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ 먼저 2단계를 실행해주세요.")
