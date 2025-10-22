
import re
import numpy as np
import pandas as pd
from typing import Any
from sklearn.cluster import AgglomerativeClustering

# Sentence-BERT 전역 로딩 (지연 로드)
_model = None
def load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    return _model

# -----------------------------
#  텍스트 전처리 (한국어 주석 유지)
# -----------------------------
def clean_text(text: Any) -> str:
    text = str(text)
    text = re.sub(r"[\[\]\(\)]", " ", text)
    text = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

_WS_HARD = r"[\u00A0\u202F\u2007\u2000-\u2006\u2008-\u200A\u200B-\u200D\u2060]"
def _normalize_spaces(s: str) -> str:
    s = re.sub(_WS_HARD, " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _build_fuzzy_pattern(cat: str) -> str:
    base = re.sub(_WS_HARD, "", cat)
    base = re.sub(r"[\s\W_]+", "", base)
    parts = [re.escape(ch) for ch in base]
    if not parts:
        return r"$^"
    return r"(?:[\s\W_]*)".join(parts)

def clean_and_trim_text(row: pd.Series,
                        pname_col="pname",
                        category_col="L_category",
                        site_name_col="site_name") -> str:
    text = str(row[pname_col])
    category = str(row.get(category_col, ""))
    site_name = str(row.get(site_name_col, ""))

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

def extract_core_phrase(text: str) -> str:
    text = str(text)
    text = re.sub(r"\bv\d+|\bver\d+|\b\d{2,4}(\.\d+)?\b", "", text)
    text = re.sub(r"(교재|포함|미포함|only|온리|ver|버전)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    core = " ".join(tokens[:3]) if len(tokens) > 3 else text
    return core.strip()

# -----------------------------
#  임베딩 (앞단/뒷단 가중치)
# -----------------------------
def front_only_embedding(text: str, front_ratio=0.7, min_tokens=2, decay_rate=0.25):
    model = load_model()
    tokens = re.split(r"\s+", str(text).strip())
    n = len(tokens)
    if n == 0:
        return np.zeros(model.get_sentence_embedding_dimension())
    split_idx = max(min_tokens, int(n * front_ratio))
    front_tokens = tokens[:split_idx]
    back_tokens = tokens[split_idx:]

    front_emb = model.encode(front_tokens, show_progress_bar=False)
    front_weights = np.array([np.exp(-decay_rate * i) for i in range(len(front_tokens))])
    front_weights /= front_weights.sum()
    front_vec = np.average(front_emb, axis=0, weights=front_weights)

    back_vec = (np.mean(model.encode(back_tokens, show_progress_bar=False), axis=0) * 0.01
                if back_tokens else np.zeros(model.get_sentence_embedding_dimension()))
    return front_vec + back_vec

def back_only_embedding(text: str, back_ratio=0.5, min_tokens=2, decay_rate=0.25):
    model = load_model()
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

# -----------------------------
#  1단계 / 2단계 클러스터링 파이프라인
# -----------------------------
def stage1_cluster(df: pd.DataFrame,
                   site_col="site_code",
                   category_col="L_category",
                   site_name_col="site_name",
                   pname_col="pname",
                   distance_threshold=9) -> pd.DataFrame:
    """L_category 내에서 앞단 중심 임베딩으로 1단계 군집"""
    df = df.copy()
    df["상품명_clean"] = df[pname_col].astype(str).apply(clean_text)
    df["상품명_trim"] = df.apply(
        lambda r: clean_and_trim_text(r, pname_col=pname_col,
                                      category_col=category_col,
                                      site_name_col=site_name_col), axis=1)

    results = []
    for cat in sorted(df[category_col].dropna().unique()):
        subset = df[df[category_col] == cat].copy()
        if len(subset) < 2:
            continue
        subset["embedding_front"] = subset["상품명_trim"].apply(front_only_embedding)
        X = np.vstack(subset["embedding_front"].values)
        cluster = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage="ward"
        )
        subset["cluster_lv1"] = cluster.fit_predict(X)
        results.append(subset)

    if not results:
        return pd.DataFrame(columns=df.columns.tolist() + ["cluster_lv1", "대표핵심어", "대표_lv1명"])

    df_lv1 = pd.concat(results, ignore_index=True)
    df_lv1["대표핵심어"] = df_lv1["상품명_trim"].apply(extract_core_phrase)

    rep = (
        df_lv1.groupby([site_name_col, category_col, "cluster_lv1"])
        .agg(대표핵심어=("대표핵심어", lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]))
        .reset_index()
    )
    rep["대표_lv1명"] = rep.apply(
        lambda r: f"{r[site_name_col]}_{r[category_col]}_{r['대표핵심어']}", axis=1
    )
    df_lv1 = df_lv1.merge(
        rep[[site_name_col, category_col, "cluster_lv1", "대표_lv1명"]],
        on=[site_name_col, category_col, "cluster_lv1"], how="left"
    )
    return df_lv1

def stage2_segment(df_lv1: pd.DataFrame,
                   distance_threshold=5) -> pd.DataFrame:
    """1단계 결과를 입력으로 받아 뒷단 중심 임베딩으로 세분화"""
    df_lv1 = df_lv1.copy()
    results = []
    for lv1 in sorted(df_lv1["cluster_lv1"].dropna().unique()):   
        subset = df_lv1[df_lv1["cluster_lv1"] == lv1].copy()
        if len(subset) < 3:
            subset["cluster_lv2"] = 0
            results.append(subset)
            continue
        subset["embedding_back"] = subset["상품명_trim"].apply(back_only_embedding)
        X = np.vstack(subset["embedding_back"].values)
        cluster = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage="ward"
        )
        subset["cluster_lv2"] = cluster.fit_predict(X)
        results.append(subset)
    return pd.concat(results, ignore_index=True) if results else df_lv1

# -----------------------------
#  헬퍼: 리포팅용 요약 테이블
# ----------------------------- 
def summarize_for_reporting(df_lv1: pd.DataFrame,
                            site_name_col="site_name",
                            category_col="L_category",
                            pname_col="pname") -> pd.DataFrame:
    cols = [site_name_col, category_col, "대표_lv1명", pname_col]
    return df_lv1[cols].sort_values([category_col, "대표_lv1명"]).reset_index(drop=True)
