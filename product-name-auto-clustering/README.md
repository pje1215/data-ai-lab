# data-ai-lab

**데이터 분석 · 머신러닝 실험 저장소 (Research/PoC)**

이 저장소는 텍스트 전처리–임베딩–클러스터링 기반의 자동 분류 실험,
및 관련 연구 노트/데모를 정리하기 위해 운영됩니다.  
실제 회사·브랜드·내부 명칭은 모두 블라인드 처리되었습니다.

---

##  프로젝트 하이라이트: 상품명 자동 분류 (연구용)

- **목표**: 비정형 상품명 텍스트를 의미 기반으로 자동 군집화하여, 운영/분석 관리를 자동화
- **핵심 기법**
  - 텍스트 정제(괄호/특수문자/공백 정규화) + 사이트/카테고리명 제거
  - Sentence-BERT(`ko-sroberta-multitask`) 임베딩
  - **앞단 가중치 임베딩 → 1단계 군집화**, **뒷단 가중치 임베딩 → 2단계 세분화**
  - 대표 핵심어 추출 → 대표 그룹명 생성
- **구성**
  - `modules/clustering_core.py` : UI 없이 호출 가능한 순수 파이썬 엔진
  - `notebooks/Cluster_model.ipynb` : 연구/실험 기록 (블라인드 버전)
  - `streamlit_demo/app.py` : 연구용 미니 데모 UI

---

##  빠른 시작

```bash
pip install -r requirements.txt
# Streamlit 데모 (연구용)
streamlit run streamlit_demo/app.py
```

- 입력 컬럼 예시(범용 명칭): `site_code`, `site_name`, `L_category`, `pname`

---

##  모듈 사용 예시 (코드에서 직접 호출)

```python
import pandas as pd
from modules.clustering_core import stage1_cluster, stage2_segment, summarize_for_reporting

df = pd.read_excel("sample.xlsx")  # 범용 컬럼명 형태 추천
df_lv1 = stage1_cluster(df, site_col="site_code", category_col="L_category", site_name_col="site_name", pname_col="pname")
df_lv2 = stage2_segment(df_lv1)

report = summarize_for_reporting(df_lv1, site_name_col="site_name", category_col="L_category", pname_col="pname")
report.to_csv("stage1_report.csv", index=False, encoding="utf-8-sig")
```

---

##  블라인드 원칙
- 회사/브랜드/고유 식별자는 공개 저장소에 포함하지 않음
- 예제 데이터는 익명화/샘플링된 가짜 값 사용
- 코드/주석의 한국어는 유지하되, 특정 조직/정책 유추 가능 표현은 중립화