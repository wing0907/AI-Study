# AI-Study Monorepo

실험용 머신러닝·딥러닝 스크립트를 모아둔 저장소입니다. 각 하위 폴더는 독립적인 실습 노트북이나 스크립트 세트를 담고 있습니다.

## 빠른 목차

- [`AI-Study/_save`](#ai-study_save) — 훈련 결과물과 체크포인트 저장소
- [`AI-Study/AE`](#ai-studyae) — 오토인코더 관련 실험
- [`AI-Study/class`](#ai-studyclass) — 수업 예제 코드 모음
- [`AI-Study/competition`](#ai-studycompetition) — 캐글 등 대회용 스크립트
- [`AI-Study/competition_ai`](#ai-studycompetition_ai) — 추가 대회/실험 코드
- [`AI-Study/competition_전력`](#ai-studycompetition_전력) — 전력 수요 예측 대회 자료
- [`AI-Study/FashionMNIST`](#ai-studyfashionmnist) — 패션 MNIST 실험
- [`AI-Study/fastapi`](#ai-studyfastapi) — FastAPI 예제
- [`AI-Study/final_project`](#ai-studyfinal_project) — 최종 프로젝트 (웹·백엔드 포함)
- [`AI-Study/keras`](#ai-studykeras) — 케라스 기본 실습
- [`AI-Study/keras2`](#ai-studykeras2) — 케라스 확장 실습
- [`AI-Study/ml`](#ai-studyaiml) — 전통 ML 알고리즘 실습
- [`AI-Study/torch`](#ai-studytorch) — PyTorch 실험
- 기타 폴더: `gpt1`, `html`, `js`, `llm`, `openai`, `python`, `tf114` 등 실습별 추가 자료

### `AI-Study/_save`
- 각 프로젝트의 모델 가중치, 전처리 결과, 로그를 CSV/PKL 등으로 저장
- `cat_dog`, `jena`, `m15_cv_results` 등 서브폴더별로 실험 결과 구분

### `AI-Study/AE`
- 다양한 오토인코더 구조 실험 (`a01_autoencoder.py`, `a06_CAE.py` 등)
- 노이즈 제거, PCA 결합, 시각화 스크립트 포함

### `AI-Study/class`
- `ai01`~`ai05` 시리즈로 구성된 강의용 예제
- 데이터 로딩, 전처리, 기본 모델 학습 스크립트

### `AI-Study/competition`
- 캐글·DACON 등 대회 제출 코드와 CV 결과 정리
- `c0_00000.py`부터 `c14_01_gs_boost_up_cv_results.py`까지 단계별 실험 로그

### `AI-Study/competition_ai`
- 다양한 데이터셋·대회용 실험 스크립트와 CSV 결과
- EDA, 피처 엔지니어링, 모델 튜닝 파일 다수

### `AI-Study/competition_전력`
- 전력 수요 예측 관련 JSON/CSV/파이썬 스크립트 대량 저장
- 시계열 전처리, 모델 비교, 제출 파일 관리

### `AI-Study/FashionMNIST`
- Fashion-MNIST 데이터셋 기반 기본 모델 및 로그
- CNN, 간단한 분류 실험 위주

### `AI-Study/fastapi`
- FastAPI로 작성된 간단한 API 예제
- 백엔드 연동 실습 파일

### `AI-Study/final_project`
- 리액트(TypeScript) 프런트엔드와 파이썬 백엔드가 함께 위치
- 프로젝트 전반의 설정, API, 컴포넌트 구성 포함

### `AI-Study/keras`
- 케라스 기반 딥러닝 실습 스크립트 수백 개
- CNN/RNN/전처리 등 주제별 예제가 폴더 단위로 정리

### `AI-Study/keras2`
- 케라스 심화 실습 또는 변형 예제
- 최신 API나 변형된 실험 스크립트 포함

### `AI-Study/ml`
- 사이킷런 등 전통적인 머신러닝 실습 중심
- 분류, 회귀, 클러스터링 예제 파일 다수

### `AI-Study/torch`
- PyTorch 기반 모델링 실험
- CNN, RNN, 전처리 등 주제별 파이썬 스크립트 포함

---

필요한 폴더로 바로 이동해 실험 스크립트를 확인하거나 기존 결과물(`_save`)을 참고하세요.
