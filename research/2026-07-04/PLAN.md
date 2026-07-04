# Predicting Student Health Risk 연구 계획

## 현재 확인한 대회 구조

- Kaggle Playground Series Season 6 Episode 7
- 목표: `health_condition` 3-class 분류
- 평가지표: balanced accuracy
- train: 690,088 rows
- test: 295,753 rows
- target:
  - `at-risk`: 592,561 rows, 85.8675%
  - `unhealthy`: 57,724 rows, 8.3647%
  - `fit`: 39,803 rows, 5.7678%
- 수치형과 범주형 feature 대부분에 결측치가 존재합니다.

이전 Stellar Class 대회와 마찬가지로 대규모 tabular 3-class balanced accuracy
문제입니다. OOF probability bank, class-wise stacking, subset error analysis,
public/private track 분리 방식을 재사용할 수 있습니다.

다만 이번 데이터는 target 불균형과 결측치가 훨씬 강합니다. 이전 모델 파일을
그대로 복사하지 않고 검증 원칙과 연구 구조만 재사용합니다.

## 절대 유지할 검증 원칙

1. 모든 모델은 동일한 Stratified fold assignment를 사용합니다.
2. train fold로 학습하고 validation fold 예측만 OOF에 기록합니다.
3. 조기 종료와 최적 iteration 선택 기준은 balanced accuracy입니다.
4. logloss는 확률 품질 진단용으로 함께 저장하지만 최종 iteration 기준이 아닙니다.
5. target encoding과 target statistics는 반드시 fold 안에서만 fit합니다.
6. external data를 사용할 때 validation은 competition train만 유지합니다.
7. public leaderboard 점수는 private/generalization 후보 선택에 사용하지 않습니다.
8. 공개 OOF/test prediction은 출처, class order, fold 구조를 확인한 뒤 별도 bank로 관리합니다.
9. 다른 사람의 최종 submission CSV를 정답처럼 사용하지 않습니다.
10. OOF score뿐 아니라 class recall, fold 분산, 취약 subset, prediction 변경량을 확인합니다.

## 단계별 계획

### 1. OOF/CV 환경

- class order를 `fit`, `at-risk`, `unhealthy`로 고정합니다.
- 5-fold StratifiedKFold와 fold assignment CSV를 생성합니다.
- CatBoost, LightGBM, XGBoost가 같은 fold를 공유하도록 합니다.
- 모델별 OOF/test probability를 `float32 .npy`로 저장합니다.
- fold score, class recall, confusion matrix, training history를 저장합니다.
- smoke mode 2-fold 소표본으로 입출력과 metric 방향을 먼저 검증합니다.

### 2. 강한 단일 모델

- CatBoost native categorical + missing value
- LightGBM encoded categorical + missing indicators
- XGBoost histogram tree + missing indicators
- balanced class weight 사용/미사용 비교
- class bias와 probability calibration은 OOF에서만 탐색
- early-stop iteration과 fold별 성능 분산 비교

### 3. Feature 연구

- row missing count와 feature별 missing indicator
- sleep, BMI, heart rate 기준점과의 거리
- step/exercise/calorie/water ratio
- 활동량과 수면의 interaction
- 수치 feature quantile bin/category view
- 범주형 interaction
- fold-safe target encoding
- target이 순서형이라는 가설을 이용한 `fit vs rest`, `unhealthy vs rest` 보조 모델

### 4. 모델 다양성

- one-vs-rest CatBoost/XGBoost
- ordinal two-threshold decomposition
- RealMLP/TabM 계열
- seed와 feature view가 다른 모델
- 원본 College Student Health Behavior Dataset의 외부 일반화/낮은 가중치 학습

### 5. OOF stacking

- source별 OOF score와 class recall
- source 간 error/probability correlation
- multinomial logistic stacker
- class-wise logistic blender
- greedy probability blend
- repeated meta-fold stability
- source ablation

### 6. Subset error analysis

- target class
- missing count
- 각 categorical group
- sleep/BMI/heart-rate/activity quantile
- high-stress/poor-sleep/sedentary 조합
- train/test distribution shift

### 7. 최종 제출

- Generalization: OOF/CV, class recall, fold stability로만 선택
- Public exploration: 별도 폴더와 별도 이름으로 관리
- 마지막에는 두 track을 혼합하지 않고 최대 2개 final submission을 선택

## 첫 구현

`notebooks/01_cv_oof_baseline.ipynb`

- 기본값은 `RUN_MODE = "smoke"`입니다.
- smoke 검증 후 `RUN_MODE = "full"`로 변경합니다.
- full mode는 5-fold CatBoost, LightGBM, XGBoost와 equal-probability ensemble을 생성합니다.
- 모든 산출물은 `artifacts/oof_cv_baseline/`에 저장합니다.

## 첫 성공 조건

- smoke mode가 오류 없이 종료
- OOF probability row sum이 1
- sample submission ID와 완전 일치
- 각 모델의 fold validation balanced accuracy 출력
- class별 recall과 confusion matrix 저장
- full mode에서 모든 train row가 정확히 한 번 validation으로 예측됨
- 세 모델의 OOF/test class order가 동일
