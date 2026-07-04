# 2026-07-04 OOF/CV 코드 검증

## 목적

full 5-fold를 실행하기 전에 다음 항목을 작은 표본으로 확인했습니다.

- target column 누수 여부
- train/test feature column 일치
- 세 모델의 공통 fold 사용
- balanced accuracy 방향의 조기 종료
- OOF/test probability 저장
- class recall, confusion matrix, 학습 곡선 저장
- equal probability ensemble 생성

## 검증 중 발견하고 수정한 문제

첫 feature 검증에서 train feature가 38열, test feature가 37열로 달랐습니다.
train frame에서 `health_condition`을 제거하지 않은 것이 원인이었습니다.

full run 전에 `add_features()`가 `id`와 `health_condition`을 모두 제거하도록
수정했습니다.

수정 후 결과:

```text
train feature shape  (2000, 37)
test feature shape   (1000, 37)
target leakage       none
column order         identical
```

LightGBM은 조기 종료가 발생하지 않고 maximum round에 도달하면 library의
`best_iteration_`이 마지막 round를 가리킬 수 있었습니다. 저장된 validation
balanced accuracy 전체 이력에서 argmax iteration을 직접 선택하도록 수정했습니다.
logloss 최저점은 최종 prediction iteration으로 사용하지 않습니다.

## Code-check 설정

```text
train sample          12,000 rows
folds                 2
CatBoost              120 iterations
LightGBM              160 iterations
XGBoost               160 iterations
early-stop patience   30
```

이 실행은 모델 성능 비교용이 아니라 코드와 산출물 구조 검증용입니다.

## Code-check OOF 결과

```text
XGBoost        0.941426
Equal ensemble 0.941011
LightGBM       0.939700
CatBoost       0.938175
```

Class recall:

```text
XGBoost
fit        0.934971
at-risk    0.933133
unhealthy  0.956175

LightGBM
fit        0.930636
at-risk    0.927310
unhealthy  0.961155

CatBoost
fit        0.929191
at-risk    0.925175
unhealthy  0.960159
```

소표본에서도 class weight와 balanced accuracy 기준이 작동했고, 다수 class인
`at-risk`만 예측하는 붕괴는 발생하지 않았습니다.

## 생성 확인

각 모델 폴더에 다음 파일이 생성됐습니다.

```text
oof_proba.npy
test_proba.npy
fold_scores.csv
training_history.csv
report.json
submission.csv
*_confusion_matrix.png
*_class_recall.png
*_training_curves.png
fold model file
```

공통 폴더:

```text
data_audit.json
feature_manifest.json
fold_assignments.csv
model_summary.csv
run_config.json
equal_ensemble/
```

## 다음 실행

notebook 기본 smoke mode 60,000행 실행까지 완료했습니다.

## 60,000행 smoke 결과

```text
XGBoost         0.944928
Equal ensemble  0.944460
LightGBM        0.943124
CatBoost        0.942708
```

XGBoost class recall:

```text
fit        0.938746
at-risk    0.937481
unhealthy  0.958557
```

XGBoost fold 점수는 `0.944505`, `0.945351`이며 표준편차는 `0.000423`이었다.
최적 반복은 fold별 105, 47이었다. 검증 balanced accuracy는 초반에 상승한 뒤
평탄해졌고, 학습 점수만 계속 상승해 조기 종료 방향이 정상임을 확인했다.

모델 간 예측 불일치율:

```text
CatBoost vs LightGBM  1.2217%
CatBoost vs XGBoost   1.2500%
LightGBM vs XGBoost   0.6950%
```

동일 가중치 평균은 XGBoost 단독보다 낮았다. smoke OOF의 단순 가중치 탐색에서는
XGBoost 비중 0.7~0.9 구간이 유리했으나, 이 결과는 소표본에서 선택한 값이므로
최종 가중치로 고정하지 않는다.

혼동행렬에서 가장 큰 오류는 at-risk를 unhealthy로 예측한 2,040행과
at-risk를 fit으로 예측한 1,181행이었다. 전체 클래스 재현율은 균형을 유지했으며
다수 클래스 단일 예측 붕괴는 없었다.

부분집합 경고는 실제 존재하는 클래스의 재현율만 평균하는 전용 함수로 수정했고,
기존 OOF로 `subset_metrics.csv`와 그래프를 다시 생성했다.

## 다음 실행

1. `RUN_MODE = "full"`로 변경해 공통 5-fold를 실행합니다.
2. full OOF에서 단일 모델 순위와 모델 다양성을 다시 측정합니다.
3. XGBoost 중심 가중치 혼합은 repeated meta-fold 또는 nested validation으로 검증합니다.
4. full OOF 결과를 첫 일반화 baseline으로 고정합니다.
