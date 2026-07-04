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

1. 수정된 LightGBM iteration 선택을 짧게 재검증합니다.
2. notebook 기본 smoke mode 60,000행을 실행합니다.
3. 그래프와 fold 편차를 확인합니다.
4. 문제가 없으면 `RUN_MODE = "full"`로 변경해 5-fold를 실행합니다.
5. full OOF 결과를 첫 일반화 baseline으로 고정합니다.
