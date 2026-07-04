# Shared Notebook Audit

확인 대상:

- `ps-s6e7-eda-ensemble-lb-0-95075.ipynb`
- `hmnshudhmn24_preds_0.95021.csv`
- `submission_0.csv`

## 결론

공유 노트북은 모델 학습 노트북이 아니라 EDA와 공개 제출물 혼합 노트북이다.
최종 제출은 두 공개 예측을 `0.6 : 0.4`로 섞어 만들며, OOF 예측이나 모델별
교차 검증 점수는 없다. 따라서 공개 점수 `0.95075`는 일반화 성능의 근거로
사용하지 않는다.

공유 CSV는 향후 public 전용 경계 탐색 자료로만 격리한다. OOF 스태커와
private/generalization 모델의 입력에는 사용하지 않는다.

## 검증된 데이터 특징

- 목표 비율: at-risk 85.8675%, unhealthy 8.3647%, fit 5.7678%
- 표준화한 클래스 평균 차이가 큰 수치 변수:
  - sleep_duration: 2.129
  - bmi: 0.924
  - step_count: 0.826
  - exercise_duration: 0.819
  - calorie_expenditure: 0.429
- stress=medium에서 at-risk 비율은 99.39%였다.
- stress=low에서 fit 비율은 20.06%였다.
- stress=high에서 unhealthy 비율은 27.87%였다.
- activity=active에서 fit 비율은 17.16%였다.
- exercise_duration=0 구간의 fit 비율은 0.108%였다.
- calorie_expenditure < 1950 구간은 fit 비율이 낮았다.
- 대부분의 결측은 클래스별 차이가 작지만 BMI 결측은 최대 약 2.16%p 차이가
  있어 결측 지시자를 유지한다.

공유 노트북의 `physical_activity_level` 설명에는 sleep 분석 문장이 복사된
오류가 있었다. 서술은 그대로 신뢰하지 않고 수치로 다시 검증했다.

## CSV 경계 분석

`hmnshudhmn24_preds_0.95021.csv`:

- 295,753행, 세 클래스 확률
- ID 열이 없어 행 순서 의존
- 평균 최대 확률 0.940638
- 평균 1위-2위 확률 차이 0.891818

`submission_0.csv`와 hmn 확률 argmax의 일치율은 99.3765%였고, 1,844행이
달랐다. 다른 행의 평균 확률 차이는 0.20958로 전체의 불확실한 경계에
집중됐다.

주요 전이:

- at-risk -> unhealthy: 943
- unhealthy -> at-risk: 569
- at-risk -> fit: 162
- fit -> unhealthy: 73
- fit -> at-risk: 53
- unhealthy -> fit: 44

첫 경계 연구 우선순위는 `at-risk <-> unhealthy`이다. 다만 이 결론도
정답이 아니라 공개 모델 간 불일치 구조이므로, 실제 채택 여부는 우리 OOF와
반복 교차 검증으로만 결정한다.

## 베이스라인 반영 사항

- 스트레스, 수면 품질, 활동 수준, 흡연·음주에 순서형 수치 표현 추가
- `exercise_is_zero`, `calorie_low_mode` 추가
- 스트레스·수면·활동 순서형 상호작용 추가
- 전체 OOF 외에 stress/sleep/activity/missingness/클래스 경계별 성능 저장
- 모든 진단은 동일한 OOF fold 예측으로 계산

