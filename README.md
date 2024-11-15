# Bike-Sharing-Demand
https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
![image](https://github.com/user-attachments/assets/953ab93c-2e6b-4212-94c1-cac89ff2d00b)


Training with lr=0.01, batch_size=32, dropout_rate=0.1
Validation loss: 5395.2891
Training with lr=0.01, batch_size=32, dropout_rate=0.2
Validation loss: 5482.6572
Training with lr=0.01, batch_size=32, dropout_rate=0.3
Validation loss: 6582.0435
Training with lr=0.01, batch_size=64, dropout_rate=0.1
Validation loss: 3770.2773
Training with lr=0.01, batch_size=64, dropout_rate=0.2
Validation loss: 4373.5654
Training with lr=0.01, batch_size=64, dropout_rate=0.3
Validation loss: 4660.6055
Training with lr=0.01, batch_size=128, dropout_rate=0.1
Validation loss: 3231.8926
Training with lr=0.01, batch_size=128, dropout_rate=0.2
Validation loss: 3549.3298
Training with lr=0.01, batch_size=128, dropout_rate=0.3
Validation loss: 3756.0415
Training with lr=0.001, batch_size=32, dropout_rate=0.1
Validation loss: 5905.7104
Training with lr=0.001, batch_size=32, dropout_rate=0.2
Validation loss: 6545.5483
Training with lr=0.001, batch_size=32, dropout_rate=0.3
Validation loss: 7154.2412
Training with lr=0.001, batch_size=64, dropout_rate=0.1
Validation loss: 5050.3408
Training with lr=0.001, batch_size=64, dropout_rate=0.2
Validation loss: 4860.4346
Training with lr=0.001, batch_size=64, dropout_rate=0.3
Validation loss: 5658.7090
Training with lr=0.001, batch_size=128, dropout_rate=0.1
Validation loss: 4686.4517
Training with lr=0.001, batch_size=128, dropout_rate=0.2
Validation loss: 5257.7778
Training with lr=0.001, batch_size=128, dropout_rate=0.3
Validation loss: 5511.2886
Training with lr=0.0001, batch_size=32, dropout_rate=0.1
Validation loss: 10302.5166
Training with lr=0.0001, batch_size=32, dropout_rate=0.2
Validation loss: 10597.6182
Training with lr=0.0001, batch_size=32, dropout_rate=0.3
Validation loss: 10788.1377
Training with lr=0.0001, batch_size=64, dropout_rate=0.1
Validation loss: 10614.5957
Training with lr=0.0001, batch_size=64, dropout_rate=0.2
Validation loss: 10344.2920
Training with lr=0.0001, batch_size=64, dropout_rate=0.3
Validation loss: 11082.3740
Training with lr=0.0001, batch_size=128, dropout_rate=0.1
Validation loss: 11221.4014
Training with lr=0.0001, batch_size=128, dropout_rate=0.2
Validation loss: 11614.7246
Training with lr=0.0001, batch_size=128, dropout_rate=0.3
Validation loss: 11728.8730
Best hyperparameters: {'lr': 0.01, 'batch_size': 128, 'dropout_rate': 0.1} with validation loss: 3231.8926
검증 데이터 RMSE: 62.0871
검증 데이터 MAE: 38.6912
제출 파일이 생성되었습니다.


이 출력은 모델의 하이퍼파라미터 튜닝 과정과 그 결과를 보여줍니다. 여러 하이퍼파라미터 조합에 대해 모델을 훈련시키고, 검증 손실(validation loss)을 비교하여 가장 성능이 좋은 하이퍼파라미터 조합을 선택한 후, 검증 데이터를 통해 성능을 평가한 결과입니다.

출력의 각 부분을 단계별로 자세히 설명해드릴게요.

1. 하이퍼파라미터 튜닝 및 결과
이 부분에서는 학습률 (lr), 배치 크기 (batch_size), **드롭아웃 비율 (dropout_rate)**의 세 가지 하이퍼파라미터에 대해 조합을 시도하면서 모델을 훈련시키고, 각 하이퍼파라미터 조합에 대한 **검증 손실 (Validation Loss)**을 출력하고 있습니다.

하이퍼파라미터 조합과 검증 손실
학습률 (lr): 모델이 학습하는 속도를 제어하는 파라미터입니다. 너무 크면 학습이 불안정하고, 너무 작으면 학습이 너무 느려질 수 있습니다.
배치 크기 (batch_size): 모델이 한 번에 처리하는 데이터 샘플의 수입니다. 배치 크기가 크면 학습이 안정적이지만, 메모리 사용량이 커지고, 작은 배치는 더 많은 갱신을 하여 빠르게 최적화될 수 있습니다.
드롭아웃 비율 (dropout_rate): 모델의 과적합을 방지하기 위해 학습 중에 무작위로 일부 뉴런을 비활성화하는 비율입니다.
예를 들어:

lr=0.01, batch_size=32, dropout_rate=0.1일 때, 검증 손실은 5395.2891입니다.
lr=0.01, batch_size=128, dropout_rate=0.1일 때, 검증 손실은 3231.8926입니다.
위의 결과에서, lr=0.01, batch_size=128, dropout_rate=0.1일 때 검증 손실이 가장 낮다는 것을 확인할 수 있습니다.

베스트 하이퍼파라미터
하이퍼파라미터 튜닝 후, 가장 낮은 검증 손실을 기록한 조합은 다음과 같습니다:

학습률: 0.01
배치 크기: 128
드롭아웃 비율: 0.1
최적의 하이퍼파라미터 조합으로 선택된 이유는 이 조합이 검증 손실을 가장 낮추었기 때문입니다. 이때의 **검증 손실 (Validation Loss)**은 3231.8926으로, 다른 하이퍼파라미터 조합에 비해 월등히 낮습니다.

2. 검증 데이터 성능 평가
최적의 하이퍼파라미터 조합을 찾은 후, 해당 모델을 검증 데이터에 대해 평가한 결과가 출력됩니다. 이때, 모델의 성능을 평가하기 위해 **RMSE (Root Mean Squared Error)**와 **MAE (Mean Absolute Error)**를 사용합니다.

RMSE (Root Mean Squared Error)
plaintext
코드 복사
검증 데이터 RMSE: 62.0871
RMSE는 모델의 예측값과 실제 값 사이의 차이를 제곱하여 평균을 내고, 그 값의 제곱근을 구한 것입니다.
이 값은 모델의 예측값이 실제 값에서 얼마나 떨어져 있는지의 정도를 나타냅니다.
RMSE 값이 작을수록 모델의 예측이 실제 값에 가까워짐을 의미합니다.
여기서 RMSE는 62.0871로, 이는 모델이 예측한 대여 수가 실제 값과 평균적으로 약 62회 정도 차이가 있다는 것을 나타냅니다.
MAE (Mean Absolute Error)
plaintext
코드 복사
검증 데이터 MAE: 38.6912
MAE는 예측값과 실제 값 사이의 절대 차이의 평균입니다. 이 값은 모델의 예측 오차가 평균적으로 얼마나 큰지를 보여줍니다.
MAE 값도 낮을수록 좋은 모델입니다.
여기서 MAE는 38.6912로, 모델이 예측한 대여 수가 실제 값과 평균적으로 약 39회 정도 차이가 있다는 것을 의미합니다.
결론
모델의 검증 성능은 RMSE 62.0871과 MAE 38.6912로, 예측 오차가 상당히 작은 편입니다.
이는 모델이 주어진 데이터에 대해 잘 학습하고 예측하고 있음을 의미합니다.
3. 제출 파일 생성
plaintext
코드 복사
제출 파일이 생성되었습니다.
모델 학습이 끝난 후, 테스트 데이터에 대해 예측을 수행하고, 결과를 제출 파일로 저장합니다. 이 파일은 Kaggle과 같은 대회에서 제출용으로 사용됩니다.

제출 파일은 sample_submission.csv라는 이름으로 저장되며, datetime과 count 열을 포함합니다.
datetime: 테스트 데이터의 날짜 및 시간
count: 모델이 예측한 자전거 대여 수
결론
이번 과정에서:

하이퍼파라미터 튜닝을 통해 최적의 학습률, 배치 크기, 드롭아웃 비율을 찾았고,
해당 모델을 검증 데이터에 대해 평가하여 RMSE와 MAE 값을 계산하여 성능을 측정했습니다.
마지막으로, 모델의 예측 결과를 제출 파일로 저장하였습니다.
모델의 성능은 전반적으로 매우 좋았으며, 이 모델은 테스트 데이터에 대해서도 유효한 예측을 할 것으로 기대됩니다.
