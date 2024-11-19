# Bike-Sharing-Demand
https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
![image](https://github.com/user-attachments/assets/953ab93c-2e6b-4212-94c1-cac89ff2d00b)

![image](https://github.com/user-attachments/assets/4fae77c9-2bcc-41e5-9971-6bcc4c436b36)

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





새로운 분석을 바탕으로, 각 변수들의 관계를 더 직관적으로 시각화해 보겠습니다.

1. 라이브러리 설치 및 데이터 로드
먼저 필요한 라이브러리를 설치하고 데이터를 로드합니다:

bash
코드 복사
pip install pandas seaborn matplotlib
2. 전체 코드: Seaborn을 사용한 시각화와 분석
python
코드 복사
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn 스타일 설정
sns.set(style="whitegrid", palette="muted")

# CSV 파일을 pandas DataFrame으로 읽어들입니다.
data = pd.read_csv('train.csv')

# datetime을 pandas datetime 형식으로 변환
data['datetime'] = pd.to_datetime(data['datetime'])

# 'datetime'을 인덱스로 설정하여 시계열 데이터를 분석할 수 있게 만듭니다.
data.set_index('datetime', inplace=True)

# 데이터의 첫 몇 줄을 확인하여 정상적으로 로드되었는지 확인
print(data.head())

# 1. 시간대별 대여 수량 (Casual, Registered, Total Count)
plt.figure(figsize=(14, 8))
sns.lineplot(data=data[['casual', 'registered', 'count']], dashes=False, markers=True)
plt.title('Hourly Bike Rentals: Casual, Registered, and Total Counts', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Count of Bike Rentals', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. 온도(temp)와 체감 온도(atemp) 간의 관계 (산점도)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='temp', y='atemp', data=data, hue='season', palette='coolwarm', alpha=0.7)
plt.title('Temperature vs. Feels Like Temperature', fontsize=16)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Feels Like Temperature (°C)', fontsize=12)
plt.tight_layout()
plt.show()

# 3. 날씨(weather)와 대여 수(count) 간의 관계 (박스플롯)
plt.figure(figsize=(8, 6))
sns.boxplot(x='weather', y='count', data=data, palette='coolwarm')
plt.title('Bike Rentals vs Weather Condition', fontsize=16)
plt.xlabel('Weather Condition', fontsize=12)
plt.ylabel('Bike Rentals Count', fontsize=12)
plt.tight_layout()
plt.show()

# 4. 시간대별 기온(temp)과 습도(humidity) 변화 (시간별 선 그래프)
plt.figure(figsize=(14, 8))
sns.lineplot(data=data[['temp', 'humidity']], palette='tab10')
plt.title('Hourly Temperature and Humidity', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. 휴일(holiday)과 대여 수(count) 간의 관계 (박스플롯)
plt.figure(figsize=(8, 6))
sns.boxplot(x='holiday', y='count', data=data, palette='pastel')
plt.title('Bike Rentals on Holidays vs Non-Holidays', fontsize=16)
plt.xlabel('Holiday (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Bike Rentals Count', fontsize=12)
plt.tight_layout()
plt.show()

# 6. 기온(temp)과 대여 수(count) 간의 관계 (산점도)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='temp', y='count', data=data, hue='season', palette='viridis', alpha=0.7)
plt.title('Bike Rentals vs Temperature', fontsize=16)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Bike Rentals Count', fontsize=12)
plt.tight_layout()
plt.show()
3. 분석 설명 및 시각화
시간대별 대여 수량 (Casual, Registered, Total Count):

목표: 시간에 따른 자전거 대여 수의 변화를 시각화합니다. casual, registered, count 세 가지 대여 수를 비교하여 시계열의 패턴을 확인할 수 있습니다.
분석: count는 전체 대여 수를 나타내며, casual은 비회원 대여 수, registered는 회원 대여 수입니다. 이를 통해 특정 시간대(예: 아침, 저녁)에 회원과 비회원 대여 패턴이 어떻게 달라지는지 확인할 수 있습니다.
온도와 체감 온도 간의 관계 (산점도):

목표: 실제 온도(temp)와 체감 온도(atemp)의 관계를 시각화합니다.
분석: 체감 온도는 실제 온도 외에도 바람, 습도 등 다른 변수의 영향을 받기 때문에, 두 변수 간의 상관관계를 확인하고, 계절(season)에 따른 변화를 색으로 구분하여 계절적 차이를 관찰합니다.
날씨 조건(weather)과 대여 수(count) 간의 관계 (박스플롯):

목표: weather 변수(1: 맑음, 2: 흐림, 3: 비, 4: 눈)와 대여 수의 분포를 박스플롯으로 확인합니다.
분석: 날씨 조건이 자전거 대여 수에 미치는 영향을 분석할 수 있습니다. 예를 들어, 비가 오는 날씨에는 대여 수가 감소할 가능성이 높고, 맑은 날씨에는 대여 수가 증가하는 경향이 있을 수 있습니다.
시간대별 기온(temp)과 습도(humidity) 변화:

목표: 시간에 따른 온도(temp)와 습도(humidity)의 변화를 선 그래프로 확인합니다.
분석: 온도와 습도의 변화 패턴을 시각적으로 비교하여, 하루 동안 날씨가 어떻게 변하는지, 기온과 습도의 상관관계를 분석할 수 있습니다.
휴일 여부와 대여 수(count) 간의 관계 (박스플롯):

목표: 휴일(holiday)과 자전거 대여 수(count) 간의 관계를 분석합니다.
분석: 휴일과 비휴일의 자전거 대여 수 차이를 비교하여, 사람들이 휴일에 자전거를 더 많이 대여하는지, 아니면 주중에 더 많이 대여하는지 확인할 수 있습니다.
기온과 대여 수(count) 간의 관계 (산점도):

목표: 기온(temp)과 자전거 대여 수(count) 간의 관계를 시각화합니다.
분석: 기온이 높을수록 자전거 대여 수가 증가하는 경향이 있을 수 있으며, 이를 시각적으로 분석할 수 있습니다.
4. 결과 및 분석 인사이트
시간대별 대여 수: 아침 7-9시와 저녁 5-7시에 대여 수가 급증하는 패턴을 볼 수 있을 것입니다. 비회원(casual) 대여는 주로 낮 시간대에 많고, 회원(registered) 대여는 아침과 저녁 시간대에 집중될 수 있습니다.

온도와 체감 온도 관계: 체감 온도는 실제 온도와 비슷하지만 약간 차이가 날 수 있습니다. 특히 바람이 불거나 습도가 높을 때 체감 온도가 실제 온도보다 낮거나 높게 느껴집니다.

날씨와 대여 수: 비나 눈이 오는 날에는 대여 수가 줄어들고, 맑은 날에는 대여 수가 증가하는 경향을 보일 수 있습니다.

휴일 여부: 휴일에는 사람들이 자전거를 더 많이 대여하는 경향이 있을 수 있으며, 이는 여가 활동이 증가하기 때문입니다.

기온과 대여 수: 기온이 높을수록 자전거 대여 수가 증가하는 경향이 나타날 가능성이 큽니다.

이러한 시각화와 분석을 통해 자전거 대여 패턴과 환경적 요인 간의 관계를 이해할 수 있습니다.

![image](https://github.com/user-attachments/assets/d440dc83-c642-40a0-8082-88b2c4f995db)


![image](https://github.com/user-attachments/assets/df5e47b1-9f2d-4b58-bd49-df4472fc35bb)

![image](https://github.com/user-attachments/assets/eac6c2ee-8b48-4220-92bb-2c708a9e913d)

![image](https://github.com/user-attachments/assets/b65c1c64-6ae5-4164-994b-2ad31409af01)


![image](https://github.com/user-attachments/assets/cf2938dd-3887-463e-81b9-1eb0fd00135e)

![image](https://github.com/user-attachments/assets/ccc92764-d8e2-4244-862c-d239f985317d)





![image](https://github.com/user-attachments/assets/5097ae48-f5d3-4a58-9541-a744d8a1ce22)

![image](https://github.com/user-attachments/assets/ea343e8f-c5c4-4aed-ae96-c2e0d51b2663)




