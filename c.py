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
