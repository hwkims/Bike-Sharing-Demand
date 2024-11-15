import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# 데이터 로딩 및 전처리
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 날짜 데이터 처리
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

# 유용한 특성 추출
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['dayofweek'] = train['datetime'].dt.dayofweek

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour
test['dayofweek'] = test['datetime'].dt.dayofweek

# 특성 선택 및 전처리
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'month',
            'day', 'hour', 'dayofweek']
X = train[features]
y = train['count']

# 특성 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# PyTorch 모델 정의
class RegressionModel(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


# 하이퍼파라미터 튜닝을 위한 함수 정의
def train_model(lr, batch_size, dropout_rate):
    model = RegressionModel(input_dim=X_train.shape[1], dropout_rate=dropout_rate)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 데이터 텐서 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # 배치 학습을 위한 DataLoader 생성
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 훈련
    num_epochs = 100
    best_val_loss = float('inf')  # 초기화
    for epoch in range(num_epochs):
        model.train()

        for i, (inputs, targets) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 검증
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()

    return best_val_loss


# 하이퍼파라미터 범위 설정
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [32, 64, 128]
dropout_rates = [0.1, 0.2, 0.3]

# 하이퍼파라미터 튜닝
best_loss = float('inf')
best_params = None

for lr in learning_rates:
    for batch_size in batch_sizes:
        for dropout_rate in dropout_rates:
            print(f"Training with lr={lr}, batch_size={batch_size}, dropout_rate={dropout_rate}")
            val_loss = train_model(lr, batch_size, dropout_rate)
            print(f"Validation loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = {'lr': lr, 'batch_size': batch_size, 'dropout_rate': dropout_rate}

print(f"Best hyperparameters: {best_params} with validation loss: {best_loss:.4f}")

# 최적의 하이퍼파라미터로 최종 모델 학습
model = RegressionModel(input_dim=X_train.shape[1], dropout_rate=best_params['dropout_rate'])
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
criterion = nn.MSELoss()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

# 최종 모델 훈련
num_epochs = 200
for epoch in range(num_epochs):
    model.train()

    for i, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 검증 성능 평가
model.eval()
with torch.no_grad():
    val_pred = model(X_val_tensor).view(-1).numpy()

rmse = np.sqrt(mean_squared_error(y_val, val_pred))
mae = np.mean(np.abs(y_val - val_pred))

print(f'검증 데이터 RMSE: {rmse:.4f}')
print(f'검증 데이터 MAE: {mae:.4f}')

# 테스트 데이터 예측
X_test = test[features]
X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    test_pred = model(X_test_tensor).view(-1).numpy()

# 제출 파일 생성
submission = pd.DataFrame({
    'datetime': test['datetime'],
    'count': test_pred
})

submission.to_csv('sample_submission.csv', index=False)
print('제출 파일이 생성되었습니다.')
