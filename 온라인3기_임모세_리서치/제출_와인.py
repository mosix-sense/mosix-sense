from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

# 손글씨 데이터 불러오기
wine = load_wine()

# Feature Data 지정하기
X = wine.data

# Label Data 지정하기
y = wine.target

# Feature Data 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Feature Data와 Label Data를 학습 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 의사결정트리 모델 생성
dt = DecisionTreeClassifier(random_state=42)

# 랜덤포레스트 모델 생성
rf = RandomForestClassifier(random_state=42)

# SVM 모델 생성
svm = SVC(random_state=42)

# SGD Classifier 모델 생성
sgd = SGDClassifier(random_state=42)

# Logistic Regression 모델 생성
lr = LogisticRegression(random_state=42, max_iter=10000)

# 각 모델 학습 및 예측
models = {'Decision Tree': dt, 'Random Forest': rf, 'SVM': svm, 'SGD Classifier': sgd, 'Logistic Regression': lr}
for name, model in models.items():
    print(f"Model: {name}")
    # 모델 학습
    model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = model.predict(X_test)

    # 성능 지표 출력
    print(classification_report(y_test, y_pred))