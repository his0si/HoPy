#데이터를 분리하는 방법2 K-분할 교차검증

# 코드의 실행에 필요한 모듈을 import 합니다
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

# Iris 데이터셋을 읽어들입니다.
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 머신러닝 알고리즘 SVM을 사용합니다
svc = svm.SVC(C=1, kernel="rbf", gamma=0.001)

# 교차 검증을 이용하여 점수를 구합니다
# 내부에서는 X_train, X_test, y_train, y_test 로 분할 처리됩니다
scores = cross_val_score(svc, X, y, cv=5)

#훈련 데이터와 테스트 데이터의 크기를 확인합니다
print(scores)
print("average score: ", scores.mean())