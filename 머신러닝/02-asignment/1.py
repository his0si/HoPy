#라이브러리 로딩
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier as DT, plot_tree #아이리스 품종 분류 데이터
from sklearn.model_selection import train_test_split

plt.rcParams['axes.unicode_minus']=False
plt.rcParams["font.family"]='NanumBarunGothic'

#데이터 로드 및 분할
data=load_iris()
#X_train, X_test, Y_train, Y_test에 데이터를 저장합니다.
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target,random_state=0)

#의사결정나무 적합
clf0=DT(max_depth=3, ramdom_state=0)
clf0.fit(X_train,Y_train)

#의사결정나무 모형 시각화
plt.figure(figsize=(5,4),dpi=200)
plot_tree(clf0,
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True)

plt.show()
