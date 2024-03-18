#데이터를 분리하는 방법1 홀드아웃 방법

from sklearn import datasets
from sklearn.model_selection import train_test_split

#Iris 데이터셋을 읽어들입니다
iris = datasets.load_iris()
X = iris.data
y = iris.target

#X_train, X_test, y_train, y_test에 데이터를 저장합니다.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

print("X_train:" , X_train.shape)
print("y_train:" , y_train.shape)
print("X_test:" , X_test.shape)
print("y_test" , y_test)