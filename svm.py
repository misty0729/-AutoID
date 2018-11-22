import imp
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pickle


pkl_file = open('data.pkl', 'rb')
data_dic = pickle.load(pkl_file)
# import our data
X = data_dic["data"]
y = data_dic["target"]
# # 标准化 正太分布
# x = StandardScaler().fit_transform(iris.data)
# print("StandardScaler:")
# print(x)
# x_maxmin = MinMaxScaler().fit_transform(iris.data)
# print("maxmin:")
# print(x_maxmin)
# # print(fit(iris.target.reshape((-1, 1))))
# x_onehot = OneHotEncoder().fit_transform(iris.target.reshape((-1, 1)))
# print("onehot")
# print(x_onehot)
# y = iris.target
# print("mytarget")
# print(y)

# split the data to  7:3
X_train, X_test, y_train, y_test = ts(X, y, test_size=0.3)

# # select different type of kernel function and compare the score

# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf', gamma='scale')
print(clf_rbf)
clf_rbf.fit(X_train, y_train)
score_rbf = clf_rbf.score(X_test, y_test)
print("The score of rbf is : %f" % score_rbf)

# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear', gamma='scale')
print(clf_linear)
clf_linear.fit(X_train, y_train)
score_linear = clf_linear.score(X_test, y_test)
print("The score of linear is : %f" % score_linear)

# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly', gamma='scale')
print(clf_poly)
clf_poly.fit(X_train, y_train)
score_poly = clf_poly.score(X_test, y_test)
print("The score of poly is : %f" % score_poly)
