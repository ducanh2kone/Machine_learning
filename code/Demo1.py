# nhận dạng chữ số sử dung MPLclassifier
from unicodedata import digit
from matplotlib import pyplot as plt # draw result  
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
 # load dataset

digit = datasets.load_digits()
X  = digit['data']
# print(X.shape)
Y = digit['target']
# print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.3)
# build  nn model
from sklearn.neural_network import MLPClassifier 

mlp = MLPClassifier(hidden_layer_sizes=(30,20),max_iter= 100) # learning rate: nho nhung chinh xac, lr= lon--> overfiting
mlp.fit(X_train,Y_train) # input features + labels

y_predict = mlp.predict(X_test)
print('accurancy:', metrics.accuracy_score(Y_test,y_predict))
metrics.plot_confusion_matrix(mlp,X_test,Y_test )
plt.show()
# draw learning curve
# from sklearn.model_selection import learning_curve
# train_sizes, train_scores, test_scores = learning_curve(MLPClassifier(), X, Y, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)

# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
# plt.subplots(1, figsize=(10,10))
# plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
# plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# plt.title("Learning Curve")
# plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
# plt.tight_layout()
# plt.show()