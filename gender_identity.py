import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path # thu viện quản lý đường dẫn

# load image
img_folder = Path('data/GenderDataSet') # tao 1 thư mục từ đường dẫn này
test_folder = img_folder / 'test'    # khai báo đường dẫn thư mục test_folder
train_folder = img_folder / 'train' # note chú ý khoảng trăng trong '  '


test_img = list(test_folder.glob('*.png')) # list file cụ thể ra,glod tìm tên file theo đặc điểm chung của tên fi
train_img = list(train_folder.glob('*.png'))# type ò train_img tes_img -->> list
# create training data
num_train_img=len(train_img)
# tạo 2 nd array trống để chèn data sau khi stack
X_train = np.empty((0,2700),dtype=float)
Y_train = np.empty(num_train_img, dtype= np.uint8)

# print(test_img)
for i in range(num_train_img):
    img_path = str(train_img[i]) # đường dẫn từng img do đường dẫn img đang ở dạng PosixPath ta dùng str() để ép kiểu lại
    # print (img_path)
    #load image
    img=cv.imread(img_path,cv.IMREAD_GRAYSCALE)
    # cv.imshow('img',cv.imread(img_path,cv.IMREAD_GRAYSCALE))
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    # preprocessing
    img = img.astype(float) #đinh dạng img đang ở unit8: số nguyên nếu normalize thì giá trị ouput sẽ tự làm tròn thành số nguyên là 0 hoặc 1
    cv.normalize(img,img,0,1,cv.NORM_MINMAX)
    # print(img)
    img = np.reshape(img,(1,2700))
    # print(img.shape)
    X_train = np.vstack((X_train,img))
    # create Y_train
    if train_img[i].stem[0]== 'f':
        Y_train[i]= 0
    else:
        Y_train[i] = 1

# print(X_train.shape)
# print(Y_train)
# create NN and train
from sklearn.neural_network import MLPClassifier 
mlp = MLPClassifier(hidden_layer_sizes=(15,15),max_iter=500)
mlp.fit(X_train,Y_train)

# evaluate the network 
from sklearn import metrics
metrics.plot_confusion_matrix(mlp,X_train,Y_train)
plt.show()

# save model for using for other data
import pickle
filename = "gender_detect.sav"
pickle.dump(mlp,open(filename,"wb"))
 
    
