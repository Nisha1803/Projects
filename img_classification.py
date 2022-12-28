# CS 6307 - Introduction to Big Data Management and Analytics 
# Project 1 
# Nisha and Sakshi Malhotra  
 
# Image Recognition using Deep Neural Network 
# 2 cases : 3 Layer Neural Network and 4 Layer Neural Network
 
#Import Libraries for pre processing data 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
%matplotlib inline
import sys
import pyspark
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
 
# Loading the dataset using keras.datasets 
from keras.datasets import fashion_mnist
(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
 
# Printing the size and the shape of the testing and training data 
print('Shape of Training set (X): ' + str(train_X.shape))
print('Shape of Training set (Y): ' + str(train_y.shape))
print('Shape of Test set (X):  '  + str(test_X.shape))
print('Shape of Test set (Y):  '  + str(test_y.shape))
 
Shape of Training set (X): (60000, 28, 28)
Shape of Training set (Y): (60000,)
Shape of Test set (X):  (10000, 28, 28)
Shape of Test set (Y):  (10000,)
# We have a training set of 60000 and a test set of 10000 where each image is in grayscale of size 28*28 
# Visualizing some of the data 
# Looking at first 4 training images 
for i in range(4):  
    plot.subplot(2,2,1+i) #image matrix size 
    plot.imshow(train_X[i],cmap=plot.get_cmap('gray'))
plot.show()

# Looking at first 4 test images 
for i in range(4):  
    plot.subplot(2,2,1+i) #image matrix size 
    plot.imshow(test_X[i],cmap=plot.get_cmap('gray'))
plot.show()

# We reshape data in python and download them as csv files and upload them to databricks 
# Uploading training and testing data as RDD 
train_im = sc.textFile("/FileStore/tables/fashion_mnist_images_train.csv", 1)
train_lb = sc.textFile("/FileStore/tables/fashion_mnist_labels_train.csv", 1)
test_im = sc.textFile("/FileStore/tables/fashion_mnist_images_test.csv", 1)
test_lb = sc.textFile("/FileStore/tables/fashion_mnist_labels_test.csv", 1)
# Reformatting the data 
x_tr = train_im.map(lambda x : np.fromstring(x, dtype=float, sep=' ').reshape(1, 784)).zipWithIndex().map(lambda x: (str(x[1]), x[0]))
y_tr = train_lb.map(lambda x : np.fromstring(x, dtype=float, sep=' ').reshape(1, 10)).zipWithIndex().map(lambda x: (str(x[1]), x[0]))
x_tt = test_im.map(lambda x : np.fromstring(x, dtype=float, sep=' ').reshape(1, 784)).zipWithIndex().map(lambda x: (str(x[1]), x[0]))
y_tt = test_lb.map(lambda x : np.fromstring(x, dtype=float, sep=' ').reshape(1, 10)).zipWithIndex().map(lambda x: (str(x[1]), x[0]))
 
# Combining all of training and testing data 
train1 = x_tr.join(y_tr).map(lambda x: x[1])
test1 = x_tt.join(y_tt).map(lambda x: x[1])
# Defining Activation Function 
 
# Defining a general function to call different activation function 
 
def fun(x,f):
    return f(x)
# 1. Sigmoid Function 
 
def sig(x):
    return 1/(1+np.exp(-x))
 
# Derivative of sigmoid function 
 
def sigprime(x):
    s = sig(x)
    return s*(1-s)
# 2. Hyperbolic Tangent Function 
 
def tanh(x):
    return np.tanh(x);
 
#Dervative of hyperbolic tangent function 
 
def thprime(x):
    return 1-np.tanh(x)**2
# Forward propagation (3 Layers)
 
# Adding weights and bias to the input layer 
 
def inp1(x,w,b):
    return np.dot(x,w)+b
 
# Predicting the function 
 
def predict(x,W1,B1,W2,B2):
    return sig(inp1(tanh(inp1(x,W1,B1)),W2,B2))
# Backward Propagation (3 Layers)
# Derivative of Error w.r.t W1,B1,W2,B2
 
# Derivative w.r.t B2
def derB2(ypr,y,yh,fprime):
    return (ypr-y)*fprime(yh)
 
#Derivative w.r.t W2
def derW2(h,dB2):
    return np.dot(h.T,dB2)
 
#Derivative w.r.t B1
 
def derB1(hh,dB2,W2,fprime):
    return np.dot(dB2,W2.T)*fprime(hh)
 
#Derivative w.r.t W1
 
def derW1(x,dB1):
    return np.dot(x.T,dB1)
# Cost Function (Error)
 
def error(ypr,y):
    return 0.5*np.sum(np.power(ypr-y,2))
#####################################################
#Make changes here 
#####################################################
 
# Mini Batch Gradient Descent 
 
#Parameters 
#Number of iterations
iteration = 100 
#Learning Rate
learn_r = 0.4
 
#number of neurones in input layer
inputlayer = 784 
#number of neurones in hidden layer 
hiddenlayer = 64
#number of neurones in output layer 
outputlayer = 10 
 
# 4 Layer Neural Network 
#number of neurones in hidden layer 1
hiddenlayer1 = 248
#number of neurones in hidden layer 2
hiddenlayer2 = 124
 
 
# 3 Layer Neural Network (Parameter Initialization)
 
W1 = np.random.rand(inputlayer,hiddenlayer)-0.5
W2 = np.random.rand(hiddenlayer,outputlayer)-0.5
B1 = np.random.rand(1,hiddenlayer)-0.5
B2 = np.random.rand(1,outputlayer)-0.5
#4 Layer Neural Network (Parameter Initialization)
 
W1_4 = np.random.rand(inputlayer,hiddenlayer1)-0.5
W2_4 = np.random.rand(hiddenlayer1,hiddenlayer2)-0.5
W3_4 = np.random.rand(hiddenlayer2,outputlayer)-0.5
B1_4 = np.random.rand(1,hiddenlayer1)-0.5
B2_4 = np.random.rand(1,hiddenlayer2)-0.5
B3_4 = np.random.rand(1,outputlayer)-0.5
# Update at every Iteration
costfunc = []
accuracy = []
costfunc4 = []
accuracy4 = []
# 3 Layer Neural Network
# Mini Batch Implementation 
for i in range(iteration):
    #Forward Propagation
    gradientfwd1 = train1.sample(False,0.7).map(lambda a: (a[0] , inp1(a[0],W1,B1),a[1])).map(lambda a: (a[0],a[1],fun(a[1],tanh),a[2])).map(lambda a:(a[0],a[1],a[2],inp1(a[2],W2,B2),a[3])).map(lambda a: (a[0],a[1],a[2],a[3],fun(a[3],sig),a[4]))
    
    #Backward Propagation
    gradientbwd1 = gradientfwd1.map(lambda a: (a[0],a[1],a[2],error(a[4],a[5]),derB2(a[4],a[5],a[3],sigprime),int(np.argmax(a[4])==np.argmax(a[5])))).map(lambda a: (a[0], a[1], a[3], a[4],  derW2(a[2], a[4]) ,a[5])).map(lambda a: (a[0], a[2], a[3], a[4],  derB1(a[1],a[3], W2, thprime) ,a[5])).map(lambda a: (a[1], a[2], a[3], a[4], derW1(a[0], a[4]) ,a[5], 1)).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4], a[5] + b[5], a[6] + b[6]))
    
    #Cost and Accuracy 
    num_images = gradientbwd1[6] #no. of images in the mini batch
    err = gradientbwd1[0]/num_images #Error over the mini batch 
    acc = gradientbwd1[5]/num_images # Accuracy over the mini batch 
    
    costfunc.append(err)
    accuracy.append(acc)
    
    
    # Gradients 
    DB2 = gradientbwd1[1]/num_images
    DW2 = gradientbwd1[2]/num_images
    DB1 = gradientbwd1[3]/num_images
    DW1 = gradientbwd1[4]/num_images
    
    # Updating Parameters 
    B2 -= learn_r * DB2
    W2 -= learn_r * DW2
    B1 -= learn_r * DB1
    W1 -= learn_r * DW1
    
    
    print(f"   Number of iter: {i+1}/{iteration} ,  Accuracy: {accuracy[i]*100} , Error: {costfunc[i]} , Batchsize:{num_images}")
 
   Number of iter: 1/100 ,  Accuracy: 12.333436628447474 , Error: 1.811637024485621 , Batchsize:41951
   Number of iter: 2/100 ,  Accuracy: 13.428496532659723 , Error: 1.0285445507124007 , Batchsize:41963
   Number of iter: 3/100 ,  Accuracy: 14.89828527819513 , Error: 0.6939785015252122 , Batchsize:41931
   Number of iter: 4/100 ,  Accuracy: 16.697344201945832 , Error: 0.5837330073576409 , Batchsize:41833
   Number of iter: 5/100 ,  Accuracy: 18.35366288608317 , Error: 0.5342494913521437 , Batchsize:42057
   Number of iter: 6/100 ,  Accuracy: 19.930708810896753 , Error: 0.504402091991952 , Batchsize:42141
   Number of iter: 7/100 ,  Accuracy: 21.644219577662177 , Error: 0.4861410081109148 , Batchsize:42099
   Number of iter: 8/100 ,  Accuracy: 23.025514953337947 , Error: 0.47515271160512634 , Batchsize:41897
   Number of iter: 9/100 ,  Accuracy: 24.792405243998193 , Error: 0.4656393563804828 , Batchsize:42029
   Number of iter: 10/100 ,  Accuracy: 26.12657864523536 , Error: 0.45812724679586303 , Batchsize:41808
   Number of iter: 11/100 ,  Accuracy: 27.484138720602967 , Error: 0.45104136627859476 , Batchsize:41926
   Number of iter: 12/100 ,  Accuracy: 29.069022010114683 , Error: 0.4446700297947389 , Batchsize:42117
   Number of iter: 13/100 ,  Accuracy: 30.536330122487808 , Error: 0.43906637027281653 , Batchsize:42045
   Number of iter: 14/100 ,  Accuracy: 31.85540672826684 , Error: 0.433020427176523 , Batchsize:41883
   Number of iter: 15/100 ,  Accuracy: 33.42855782507261 , Error: 0.4263392559027782 , Batchsize:42006
   Number of iter: 16/100 ,  Accuracy: 34.23269500011865 , Error: 0.42279321535855624 , Batchsize:42141
   Number of iter: 17/100 ,  Accuracy: 35.61145087214145 , Error: 0.41679030891269864 , Batchsize:42023
   Number of iter: 18/100 ,  Accuracy: 36.749696666904576 , Error: 0.4115939673787639 , Batchsize:42033
   Number of iter: 19/100 ,  Accuracy: 37.70000713249804 , Error: 0.4073644173920914 , Batchsize:42061
Cancelled
#Confusion Matrix 
def metr(ypr, y):
    cm = multilabel_confusion_matrix(y, ypr)
    return (cm)
Cancelled
# Use the trained model over the Testset and get Confusion matrix per class
metrics = test1.map(lambda x: metr(np.round(predict(x[0], W1, B1, W2, B2)), x[1])).reduce(lambda x, y: x + y)
for label, label_metrics in enumerate(metrics):
    
    print(f"\n---- Image {label} ------\n")
    trueneg, falsepos, falseneg, truepos = label_metrics.ravel()
    print("True Positive:", truepos, "False Postivie:", falsepos, "False Negative:", falseneg, "True Negative:", trueneg)
 
    precision = truepos / (truepos + falsepos + 0.000001)
    print(f"\nPrecision : {precision}")
 
    recall = truepos / (truepos + falseneg + 0.000001)
    print(f"Recall: {recall}")
 
    F1 = 2 * (precision * recall) / (precision + recall + 0.000001)
    print(f"F1 score: {F1}")
Command skipped
# Predicting the function (4 Layers) 
 
def predict_4(x,W1,B1,W2,B2,W3,B3):
    return sig(inp1(tanh(inp1(tanh(inp1(x,W1,B1)),W2,B2)),W3,B3))
# Backward Propagation 
# Derivative of Error w.r.t W1,B1,W2,B2
 
# Derivative w.r.t B3
def derB3_4(ypr,y,yh,f1prime):
    return (ypr-y)*f1prime(yh)
 
# Derivative w.r.t W3
def derW3_4(u,dB3):
    return np.dot(u.T,dB3)
 
 
#Derivative w.r.t B2
def derB2_4(uh,dB3,W3,fprime):
    return np.dot(dB3,W3.T)*fprime(uh)
 
#Derivative w.r.t W2
def derW2_4(h,dB2):
    return np.dot(h.T,dB2)
 
#Derivative w.r.t B1
def derB1_4(hh,dB2,W2,fprime):
    return np.dot(dB2,W2.T)*fprime(hh)
 
#Derivative w.r.t W1
def derW1_4(x,dB1):
    return np.dot(x.T,dB1)
 
for i in range(iteration):
    gradientfwd = train1.sample(False,0.7).map(lambda a: (a[0], inp1(a[0],W1_4,B1_4),a[1])).map(lambda a: (a[0],a[1],fun(a[1],tanh),a[2])).map(lambda a:(a[0],a[1],a[2],inp1(a[2],W2_4,B2_4),a[3])).map(lambda a: (a[0],a[1],a[2],a[3],fun(a[3],tanh),a[4])).map(lambda a:(a[0],a[1],a[2],a[3],a[4],inp1(a[4],W3_4,B3_4),a[5])).map(lambda a: (a[0],a[1],a[2],a[3],a[4],a[5],fun(a[5],sig),a[6]))
    
    gradientbwd = gradientfwd.map(lambda a: (a[0],a[1],a[2],a[3],a[4],a[5],error(a[6],a[7]),derB3_4(a[6],a[7],a[5],sigprime),int(np.argmax(a[6])==np.argmax(a[7])),a[7])).map(lambda a: (a[0], a[1], a[2],a[3], a[4],a[6],a[7], derW3_4(a[4], a[7]) ,a[8],a[9])).map(lambda a : (a[0],a[1],a[2],a[3],a[5],a[6],derB2_4(a[3],a[6],W3_4,thprime),a[8],a[9],a[7])).map(lambda a:(a[0],a[1],a[2],a[4],a[6],derW2_4(a[2],a[6]),a[7],a[8],a[9],a[5])).map(lambda a: (a[0],a[1],a[3],a[4],derB1_4(a[1],a[4],W2_4,thprime),a[6],a[7],a[8],a[9],a[5])).map(lambda a:(a[0],a[2],a[4],derW1_4(a[0],a[4]),a[5],a[6],a[7],a[8],a[9],a[3])).map(lambda a: (a[1],a[2],a[3],a[9],a[8],a[7],a[6],a[4],1)).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4], a[5] + b[5], a[6] + b[6],a[7] + b[7],a[8] + b[8]))
    
  
    num_images4 = gradientbwd[8] #no. of images 
    err4 = gradientbwd[0]/num_images4
    acc4 = gradientbwd[7]/num_images4
    
    costfunc4.append(err4)
    accuracy4.append(acc4)
    
    # Gradients 
    DB3_4 = gradientbwd[5]/num_images4
    DW3_4 = gradientbwd[6]/num_images4
    
    DB2_4 = gradientbwd[3]/num_images4
    DW2_4 = gradientbwd[4]/num_images4
    DB1_4 = gradientbwd[1]/num_images4
    DW1_4 = gradientbwd[2]/num_images4
    
    
    # Updating Parameters 
    B3_4 -= learn_r * DB3_4
    W3_4 -= learn_r * DW3_4
    
    B2_4 -= learn_r * DB2_4
    W2_4 -= learn_r * DW2_4
    B1_4 -= learn_r * DB1_4
    W1_4 -= learn_r * DW1_4
    
    print(f"   Number of iter: {i+1}/{iteration} ,  Accuracy: {accuracy4[i]*100} , Error: {costfunc4[i]} , Batchsize:{num_images4}")
 
 
   
   Number of iter: 1/100 ,  Accuracy: 8.625908711714933 , Error: 2.140308789627665 , Batchsize:41955
   Number of iter: 2/100 ,  Accuracy: 10.88649189227802 , Error: 0.7295964688144431 , Batchsize:41997
   Number of iter: 3/100 ,  Accuracy: 16.964883743182472 , Error: 0.5292135317796627 , Batchsize:41804
   Number of iter: 4/100 ,  Accuracy: 21.041106324232974 , Error: 0.4923456341440908 , Batchsize:42013
   Number of iter: 5/100 ,  Accuracy: 24.6875669499393 , Error: 0.4714899992451478 , Batchsize:42009
   Number of iter: 6/100 ,  Accuracy: 27.929385453593774 , Error: 0.4539344941356592 , Batchsize:42031
   Number of iter: 7/100 ,  Accuracy: 30.990589518097995 , Error: 0.43864742475506346 , Batchsize:42187
   Number of iter: 8/100 ,  Accuracy: 33.388830748739416 , Error: 0.4260976879105722 , Batchsize:42044
   Number of iter: 9/100 ,  Accuracy: 36.20165977088465 , Error: 0.4124038054324163 , Batchsize:41813
   Number of iter: 10/100 ,  Accuracy: 38.32724512548034 , Error: 0.4021906785572542 , Batchsize:42158
   Number of iter: 11/100 ,  Accuracy: 40.634383489824124 , Error: 0.39156941292271297 , Batchsize:41962
   Number of iter: 12/100 ,  Accuracy: 42.643700038066235 , Error: 0.3811250043090401 , Batchsize:42032
   Number of iter: 13/100 ,  Accuracy: 44.13319994283264 , Error: 0.3743830748676535 , Batchsize:41982
   Number of iter: 14/100 ,  Accuracy: 45.32736962314427 , Error: 0.3676054823789155 , Batchsize:42032
   Number of iter: 15/100 ,  Accuracy: 47.23137142584372 , Error: 0.35874990481496793 , Batchsize:41898
   Number of iter: 16/100 ,  Accuracy: 48.08925548906954 , Error: 0.35233543429602693 , Batchsize:41947
   Number of iter: 17/100 ,  Accuracy: 49.00679560899112 , Error: 0.34755659616645007 , Batchsize:42086
   Number of iter: 18/100 ,  Accuracy: 50.13459751768826 , Error: 0.3420254277657939 , Batchsize:41977
   Number of iter: 19/100 ,  Accuracy: 50.848664688427306 , Error: 0.3378514692796237 , Batchsize:42125
   Number of iter: 20/100 ,  Accuracy: 51.638426157622554 , Error: 0.33234811027042016 , Batchsize:41961
   Number of iter: 21/100 ,  Accuracy: 52.35226893957431 , Error: 0.32799413053756016 , Batchsize:42002
   Number of iter: 22/100 ,  Accuracy: 53.165759901166524 , Error: 0.32455380357922287 , Batchsize:42091
   Number of iter: 23/100 ,  Accuracy: 54.090313356368405 , Error: 0.3193955066884555 , Batchsize:41965
   Number of iter: 24/100 ,  Accuracy: 54.512273670650536 , Error: 0.3172167059874579 , Batchsize:41919
   Number of iter: 25/100 ,  Accuracy: 55.15320334261838 , Error: 0.31251673717736844 , Batchsize:42003
   Number of iter: 26/100 ,  Accuracy: 55.945945945945944 , Error: 0.307972237428906 , Batchsize:41810
Cancelled
# Plot of cost over iterations
plot.subplot(2, 1, 1)
plot.plot(costfunc4)
plot.title("Error per Iteration")
plot.xlabel("Number of iterations")
plot.ylabel("Cost Function")
plot.show()
 
# Plot of accuracy over iterations
plot.subplot(2, 1, 2)
plot.plot(accuracy4)
plot.title("Accuracy per Iteration")
plot.xlabel("Number of iterations")
plot.ylabel("Accuracy")
plot.show()


# Use the trained model over the Testset and get Confusion matrix per class
metrics = test1.map(lambda x: metr(np.round(predict_4(x[0], W1_4, B1_4, W2_4, B2_4 ,W3_4,B3_4 )), x[1])).reduce(lambda x, y: x + y)
for label, label_metrics in enumerate(metrics):
    
    print(f"\n---- Image {label} ------\n")
    trueneg, falsepos, falseneg, truepos = label_metrics.ravel()
    print("True Positive:", truepos, "False Positive:", falsepos, "False Negative:", falseneg, "True Negative:", trueneg)
 
    precision = truepos / (truepos + falsepos + 0.000001)
    print(f"\nPrecision : {precision}")
 
    recall = truepos / (truepos + falseneg + 0.000001)
    print(f"Recall: {recall}")
 
    F1 = 2 * (precision * recall) / (precision + recall + 0.000001)
    print(f"F1 score: {F1}")

---- Image 0 ------

True Positive: 601 False Positive: 175 False Negative: 399 True Negative: 8825

Precision : 0.7744845350844272
Recall: 0.600999999399
F1 score: 0.6768013089938952

---- Image 1 ------

True Positive: 780 False Positive: 70 False Negative: 220 True Negative: 8930

Precision : 0.9176470577439446
Recall: 0.77999999922
F1 score: 0.8432427456189925

---- Image 2 ------

True Positive: 378 False Positive: 228 False Negative: 622 True Negative: 8772
