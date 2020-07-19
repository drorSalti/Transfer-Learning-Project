from scipy import io
import pandas as pd
from keras.optimizers import SGD, Adam, RMSprop
import scipy
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from keras.layers import Input, Dense, Flatten, Dropout, GlobalMaxPooling2D
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import precision_recall_curve, average_precision_score,confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

""" Need to install mlxtend package"""

def getParameters():
   """
   function that returns a list of the parameters
   :return: list of the parameters of the model
   """
   p = {}
   p['data_path'] = 'C:/Users/dsalt/OneDrive/Desktop/FlowerData'
   p['test_images_indices'] = list(range(301,473))
   p['train_images_indices'] = [x for x in list(range(1,472)) if x not in p['test_images_indices']]
   p['classes'] = 1
   p['batch'] = 30
   p['epochs'] = 15
   p['s'] = 224
   p['optimizer'] = 'adam'
   p['folds'] = 5
   p['learnRate'] = 0.0001
   p['numOfAug'] = 2
   p['aug'] = False
   p['valRate'] = 0.1
   p['dropRate'] = 0.0
   p['layers'] = True
   p['flat'] = True
   p['max'] = False
   p['decay'] = 0.0
   p['threshold'] = 0.7
   return p


def getData(data_path, s):
   """
   function that read the photos from the computer
   :param data_path: data_path: string- address of the folder with the photos
   :param s:  int- the size of the images
   :return: data set and its following labels
   """
   print("Stage 1: Importing Data")
   x = scipy.io.loadmat(data_path+'/FlowerDataLabels.mat')
   labels=x['Labels'].T
   dataOriginal = []
   data = []
   for i in range(1,labels.shape[0]+1):
       tmp = cv2.imread(data_path + '/' + str(i) + '.jpeg')
       tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
       tmp = cv2.resize(tmp, (s, s))
       dataOriginal.append(tmp)
       data.append(preprocess_input(tmp))
   data = np.asarray(data)
   dataOriginal = np.asarray(dataOriginal)
   return data, labels, dataOriginal


def splitData(data, labels, trainIdxList, testIdxList):
   """
   function that split the data
   :param data: matrix of all the data
   :param labels: vector of all the labels
   :param trainIdxList:  list of indices for the train set
   :param testIdxList: list of indices for the test set
   :return: train set and train labels, test set and test labels
   """
   testIdxList = [x-1 for x in testIdxList]
   trainIdxList = [x-1 for x in trainIdxList]
   testData= data[testIdxList]
   testLabels= labels[testIdxList]
   trainData=data[trainIdxList]
   trainLabels=labels[trainIdxList]
   return trainData, testData, trainLabels, testLabels


def addLayers(last_layer, dropout, outputNeurons):
   """
   function that prepare 2 more layers with dropout
   :param last_layer: the current last layer of the net
   :param dropout: rate of the dropout
   :return: 3 layers connected
   """
   y = -3
   x = Dense(512, activation='relu', name='fc-1')(last_layer)
   if (dropout>0):
       x = Dropout(dropout)(x)
   x = Dense(256, activation='relu', name='fc-2')(x)
   if (dropout > 0):
       x = Dropout(dropout)(x)
   out = Dense(units=outputNeurons, activation='sigmoid', name='output_layer')(x)
   return out, y


def defineOptimizer(optimizer, lr, decay, moment):
   """
   function that prepare the optimizer
   :param optimizer: string of the desired optimizer
   :param lr: learning rate of the net
   :param decay: weight decay
   :return: the preapred optimizer for the model
   """
   if optimizer == 'adam':
       optim = Adam(learning_rate=lr, decay=decay)
   elif optimizer == 'SGD':
       optim = SGD(learning_rate=lr, decay=decay, momentum=moment)
   else:
       optim = RMSprop(learning_rate=lr, decay=decay)
   return optim


def createModel(s, optimizer, lr, outputNeurons=1, dropoutProp=0.0, layers=False, flat=False, max=False, decay=0.0, moment=0.9):
   """
   function that preapre the model for training
   :param s: size of the photo
   :param optimizer: string- which optimizer to use
   :param lr: learning rate of the net
   :param dropoutProp: rate for dropout configuration
   :param layers: boolean for adding 2 new layers
   :param flat: boolean that indicates if to flatten the 1 before last layer
   :param max: boolean that indicates if to use max pooling instead of average pooling
   :param decay: weight decay
   :return: model after compile
   """
   input = Input(shape=(s, s, 3))
   net = ResNet50V2(include_top=True, weights='imagenet', input_tensor=input, input_shape=None, pooling=None, classes=1000)
   if (flat == True | max == True):
       lastLayer = net.get_layer('post_relu').output
       if (flat):
           lastLayer = Flatten()(lastLayer)
       else:
           lastLayer = GlobalMaxPooling2D()(lastLayer)
   else:
       lastLayer = net.get_layer('avg_pool').output
   if (layers == True):
       out, y = addLayers(lastLayer, dropoutProp, outputNeurons)
   else:
       y=-1
       out = Dense(units=outputNeurons, name='output_layer', activation='sigmoid')(lastLayer)
   model = Model(input, out)
   for layer in model.layers[:y]:
       layer.trainable=False
   optim = defineOptimizer(optimizer, lr, decay, moment)
   model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
   return model


def prediction(model, data, labels, threshold=0.5):
   """
   function that predict and calculate the accuracy
   :param model: the training model
   :param data: matrix of the test data
   :param labels: vector of the following labels
   :return: accuracy of the model, vector of predictions and confusion-matrix
   """
   print("Stage 3: Testing The Model- cross your fingers")
   thepredict = model.predict(data)
   predictions = np.where(thepredict <= threshold, 0, 1)
   acc = sum(predictions == labels) / len(predictions)
   CofustionMatrix= confusion_matrix(labels, predictions)
   return acc, thepredict, CofustionMatrix


def tuning(model, data, labels, batch, epochs, aug=False, val=0.1, num=3, s=224, threshold=0.5):
    """
    function for tuning parameters
    :param model: the NN
    :param data: matrix of the train set
    :param labels: labels of the train set
    :param batch: int- size of the batch
    :param epochs: number of epochs for training
    :param aug: boolean- used augmentation or not
    :param val: rate of the data that will used for validation set
    :param num: number of added photos to augmentation
    :param s: size of the photo
    :param threshold: threshold for classification
    :return:
    """
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size = 0.25, random_state=17)
    if aug:
        X_train, y_train = augmentation(X_train, y_train, num, s)
    early_stopping = EarlyStopping(patience=4, restore_best_weights=True)
    model.fit(x=X_train, y=y_train, batch_size=batch, epochs=epochs, validation_split=val, callbacks=[early_stopping])
    acc, _, _ = prediction(model, X_val, y_val, threshold)
    return acc


def ChoosePara(model, trainData, trainLabels, folds,  batch=32, epochs=10, threshold=0.5):
   """
   function for tuning parameters through cross-validation
   :param model: the model for training
   :param trainData: matrix of the train set
   :param trainLabels: vector of the lables
   :param folds: int - number of folds
   :param batch: size of the batch for optimization
   :param epochs: number of epochs for training
   :return: average accuracy value for specific parameters
   """
   kf = KFold(n_splits=folds, random_state=None, shuffle=False)
   thepredict = []
   for train_index, val_index in kf.split(trainData):
               X_train, X_test = trainData[train_index], trainData[val_index]
               y_train, y_test = trainLabels[train_index], trainLabels[val_index]
               model.fit(x=X_train, y=y_train, batch_size=batch, epochs=epochs)
               acc, _ , _  = prediction(model, X_test, y_test, threshold)
               thepredict.append(acc)
   return np.average(thepredict)


def augmentation(trainData, trainLabels, numOfAug=2, s=224):
   """
   function for creating augmentation on the data
   :param trainData: matrix of the train set
   :param trainLabels: vector of the labels of the data
   :param numOfAug: number of augmentation for each photo
   :param s: size of the photo (224)
   :return: matrix of the new train set with the augmentation and the following labels
   """
   datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                shear_range=0.1, zoom_range=0.2, horizontal_flip=True,
                                vertical_flip= True,fill_mode='nearest')
   augData = []
   for j in range(len(trainData)):
       img = trainData[j].reshape((1,) + trainData[j].shape)
       i = 0
       augData.append(trainData[j])
       augData.append(pca_color_augmentation_numpy1(trainData[j]))
       for batch in datagen.flow(img, batch_size=1):
           augData.append(batch[0])
           i += 1
           if i >= numOfAug:
               break  # otherwise the generator would loop infinitely
   augLabels = []
   for i in range(len(trainLabels)):
       tmp = np.repeat(trainLabels[i],(numOfAug+2))
       augLabels = np.append(augLabels,tmp)
   augData = np.reshape(augData, (len(trainData)*(numOfAug+2), s, s, 3))
   augLabels = np.reshape(augLabels, (len(trainLabels)*(numOfAug+2), 1))
   return augData, augLabels


def pca_color_augmentation_numpy1(image_array_input):
    """
    function that do pca color augmentation
    :param image_array_input: the original image
    :return: image after PCA color augmentation
    """
    assert image_array_input.ndim == 3 and image_array_input.shape[2] == 3
    assert image_array_input.dtype == np.float32

    img = image_array_input.reshape(-1, 3).astype(np.float32)
    scaling_factor = np.sqrt(3.0 / np.sum(np.var(img, axis=0)))
    img *= scaling_factor

    cov = np.cov(img, rowvar=False)
    U, S, V = np.linalg.svd(cov)

    rand = np.random.randn(3) * 0.1
    delta = np.dot(U, rand*S)
    delta = (delta * 1).astype(np.float64)[np.newaxis, np.newaxis, :]

    img_out = np.clip(image_array_input + delta, -1, 1).astype(np.float32)
    return img_out


def trainModel(model, data, labels, aug, num, s, val, batch=32, epochs=50):
   """
   function that train the model
   :param model: keras model after compile
   :param data: matrix of train set
   :param labels: vector of the labels
   :param aug: boolean variable that indicates if train set will include augmentation
   :param num: num of augmentations
   :param s: int of the size of the photo (224)
   :param batch: size of the batch for optimization
   :param epochs: number of epochs for training
   :return: the trained model and the history of the epochs
   """
   print("Stage 2: Training Model- Its Gonna Take Some Time")
   if aug:
       data, labels = augmentation(data, labels, num, s)
   early_stopping = EarlyStopping(patience=4, restore_best_weights=True)
   history = model.fit(x=data, y=labels, batch_size=batch, epochs=epochs, validation_split=val, callbacks=[early_stopping])
   return model, history


def plotPrecisionRecall(y_test, y_preds):
   """
   function that plot the precision-recall curve
   :param y_test: vector of real labels
   :param y_preds: vector of the predictions by the model
   """
   average_precision = average_precision_score(y_test, y_preds)
   precision, recall, _ = precision_recall_curve(y_test, y_preds)
   plt.figure(111)
   plt.step(recall, precision, color='b', alpha=0.2, where='post')
   plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
   plt.xlabel('Recall')
   plt.ylabel('Precision')
   plt.ylim([0.0, 1.05])
   plt.xlim([0.0, 1.0])
   plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
   plt.show()
   return


def find5Worst(y_test, y_preds):
   """
   function that find the 5 photos that were the worst classified
   :param y_test: vector of real labels
   :param y_preds: vector of the predictions by the model
   :return: two arrays of type 1 and type 2 worst errors
   """
   predictions = np.where(y_preds <= 0.5, 0, 1)
   Errors = pd.DataFrame({'Index':[], 'Error':[], 'Type': []})
   for i in range(0, len(y_test)):
       if (y_test[i] == 0) & (predictions[i] == 1):  # type 1 error
           Errors = Errors.append({'Index': i, 'Error': np.abs(y_preds[i]-0.5), 'Type': 1}, ignore_index=True)
       if (y_test[i] == 1) & (predictions[i] == 0):  # type 2 error
           Errors = Errors.append({'Index': i, 'Error': np.abs(y_preds[i]-0.5), 'Type': 2}, ignore_index=True)
   Errors.sort_values(by='Error', inplace=True, ascending=False)
   errors1 = Errors.loc[Errors['Type'] == 1].head(min(5, len(Errors.loc[Errors['Type'] == 1])))
   errors2 = Errors.loc[Errors['Type'] == 2].head(min(5, len(Errors.loc[Errors['Type'] == 2])))
   errors1.reset_index(drop=True, inplace=True)
   errors2.reset_index(drop=True, inplace=True)
   return errors1, errors2


def plotWorst(testLabels, preds, testData):
   """
   funtion that plot the 5 photos that were the worst classified
   :param testLabels: vector of real labels
   :param preds: vector of the predictions by the model
   :param testData: matrix of the images of the test set
   :return:
   """
   e1, e2 = find5Worst(testLabels, preds)
   fig1 = plt.figure(111)
   count = 0
   for i in e1['Index']:
       count+=1
       a = fig1.add_subplot(1, 5, count)
       img = testData[int(i)]
       plt.imshow(img)
       a.set_title(str(count)+': class_score ' + str(preds[int(i)]))
   fig1.set_size_inches(np.array(fig1.get_size_inches()) * 5)
   fig1.suptitle('Type 1 Errors')
   fig2 = plt.figure(112)
   count = 0
   for i in e2['Index']:
       count+=1
       a = fig2.add_subplot(1, 5, count)
       img = testData[int(i)]
       plt.imshow(img)
       a.set_title(str(count)+': class_score ' + str(preds[int(i)]))
   fig2.set_size_inches(np.array(fig2.get_size_inches()) * 5)
   fig2.suptitle('Type 2 Errors')
   return


def main():
    print('Stage 0: Importing Packages')
    param = getParameters()
    data, labels, dataOriginal = getData(param['data_path'], param['s'])
    trainData, testData, trainLabels, testLabels = splitData(data, labels, param['train_images_indices'],
                                                            param['test_images_indices'])
    model = createModel(s=param['s'], optimizer=param['optimizer'], outputNeurons=param['classes'],
                        lr=param['learnRate'], dropoutProp=param['dropRate'], layers=param['layers'],
                        flat=param['flat'], max=param['max'], decay=param['decay'])
    model, rrr = trainModel(model=model, data=trainData, labels=trainLabels, aug=param['aug'],
                            num=['numOfAug'], s=param['s'], val=param['valRate'],
                            batch=param['batch'], epochs=param['epochs'])
    acc, predicts, confusion = prediction(model=model, data=testData, labels=testLabels, threshold=param['threshold'])
    print("The accuracy rate is:" + str(acc))
    plot_confusion_matrix(conf_mat=confusion)
    plt.suptitle('Confusion-Matrix')
    plotPrecisionRecall(testLabels, predicts)
    return


if __name__ == "__main__":
    main()


