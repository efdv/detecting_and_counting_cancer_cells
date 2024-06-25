from models import unet
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

def environment_model_graph(data1, name1, data2, name2, data3, name3, data4, name4, rootsave, nameSave = "graph", var_title="graph", xlabel="Axes X", ylabel="Axes Y"):
    num_epochs = len(data1[0])
    x = [i for i in range(num_epochs)]
    
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    for i,j,m,n in zip(data1[0], data2[0], data3[0], data4[0]):
        y1.append(i)
        y2.append(j)
        y3.append(m)
        y4.append(n)
            
    sns.set(style="darkgrid")
    
    fig, axes = plt.subplots(2,1,figsize=(10,8))
    
    axes[0].plot(x, y1, label=name1)
    axes[0].plot(x, y2, label=name2)
    axes[0].set_title(var_title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    
    axes[1].plot(x, y3, label=name3)
    axes[1].plot(x, y4, label=name4)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].legend()
    
    plt.tight_layout()
    
    name = nameSave+'.png'
    plt.savefig(rootsave+name, dpi=300)

def normResults(y_pred):
    yp = []
    for row in y_pred:
        valmax = max(row)
        for i in range(len(row)):
            if row[i] == valmax:
                row[i] = 1
            else:
                row[i] = 0
        
        yp.append(row)
    
    return yp

def normResults2D(y_pred, threshold):
    
    noimgs, u,v,channels = y_pred.shape
    for no in range(noimgs):
        for i in range(u):
            for j in range(v):
                for c in range(channels):
                    if y_pred[no,i,j,c] >= range:
                        y_pred[no,i,j,c] = 1
                    else:
                        y_pred[no,i,j,c] = 0

    
    return y_pred

def add_column_df(df, kfold,precision, recall, f1_list, weighted_avg):
    
    df['kfold'] = [kfold, kfold]
    df['class'] = [0,1]
    df['precision'] = precision
    df['recall'] = recall
    df['f1'] = f1_list
    df['weighted_avg'] = [weighted_avg, weighted_avg]
    
    return df



"""Paths and data"""
root = '../experimental_test_unet/'
root_graphs = root+'figures/'
#load data
y_train = np.load("../../../datasets/TCIA_SegPC_dataset/traindataset.npy")
y_test = np.load("../../../datasets/TCIA_SegPC_dataset/validationdataset.npy")
X_train = np.load("../../../datasets/TCIA_SegPC_dataset/trainlabels.npy")
X_test = np.load("../../../datasets/TCIA_SegPC_dataset/validationlabels.npy")

X_train = X_train/255.0
X_test = X_test/255.0
y_train = y_train/255
y_test = y_test/255
model = unet.unet_model()
no_model = 10
#X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

columns = ['kfold', 'phase', 'precision', 'recall', 'f1', 'weighted_avg']
df = pd.DataFrame(columns=columns)
df_parcial = pd.DataFrame(columns=columns)

#y_train = np.expand_dims(y_train, axis=-1)
#y_test = np.expand_dims(y_test, axis=-1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=500, batch_size=32, callbacks=[early_stopping])

name_model = 'model_unet_' + '_' +str(no_model)
model.save(root+'models/'+name_model+'.h5')

accuracy = []
loss = []
val_accuracy = []                                                      
val_loss = []
accuracy_model = []

accuracy.append(history.history['accuracy'])
loss.append(history.history['loss'])
val_accuracy.append(history.history['val_accuracy'])
val_loss.append(history.history['val_loss'])

name_graph = 'train/model_CNN_' + '_' + str(no_model) + '_' + 'val_and_loss'
environment_model_graph(accuracy, 
                        'Training Accuracy', 
                        val_accuracy, 
                        'Validation Accuracy', 
                        loss, 
                        'Training Loss', 
                        val_loss, 
                        'Validation Loss', 
                        root_graphs, 
                        name_graph, 
                        var_title="Training and Validation", 
                        xlabel="epochs", 
                        ylabel="")

accuracy_model.append(model.evaluate(X_test, y_test))
y_pred = model.predict(X_test)

ypred_cm = []
y_test_cm = []
for i in range(len(y_pred)):
    ypred_cm.append(np.argmax(y_pred[i]))
    y_test_cm.append(np.argmax(y_test[i]))

#yp = [] 
#yp = normResults2D(y_pred) #without numpy
threshold = 0.6
yp = np.where(y_pred >= threshold, 1, 0)
a = yp[0,:,:,1]
a = a*255
dip.showImgbyplt(a.astype('uint8'))

results = []
n_correct = sum(yp == y_test)
results.append(n_correct/len(y_pred))

rootcm = root_graphs+"cm/cm_"
nameimg = "Confusion matrix" 
class_names = ['cells', 'No_cells' ]

cm_display = ConfusionMatrixDisplay.from_predictions(y_test_cm,ypred_cm, cmap=plt.cm.Blues)
cm_display.ax_.set_xticklabels(class_names, fontsize=12)
cm_display.ax_.set_yticklabels(class_names, fontsize=12)
cm_display.figure_.savefig(rootcm + '_' + str(no_model) + '.png', dpi=300)

rootcm = root_graphs+"cm_norm/cm_"
class_names = ['cells', 'impurities' ]
cm_display = ConfusionMatrixDisplay.from_predictions(y_test_cm,ypred_cm, cmap=plt.cm.Blues, normalize="true", values_format = ".3f")
cm_display.ax_.set_xticklabels(class_names, fontsize=12)
cm_display.ax_.set_yticklabels(class_names, fontsize=12)

cm_display.figure_.savefig(rootcm  + '_' + str(no_model) + '.png', dpi=300)

yp2 = np.array(yp)
precision = []
recall = []
f1_list = []
for i in range(2):
    precision.append(precision_score(y_test[:,i],yp2[:,i]))
    recall.append(recall_score(y_test[:,i],yp2[:,i]))
    f1 = f1_score(y_test[:,i],yp2[:,i])
    f1_list.append(f1)

weighted_avg = []
f1_list = np.array(f1_list)    
weight = np.array(list(sum(y_test == 1)))
weighted_sum = np.sum(f1_list * weight)
total_weight = np.sum(weight)
weighted_avg.append(weighted_sum / total_weight)


df_parcial = add_column_df(df_parcial,1, precision, recall, f1_list, weighted_sum / total_weight)
df = pd.concat([df, df_parcial])
root_df = root+"sheets/"
name_excel = "result_model_" + str(no_model) + ".xlsx"
df.to_excel(root_df + name_excel, index=False)
