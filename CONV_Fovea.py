from skimage import io
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import numpy as np
def preprocess(image_path):
  img_gray=io.imread(image_path,as_gray=True)
  #plt.imshow(img_gray)
  #excel_loc= 

  orig = img_gray.copy()
  (H, W) = img_gray.shape[:2]

  (newW, newH) = (700,700)
  rW = W / float(newW)
  rH = H / float(newH)

# resize the image
  img_gray = cv2.resize(img_gray, (newW, newH))
  (H, W) = img_gray.shape[:2]

  #plt.contour(img_gray, [0, 100])

  #plt.imshow(img_gray, cmap=plt.cm.gray, interpolation='bilinear') 

  im = ndimage.gaussian_filter(img_gray, 8)
  #plt.imshow(im)
  mask = (im > im.mean()).astype(np.float)
  mask += 0.1 * im
  img = mask + 0.2*np.random.randn(*mask.shape)

  hist, bin_edges = np.histogram(img, bins=60)
  bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

  binary_img = img > 0.5

  #plt.imshow(binary_img,cmap='gray')

# Remove small white regions
  open_img = ndimage.binary_opening(binary_img)
# Remove small black hole
  close_img = ndimage.binary_closing(open_img)

#close_img=np.array(close_img)
  close_img = np.array(close_img, dtype=np.uint8)

  plt.imshow(close_img,cmap='gray')

  #lx, ly = close_img.shape
  #X, Y = np.ogrid[0:lx, 0:ly]
  #mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly/4
  #close_img[mask] = 1
  #plt.imshow(close_img,cmap='gray')
  return(close_img)

img_array=np.zeros((813,32,32,3))
for i in range(1,201):
  img1=tf.keras.preprocessing.image.load_img('/content/drive/My Drive/Colab Notebooks/project prog/REFUGE-Validation400/REFUGE-Validation400/V'+str(i)+'.jpg',target_size=(32,32))
  img_array[i-1]=tf.keras.preprocessing.image.img_to_array(img1)
for i in range(201,401):
  img1=tf.keras.preprocessing.image.load_img('/content/drive/My Drive/Colab Notebooks/project prog/REFUGE-Validation400/REFUGE-Validation400/V0'+str(i)+'.jpg',target_size=(32,32))
  img_array[i-1]=tf.keras.preprocessing.image.img_to_array(img1)

  



import pandas as pd
data_loc=pd.read_excel('/content/drive/My Drive/Colab Notebooks/project prog/REFUGE-Validation400/Fovea_locations.xlsx')
data_loc['X']=data_loc['Fovea_X']
data_loc['Y']=data_loc['Fovea_Y']


from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
from keras.optimizers import Adam

model=Sequential()
model.add(Conv2D(64,kernel_size=(3,3),input_shape=(32,32,3),kernel_initializer='glorot_uniform'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D((2,2),padding='same'))
model.add(Conv2D(32,(3,3),kernel_initializer='he_normal'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D((2,2),padding='same'))
model.add(Conv2D(16,(3,3),kernel_initializer='he_uniform'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D((2,2),padding='same'))
model.add(Flatten())
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='linear'))

model.summary()
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.001),metrics=['accuracy'])

tar_x=np.asarray(data_loc['X'].values)
tar_y=np.asarray(data_loc['Y'].values)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(img_array,tar_x,test_size=0.1,random_state=2)

from keras.callbacks import ModelCheckpoint
path='/content/drive/My Drive/Colab Notebooks/project prog/REFUGE-Validation400/REFUGE-Validation400/temp'
calls=ModelCheckpoint(path,monitor='val_loss',mode='min',save_best_only=True)

model.fit(X_train,y_train,batch_size=32,epochs=100,callbacks=[calls],validation_split=0.1)

model.load_weights(path)
preds=model.predict(X_test)

error=[]
for i in range(len(preds)):
  error.append(abs(preds[i]-y_test[i]))

data_loc['X_pred']=preds


model=Sequential()
model.add(Conv2D(64,kernel_size=(3,3),input_shape=(32,32,3),kernel_initializer='glorot_uniform'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D((2,2),padding='same'))
model.add(Conv2D(32,(3,3),kernel_initializer='he_normal'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D((2,2),padding='same'))
model.add(Conv2D(16,(3,3),kernel_initializer='he_uniform'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D((2,2),padding='same'))
model.add(Flatten())
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='linear'))

model.summary()
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.001),metrics=['accuracy'])


from sklearn.model_selection import train_test_split
X_train,X_test,z_train,z_test=train_test_split(img_array,tar_y,test_size=0.1,random_state=2)

from keras.callbacks import ModelCheckpoint
path1='/content/drive/My Drive/Colab Notebooks/project prog/REFUGE-Validation400/REFUGE-Validation400/temp1'
calls=ModelCheckpoint(path1,monitor='val_loss',mode='min',save_best_only=True)

model.fit(X_train,z_train,batch_size=32,epochs=100,callbacks=[calls],validation_split=0.1)

model.load_weights(path1)
preds=model.predict(X_test)

error=[]
for i in range(len(preds)):
  error.append(abs(preds[i]-z_test[i]))

data_loc['Y_pred']=preds  
#Percentage Error in each set of predictions
per_x=[]
per_y=[]
for i in range(len(data_loc)):
  per_x.append((abs(data_loc['X'][i]-data_loc['X_pred'][i])/data_loc['X'][i])*100)
  per_y.append((abs(data_loc['Y'][i]-data_loc['Y_pred'][i])/data_loc['Y'][i])*100) 


data_loc['Per_x']=per_x
data_loc['Per_y']=per_y 

'''
#VGG16-Transfer Learning Model
from  keras.applications import VGG16
from keras.models import Model 

# load model without classifier layers
model = VGG16(include_top=False, input_shape=(32, 32, 3))
for layer in model.layers:
  layer.trainable = False
# add new classifier layers
flat1 = Flatten()(model.outputs)
class1 = Dense(8, activation='relu')(flat1)
output = Dense(1, activation='linear')(class1)
model = Model(inputs=model.inputs, outputs=output)

model.summary()
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.001),metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint
path='/content/drive/My Drive/Colab Notebooks/project prog/REFUGE-Validation400/REFUGE-Validation400/temp'
calls=ModelCheckpoint(path,monitor='val_loss',mode='min',save_best_only=True)

model.fit(img_array,tar_x,batch_size=1,epochs=100,callbacks=[calls],validation_split=0.1)
'''



