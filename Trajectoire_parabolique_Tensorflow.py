import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

############### Generate data base ###############

Ny = 95
Nx = 45
y= np.zeros((30000,2))
theta0 = np.zeros((30000,))
theta0_rel = np.zeros((30000,))
v0 = np.zeros((30000,))
v0_rel = np.zeros((30000,))
X_data = np.zeros((30000, 80))
def data():
    v = random.uniform(20,30)
    Traj = np.zeros(80,)
    z = random.uniform(30,60)
    g = 9.81
    theta =z/180.0*np.pi
    for x in range(0,80,1):
        y = -(g * x * x) / (2 * (v * np.cos(theta)) * (v * np.cos(theta))) + x * np.tan(theta)
        if y >=0:
            Traj[int(x)]= y
        if y<0:
            Traj[int(x)]= 0
    return v,z,Traj
for i in range(30000):
    v0[i],theta0[i], X_data[i] = data()
    v0_rel[i] = (v0[i]-19)/12
    theta0_rel[i] = ((theta0[i]/5)-6)/7
    y[i,0] = v0_rel[i]
    y[i,1] = theta0_rel[i]
X_scale = StandardScaler()
X = X_scale.fit_transform(X_data)
X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.30)
X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.30)

############### Model ###############

tf.keras.backend.set_floatx('float64')
model = keras.Sequential([
    keras.layers.Dense(units=80, activation='relu'),
    keras.layers.Dense(units=50, activation='relu'),
    keras.layers.Dense(units=2, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss=tf.losses.MeanSquaredError(),
              metrics=['mean_squared_error'])

training = model.fit(X_train, y_train, batch_size=128,epochs=20,verbose=1,validation_data=(X_val, y_val))

############### Evalutation ###############

test_eval = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

y_pred = model.predict(X_test)
a=0
for i in range(len(y_test)):
    a += (np.absolute((y_test[i,0]-y_pred[i,0])) + np.absolute((y_test[i,1]-y_pred[i,1])))/2
pourcent = (a/len(y_test))*100
print("{} %".format(pourcent))