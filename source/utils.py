from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os, datetime, pickle, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
from tensorflow.keras import backend
from tensorflow.keras import activations
from tensorflow.keras import constraints
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import quaternion
import argparse
from functools import partial
from tensorflow.keras import callbacks

from sklearn.metrics import mean_squared_error
from math import log10, sqrt
from scipy.interpolate import BSpline
from scipy import signal
from random import sample
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from tensorflow import keras as K
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize
from statsmodels.distributions.empirical_distribution import ECDF
from tensorflow.keras.callbacks import TensorBoard
# from google.colab import output

sns.set(color_codes=True)

class CustomTensorBoard(TensorBoard):
    def __init__(self,
               log_dir='logs',
               histogram_freq=0,
               write_graph=True,
               write_images=False,
               update_freq='epoch',
               profile_batch=2,
               embeddings_freq=0,
               embeddings_metadata=None,
               **kwargs):
        super().__init__(log_dir, histogram_freq, write_graph, write_images, update_freq, 
        profile_batch, embeddings_freq, embeddings_metadata)

        self.g_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        self._log_epoch_metrics(self.g_epoch, logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(self.g_epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(self.g_epoch)

def rmse(a, b):
    return np.mean(np.linalg.norm(a-b, axis=1, keepdims=True))

def optimizerparser(argstr):
    '''
    Keras Optimizers: (https://faroit.com/keras-docs/0.2.0/optimizers/)
    keras.optimizers.SGD(lr=0.01, momentum=0., decay=0., nesterov=False)
    keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)
    keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
    keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
    keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    '''
    args = dict(subString.split("=") for subString in argstr.split(" "))
    if args['opt'].lower() == 'sgd':
        arglr = float(args['lr']) if 'lr' in args else 0.01
        argmom = float(args['momentum']) if 'momentum' in args else 0.
        argdec = float(args['decay']) if 'decay' in args else 0.
        argnes = bool(args['nesterov']) if 'nesterov' in args else False
        return tf.keras.optimizers.SGD(lr=arglr, momentum=argmom, decay=argdec, nesterov=argnes)
    elif args['opt'].lower() == 'adagrad':
        arglr = float(args['lr']) if 'lr' in args else 0.01
        argep = float(args['epsilon']) if 'epsilon' in args else 1e-6
        return tf.keras.optimizers.Adagrad(lr=arglr, epsilon=argep)
    elif args['opt'].lower() == 'adadelta':
        arglr = float(args['lr']) if 'lr' in args else 1.0
        argrho = float(args['rho']) if 'rho' in args else 0.95
        argep = float(args['epsilon']) if 'epsilon' in args else 1e-6
        return tf.keras.optimizers.Adadelta(lr=arglr, rho=argrho, epsilon=argep)
    elif args['opt'].lower() == 'rmsprop':
        arglr = float(args['lr']) if 'lr' in args else 0.001
        argrho = float(args['rho']) if 'rho' in args else 0.9
        argep = float(args['epsilon']) if 'epsilon' in args else 1e-6
        return tf.keras.optimizers.RMSprop(lr=arglr, rho=argrho, epsilon=argep)
    elif args['opt'].lower() == 'adam':
        arglr = float(args['lr']) if 'lr' in args else 0.001
        argb1 = float(args['beta_1']) if 'beta_1' in args else 0.9
        argb2 = float(args['beta_2']) if 'beta_2' in args else 0.999
        argep = float(args['epsilon']) if 'epsilon' in args else 1e-8
        return tf.keras.optimizers.Adam(lr=arglr, beta_1=argb1, beta_2=argb2, epsilon=argep)
        

def datasetcompilecommon(self, fp, imuonly=False, align=True, aligngrvonly=False):
    grav = pd.read_csv(str(fp).replace('acce', 'gravity'), names=self.acce_col, skiprows=1)[self.acce_comp].values
    acce = pd.read_csv(fp, names=self.acce_col, skiprows=1)[self.acce_comp].values-grav 
    gyro = pd.read_csv(str(fp).replace('acce', 'gyro'), names=self.gyro_col, skiprows=1)[self.gyro_comp].values
    game_rv = pd.read_csv(str(fp).replace('acce', 'game_rv'), names=self.game_rv_col, skiprows=1)[self.game_rv_comp].values
    rss = 0
    
    if imuonly == False:      
        rss = pd.read_csv(str(fp).replace('acce', 'ble2'), names=self.rss_col, skiprows=1)[self.rss_comp].values        
        rss /= -50

    
    game_rv = quaternionorder(game_rv, 'wxyz')

    vi_all = pd.read_csv(str(fp).replace('imu_acce', 'ar_pose'), names=self.vi_col, skiprows=1)
    vi = vi_all[self.gt_comp].values   
    vi_quat = vi_all[self.gt_quat_comp].values
    
    vi_quat = quaternionorder(vi_quat, 'wxyz')
    vi[:,1] = -vi[:,1]


    if align:
        with open(str(fp).replace('imu_acce.txt', 'info.json'),'r') as load_f:
            R_LI2LT = json.load(load_f)['R_LI2LT']
            R_LI2LT = quaternion.quaternion(0, R_LI2LT[0], R_LI2LT[1], 0)
        
        R_xz2xy = quaternion.quaternion(0.707106781,-0.707106781,0,0)
 
        init_tango_ori = quaternion.quaternion(*vi_quat[0])
        ori_q = quaternion.from_float_array(game_rv)
        init_rotor = R_xz2xy*init_tango_ori * R_LI2LT * ori_q[0].conj()
        ori_q = init_rotor * ori_q
        
        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]
    if aligngrvonly:
        ori_q = quaternion.from_float_array(game_rv)
        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

    imu = np.concatenate([gyro, acce], axis=1)
    return imu, rss, vi



# Calculate Average Positioning Error
def cal_avg_err(y_pred, y_test):
    avgerr = np.mean(np.linalg.norm(y_pred-y_test, axis=1, keepdims=True))
    return avgerr



class Datadetail:
    def __init__(self):
        self.data_path = Path('./data')
        # IMU data details
        self.imu_col = ['Time', 'roll', 'pitch', 'yaw', 'gyro_x', 'gyro_y', 'gyro_z', 
        'grav_x', 'grav_y', 'grav_z', 'acc_x', 'acc_y', 'acc_z', 'mag_x', 'mag_y', 'mag_z']
        self.vi_col = ['Time', 'Header', 'x', 'y', 'z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']


# =============================================================================
# Coding Utils
# =============================================================================
def save_numpys(path, names, *vararrays):
    i = 0
    for var in vararrays:
        var = np.array(var)
        np.save(path+names[i], var)
        i += 1


# =============================================================================
# Model Training Utils
# =============================================================================



def plot_loss(history):
    if 'val_loss' in history.history:
        # summarize history for accuracy
        plot_xy('model accuracy', 'epoch', 'accuracy', ['train', 'val']
        , 'upper left', history.history['accuracy'], history.history['val_accuracy'])
        # summarize history for loss
        plot_xy('model loss', 'epoch', 'loss', ['train', 'val']
        , 'upper left', history.history['loss'], history.history['val_loss'])
    else:
        # summarize history for accuracy
        plot_xy('model accuracy', 'epoch', 'accuracy', ['train']
        , 'upper left', history.history['accuracy'])
        # summarize history for loss
        plot_xy('model loss', 'epoch', 'loss', ['train']
        , 'upper left', history.history['loss'])


def plot_xy(title, xlabel, ylabel, legend, legend_loc, *varins):
    plt.figure()
    for varin in varins:
        plt.plot(varin)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc=legend_loc)
    plt.show()


def build_regressor():
    regressor = K.Sequential()
    regressor.add(K.layers.Dense(units=44, activation="elu", input_dim=5))
    regressor.add(K.layers.Dense(units=60, activation="elu"))
    regressor.add(K.layers.Dense(units=60, activation="elu"))
    regressor.add(K.layers.Dense(units=3, activation="linear"))
    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=[K.metrics.RootMeanSquaredError()])
    return regressor


def data_loader(csv, datax, datay):
    data_set = pd.read_csv(csv)
    x, y = create_dataset(data_set[datax], data_set[datay])
    return x, y


def create_train_validation_set(x, y, test_size, batch=32):
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, shuffle=True)
    x_train, y_train = tf.constant(x_train, dtype=tf.float32), tf.constant(y_train, dtype=tf.float32)
    x_valid, y_valid = tf.constant(x_valid, dtype=tf.float32), tf.constant(y_valid, dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch)
    return train_dataset, val_dataset


def dl_dpsi_of_seq_by_frame(y_gt, ss, n_steps):

    dl = np.linalg.norm(y_gt[ss:ss+n_steps] - y_gt[ss-1:ss+n_steps-1], axis=1)

    dx = y_gt[ss:ss+n_steps, 0] - y_gt[ss-1:ss+n_steps-1, 0]
    dy = y_gt[ss:ss+n_steps, 1] - y_gt[ss-1:ss+n_steps-1, 1]
    dx_ = y_gt[ss-1:ss+n_steps-1, 0] - y_gt[ss-2:ss+n_steps-2, 0]
    dy_ = y_gt[ss-1:ss+n_steps-1, 1] - y_gt[ss-2:ss+n_steps-2, 1]
    dpsi = np.arctan2(dy, dx) - np.arctan2(dy_, dx_)

    dpsi[dpsi > np.pi] = -2 * np.pi + dpsi[dpsi > np.pi]
    dpsi[dpsi <= -np.pi] = 2 * np.pi + dpsi[dpsi <= -np.pi]

    dl_and_dpsi = np.array([dl, dpsi]).T

    return dl_and_dpsi


def calculate_dl_and_dpsi(xy_n, xy_n_1, xy_n_2, dimension=3):
    if dimension == 2:
        dl = np.sqrt(np.sum(np.power((xy_n - xy_n_1), 2)))
        dpsi = cal_psi(xy_n, xy_n_1, dimension=2) - cal_psi(xy_n_1, xy_n_2, dimension=2)
        '''Let ccw is +'''
        if dpsi > np.pi:
            dpsi = -2 * np.pi + dpsi
        elif dpsi <= -np.pi:
            dpsi = 2 * np.pi + dpsi
        dl_dpsi = [dl, dpsi]
    else:
        # TODO
        dl_dpsi = 0
    return dl_dpsi


def calculate_dl_and_dpsi_2(xy_n, xy_n_1, quat_n, quat_n_1, dimension=3):
    if dimension == 2:
        dl = np.sqrt(np.sum(np.power((xy_n - xy_n_1), 2)))
        psi_n = R.from_quat(quat_n).as_euler('zyx', degrees=False)
        psi_n_1 = R.from_quat(quat_n_1).as_euler('zyx', degrees=False)
        dpsi = psi_n[0]-psi_n_1[0]  
        '''Let ccw is +'''
        if dpsi > np.pi:
            dpsi = -2 * np.pi + dpsi
        elif dpsi <= -np.pi:
            dpsi = 2 * np.pi + dpsi
        dl_dpsi = [dl, dpsi]
    else:
        # TODO
        dl_dpsi = 0
    return dl_dpsi


def cal_psi(point_current, point_past, dimension=3):
    p1 = point_current
    p2 = point_past
    if dimension == 2:
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        psi = np.arctan2(dy, dx)
    else:
        # TODO
        psi = 0
    return psi


def cal_pose_from_polar_vec(pose_init, dl_dpsi, dimension=3):
    if dimension == 2:
        x = pose_init[0] + dl_dpsi[0] * np.cos(pose_init[2] + dl_dpsi[1])
        y = pose_init[1] + dl_dpsi[0] * np.sin(pose_init[2] + dl_dpsi[1])
        psi = pose_init[2] + dl_dpsi[1]
        pose = np.array([x, y, psi])
    else:
        # TODO
        pose = 0
    return pose


def create_dataset(xset, yset):
    x_temp = xset.values
    y_temp = yset.values
    x_set = x_temp[1:, :]
    y_set = y_temp[1:, :]
    return x_set, y_set


def create_dataset2(xset, yset):
    x_temp = xset.values
    y_temp = yset.values
    x_set = np.concatenate((x_temp[1:, :], y_temp[0:-1, :]), axis=1)
    y_set = y_temp[1:, :]
    return x_set, y_set


def cal_avg_err(y_pred, y_test):
    avgerr = np.mean(np.linalg.norm(y_pred-y_test, axis=1, keepdims=True))
    return avgerr



def draw_2d(arrx, arry, xlabel, ylabel, title):
    plt.plot(arrx, arry)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# =============================================================================
# Calculate deltaL and deltaPhi for consective data
# =============================================================================
def special_angle(v1,v2):
    x1=v1[0]
    y1=v1[1]
    #rotation matrix for pi/2
    R1=np.array([[0,-1],[1,0]])
    #rotation matrix for -pi/2
    R2=np.array([[0,1],[-1,0]])
    v2_u = v2_u=v2/np.linalg.norm(v2)
    if (x1==0 and y1==0):
        #2019.04.15 edit: replace function slope, calculate angle by np.arctan2
        return np.arctan2(v2_u[1],v2_u[0])
    else:
        v1_u=v1/np.linalg.norm(v1)
        if np.array_equal(np.dot(R1,v1_u.T),v2_u.T):
            return np.pi/2
        elif np.array_equal(np.dot(R2,v1_u.T),v2_u.T):
            return -1*np.pi/2
        else:
            x1,y1 = v1_u[0],v1_u[1]
            x2,y2 = v2_u[0],v2_u[1]
            return np.arctan2(x1*y2-y1*x2,x1*x2+y1*y2)
        

def cal_l_fai_IONet(cord,m):
    res=[]
    length=[]
    res.append(0)
    length.append(0)
    temp = cord[1]-cord[0]
    res.append(np.arctan2(temp[1],temp[0]))
    lastV = (0,0)
    for i in range(1,m-1):
        deltaL = np.sqrt(np.square(cord[i,0]-cord[i-1,0])+np.square(cord[i,1]-cord[i-1,1]))
        if deltaL>0.1:
            res.append(0)
            length.append(0)
            continue
        length.append(deltaL)
        v1 = cord[i,:]-cord[i-1,:]
        v2 = cord[i+1,:]-cord[i,:]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        x2=v2[0]
        y2=v2[1]
        x1=v1[0]
        y1=v1[1]
        if x1*x2+y1*y2==0:
            if n1==0 and n2!=0:
                #print(lastV,v2)
                x1,y1=lastV[0],lastV[1]
                if(x1*x2+y1*y2)==0:
                    a=special_angle(lastV,v2)
                else:
                    a= np.arctan2(x1*y2-y1*x2,x1*x2+y1*y2)
            elif n2==0 and n1!=0:
                a=0
                lastV = v1
            elif n1==0 and n2==0:
                a =0
            else:
                a = special_angle(v1,v2)
            res.append(a)
        else:    
            a= np.arctan2(x1*y2-y1*x2,x1*x2+y1*y2)
            res.append(a)      
    deltaL =  np.sqrt(np.square(cord[m-1,0]-cord[m-2,0])+np.square(cord[m-1,1]-cord[m-2,1]))
    length.append(deltaL)
    return res,length

def quaternionorder(nparr, order2):
    if nparr.ndim==1:
        nparr = nparr.reshape(1, len(nparr))
    tmp = np.copy(nparr)
    if order2 == 'wxyz':       
        nparr[:,0] = tmp[:,3]
        nparr[:,1:] = tmp[:,:3]
    elif order2 == 'xyzw':
        nparr[:,3] = tmp[:,0]
        nparr[:,:3] = tmp[:,1:]
    del tmp
    return nparr

if __name__ == '__main__':
    nparr = np.array([1,2,3,4])
    nparr = nparr.reshape(1,len(nparr))

    print(nparr.ndim)
    print(quaternionorder(nparr, 'wxyz'))