import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-st', '--start', type=int, metavar='', required=False, help='start of sequence')
parser.add_argument('-ed', '--end', type=int, metavar='', required=False, help='end of sequence')
args = parser.parse_args()

rootpath = './DEEPWIF/***/'
spath = rootpath+'figs/'
Y_test = pd.read_csv(rootpath+'Y_test.csv')
Y_pred = pd.read_csv(rootpath+'Y_pred.csv')
print(Y_test.shape)
print(Y_pred.shape)

if '2' in Y_test.columns:  
    Y_test = np.asarray(Y_test[['0', '1', '2']][::1])
    Y_pred = np.asarray(Y_pred[['0', '1', '2']][::1])
else:  
    Y_test = np.asarray(Y_test[['0', '1']][::1])
    Y_pred = np.asarray(Y_pred[['0', '1']][::1])

# Calculate Average Positioning Error
def cal_avg_err(y_pred, y_test):
    avgerr = np.mean(np.linalg.norm(y_pred-y_test, axis=1, keepdims=True))
    return avgerr

picka, pickb = 0,500
if args.start:
    picka = args.start
if args.end:
    pickb = args.end
Y_test = Y_test[picka:pickb]
Y_pred = Y_pred[picka:pickb]
# print(Y_test)
# print(Y_pred)
print('Validation set RMSE: %.3f' % cal_avg_err(Y_pred, Y_test))
# print('Validation set RMSE: %.3f' % cal_avg_err(Y_pred[picka:pickb], Y_test[picka:pickb]))


fig = plt.figure()
ax = plt.axes()
ax.plot(Y_test[:, 0], Y_test[:, 1], label='Ground Truth', c='red')
ax.plot(Y_pred[:, 0], Y_pred[:, 1], label='Estimate', c='lime')
ax.plot(Y_test[0, 0], Y_test[0, 1], label='start', marker='^', c='b')
ax.plot(Y_pred[0, 0], Y_pred[0, 1], marker='^', c='b')
ax.plot(Y_test[-1, 0], Y_test[-1, 1], label='end', marker='*', c='b')
ax.plot(Y_pred[-1, 0], Y_pred[-1, 1], marker='*', c='b')
plt.xlabel("x(m)")
plt.ylabel("y(m)")
plt.legend()
# plt.show()
fig.savefig(spath+'res2d.svg')

