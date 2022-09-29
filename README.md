# DEEPWIF
**DEEPWIF** is a open source project initiated by ASIC Center laboratory, Southeast University, Nanjing, China.
DEEPWIF is an end-to-end trainable model for wireless-inertial fusion navigation appling several deep learning methods. The experiments shows our model can achieve remarkable results in complicated indoor positioning environments.
<br>
<br>
<center><img src="images/SmartFPS_1.png" title="NAME" height="40%" width="40%">
<img src="images/SmartFPS_2.png" title="NAME2" height="40.3%" width="40.3%"></center>
<center>Performing localization in the hallway</center><br>
<br>
<center><img src="images/SmartFPS_3.png" title="NAME" height="40%" width="40%"> 
<img src="images/SmartFPS_4.png" title="NAME" height="40%" width="40%"></center>
<center><img src="images/SmartFPS_6.png" title="NAME" height="40%" width="40%"> 
<img src="images/SmartFPS_7.png" title="NAME" height="40%" width="40%"></center>
<center>Experiment results of filter methods and DEEPWIF</center><br>
<br>

**Dataset**: (1-9 are training and test set for DEEPWIF, gan_1 and gan_2 are collected for transfer learning purpose)<br>

ar_acce: accelerometer in AR device<br>
        ----  timestamp (second), x, y, z<br>
ar_gravity: gravity in AR device<br>
        ----  timestamp (second), x, y, z<br>
ar_gyro: gyroscope in AR device<br>
        ----  timestamp (second), x, y, z<br>
ar_pose: pose estimation by AR device<br>
        ----  timestamp (second), x, y, z, qx, qy, qz, qw<br>
imu_acce: accelerometer in test device<br>
        ----  Android_Time_s_ (second), Unix_time_ms_ (millisecond), x, y, z<br>
imu_ble: bluetooth rssi received by test device <br>
        ----  Android_Time_s_ (second),E1,DC,E2,DD,E3,E5,EA,DB,DF,DE,E6,DA,EB,E9,E4,D9,EC,E7,E8,E0 (addresses for 20 beacons)<br>
imu_game_rv: bluetooth rssi received by test device<br>
        ----  Android_Time_s_ (second), Unix_time_ms_ (millisecond), x, y, z, w<br>
imu_gravity: gravity in test device<br>
        ----  Android_Time_s_ (second), Unix_time_ms_ (millisecond), x, y, z<br>
imu_gyro: gyroscope in test device<br>
        ----  Android_Time_s_ (second), Unix_time_ms_ (millisecond), x, y, z<br>
info: <br>
        ----  imu_delay (for time synchronization)<br>
              R_LI2LT (rotation matrix to translate local device coordinate system to AR phone coordinate system)<br><br><br>

**File Organization**:<br><br>
-- DEEPWIF<br>
    ---- bt_prep.py: preprocess bluetooth rssi data<br>
    ---- imu_en.py: inertial encoder model<br>
    ---- wl_en.py: wireless positioning model<br>
    ---- wif_deep_env.py: merge all environment infos for DEEPWIF<br>
    ---- wif_deep.py: DEEPWIF model<br>
    ---- wif_main.py: train and test models<br>
    ---- utils.py: utilities required by all files under the folder<br>
    ---- plot_estgt.py: Plot trajectory results<br>
    ---- run.json: defined parameters for datasets and models<br>
    ---- run.sh: for automation of all commands<br><br><br>

**Usage** (Choose Python or Shell):<br><br>
A. Python<br>
For the inertial encoder:
1) Training set: python3 ./DEEPWIF/wif_main.py -m trainset -M io
2) Train model: python3 ./DEEPWIF/wif_main.py -m train -M io -se 1 -emd **
3) Test set: python3 ./DEEPWIF/wif_main.py -m testset -M io 
4) Test model: python3 ./DEEPWIF/wif_main.py -m test -M io --epochmodeldest ** --withmodelepoch 1

For the wireless positioning encoder:
1) Preprocess bluetooth rssi dataset: python3 ./DEEPWIF/bt_prep.py
2) Train model: python3 ./DEEPWIF/wif_main.py -m train -M vlp
3) Test model: python3 ./DEEPWIF/wif_main.py -m test -M vlp

For DEEPWIF:
1) Training set: python3 ./RoV/wif_main.py -m trainset -M rov 
2) Train model: python3 ./RoV/wif_main.py -m train -M rov -se 1 -emd **
3) Test set: python3 ./RoV/wif_main.py -m testset -M rov
4) Test model: python3 ./RoV/wif_main.py -m test -M rov --epochmodeldest ** --withmodelepoch 2

B. Shell
1) ./DEEPWIF/run.sh<br><br><br>

**Required library version**: (We listed all environments here. Not all libs are required.)<br><br>
absl-py                 0.15.0<br>
astunparse              1.6.3<br>
backcall                0.2.0<br>
cached-property         1.5.2<br>
cachetools              5.0.0<br>
certifi                 2021.10.8<br>
charset-normalizer      2.0.12<br>
cycler                  0.11.0<br>
decorator               5.1.1<br>
flatbuffers             1.12<br>
fonttools               4.31.2<br>
gast                    0.3.3<br>
google-auth             2.6.2<br>
google-auth-oauthlib    0.4.6<br>
google-pasta            0.2.0<br>
grpcio                  1.32.0<br>
h5py                    2.10.0<br>
idna                    3.3<br>
importlib-metadata      4.11.3<br>
ipython                 7.33.0<br>
jedi                    0.18.1<br>
joblib                  1.1.0<br>
Keras-Preprocessing     1.1.2<br>
keras-tuner             1.1.2<br>
kiwisolver              1.4.0<br>
kt-legacy               1.0.4<br>
llvmlite                0.38.0<br>
Markdown                3.3.6<br>
matplotlib              3.5.1<br>
matplotlib-inline       0.1.3<br>
numba                   0.55.1<br>
numpy                   1.19.5<br>
numpy-quaternion        2022.2.10.14.20.39<br>
oauthlib                3.2.0<br>
opt-einsum              3.3.0<br>
packaging               21.3<br>
pandas                  1.1.5<br>
parso                   0.8.3<br>
pathlib                 1.0.1<br>
patsy                   0.5.2<br>
pexpect                 4.8.0<br>
pickleshare             0.7.5<br>
Pillow                  9.0.1<br>
pip                     22.0.4<br>
plyfile                 0.7.4<br>
prompt-toolkit          3.0.29<br>
protobuf                3.19.4<br>
ptyprocess              0.7.0<br>
pyasn1                  0.4.8<br>
pyasn1-modules          0.2.8<br>
Pygments                2.12.0<br>
pyparsing               3.0.7<br>
python-dateutil         2.8.2<br>
pytz                    2022.1<br>
requests                2.27.1<br>
requests-oauthlib       1.3.1<br>
rsa                     4.8<br>
scikit-learn            1.0.2<br>
scipy                   1.7.3<br>
seaborn                 0.11.2<br>
setuptools              60.10.0<br>
six                     1.15.0<br>
statsmodels             0.13.2<br>
tensorboard             2.8.0<br>
tensorboard-data-server 0.6.1<br>
tensorboard-plugin-wit  1.8.1<br>
tensorboardX            2.5<br>
tensorflow-addons       0.17.1<br>
tensorflow-estimator    2.4.0<br>
tensorflow-gpu          2.4.0<br>
termcolor               1.1.0<br>
threadpoolctl           3.1.0<br>
tqdm                    4.63.1<br>
traitlets               5.1.1<br>
typeguard               2.13.3<br>
typing_extensions       4.1.1<br>
urllib3                 1.26.9<br>
wcwidth                 0.2.5<br>
Werkzeug                2.0.3<br>
wheel                   0.37.1<br>
wrapt                   1.12.1<br>
zipp                    3.7.0<br>