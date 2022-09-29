from DEEPWIF.wif_deep_env import *
from DEEPWIF.utils import *
from DEEPWIF.imu_en import *
from os import path

class VLP:
    def __init__(self, dataargs=0):
        print("[INFO] Initializing VLP...")
        self.savepath = dataargs['save_path']
        self.lr = dataargs['VLP_lr']
        self.epoch = dataargs['VLP_epoch']
        self.bw = 50
        self.RE = RoVLPEnv(dataargs)
    
    def build_network(self):
        act = 'selu'
        model = K.models.Sequential()
        model.add(K.layers.Conv2D(filters=16, kernel_size=7, strides=(2,2), activation=act, name='cnn_conv1', padding='same', input_shape=(2*self.bw,20,1)))
        model.add(K.layers.Conv2D(filters=32, kernel_size=5, strides=(2,2), activation=act, name='cnn_conv2', padding='same'))
        model.add(K.layers.Conv2D(filters=64, kernel_size=3, strides=(5,1), activation=act, name='cnn_conv3', padding='same'))
        model.add(K.layers.MaxPooling2D(pool_size=2, name='cnn_mp2d'))
        model.add(K.layers.GlobalAveragePooling2D(name='cnn_gap2d'))
        model.add(K.layers.Dense(32, activation=act, name='densev1'))
        model.add(K.layers.Dense(8, activation=act, name='densev2'))
        model.add(K.layers.Dense(2, name='outputv'))
        model.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss=['mse'], metrics=['accuracy'])
        self.model = model

    def rss_nml(self, rss):
        print('Rescaling RSS values ...')
        rss /= -50
        return rss

    def data_aug(self, x, y):
        x_rv = np.flip(x,axis=-2)
        return np.concatenate([x,x_rv]), np.concatenate([y,y])


    def train(self, nml=True):
        print("[INFO] Training MODEL...")
        # x = np.load(self.savepath+'trainx_rss.npy')
        # y = np.load(self.savepath+'trainy_loc.npy')

        # ts = ['1','2','3','4','7']
        # tts = ['8','9']
        ts = ['1','2','3','4','7','8','9']
        tts = ['1','2','3','4','7','8','9']
        iii = 0
        iiiv = 0
        x, y = [], []
        xv, yv = [], []
        for t in ts:          
            fp = path.join('./RoV/data2/Test20220314/syn_rov/',t,'syn/imu_ble2.txt')
            print(fp)
            rss = pd.read_csv(fp, names=['timestamp']+['b'+str(i) for i in range(1,21)], skiprows=1)[['b'+str(i) for i in range(1,21)]].values
            vi = pd.read_csv(str(fp).replace('imu_ble2', 'ar_pose'), names=['timestamp','x','y','z','qx','qy','qz','qw'], skiprows=1)[['x','z']].values
            vi[:,1] = -vi[:,1]
            print(rss.shape, vi.shape)
            if nml:
                rss = self.rss_nml(rss)
            grpx, grpy = [], []
            grpxv, grpyv = [], []
            if t not in tts:
                excseg = 20*self.bw
            else:
                excseg = 400*self.bw
            print('excseg: ', excseg)
            
            print('Seqlen: ', (len(rss)-excseg-4*self.bw)//5)
            for j in range(4*self.bw, len(rss)-excseg, 5):              
                grpx.append(rss[(j-2*self.bw+1):(j+1)])
                grpy.append(vi[j])          
            if iii == 0:
                x = grpx
                y = grpy
            else:
                x.extend(grpx)
                y.extend(grpy)
            iii += 1
            if t in tts:
                print('Used in testset ...')
                for j in range(len(rss)-excseg+2*self.bw, len(rss)-20*self.bw+2*self.bw, 5):
                    grpxv.append(rss[(j-2*self.bw+1):(j+1)])
                    grpyv.append(vi[j])
                if iiiv == 0:
                    xv = grpxv
                    yv = grpyv
                else:
                    xv.extend(grpxv)
                    yv.extend(grpyv)
                iiiv += 1
            
        
        x = np.stack(x).reshape(len(x),2*self.bw,20,1)
        y = np.stack(y)
        xv = np.stack(xv).reshape(len(xv),2*self.bw,20,1)
        yv = np.stack(yv)

        print(x.shape, y.shape)
        print(xv.shape, yv.shape)

        self.build_network()
        print(self.model.summary())       
        earlystop = K.callbacks.EarlyStopping(monitor='val_loss', patience=30)
        ckp_filepath = './RoV/save2/ckp_vlp'
        model_ckp = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_filepath, save_freq=30)
        saver = CustomSaver(self.savepath+self.RE.epochmodeldest, frequency=1)

        history = self.model.fit(x, y, validation_data=(xv, yv), validation_freq=1, 
        epochs=self.epoch, batch_size=128*1, verbose=2, callbacks=[saver, earlystop])
        # save history
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = self.savepath+'history_vlp.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
        
        # save model
        print("[INFO] saving MODEL to file 'saved_model_vlp.h5'...")
        self.model.save(self.savepath + 'saved_model_vlp.h5')

    def test(self, nml=True):
        print("[INFO] Predicting on test set...")

        ts = '9'
        tts = ['1']
        for ts in tts:

            fp = path.join('./RoV/data2/Test20220314/syn_rov/',ts,'syn/imu_ble2.txt')
            rss = pd.read_csv(fp, names=['timestamp']+['b'+str(i) for i in range(1,21)], skiprows=1)[['b'+str(i) for i in range(1,21)]].values
            vi = pd.read_csv(str(fp).replace('imu_ble2', 'ar_pose'), names=['timestamp','x','y','z','qx','qy','qz','qw'], skiprows=1)[['x','z']].values
            vi[:,1] = -vi[:,1]
            print(rss.shape, vi.shape)
            rss/=-50

            x = []
            y = []

            excseg = 400*self.bw

            for i in range(len(rss)-excseg+200, len(rss)-20*self.bw+200, 200):
                x.append(rss[(i-2*self.bw+1):(i+1)])
                y.append(vi[i])

            x = np.stack(x).reshape(len(x),2*self.bw,20,1)
            y = np.stack(y)
            print(x.shape, y.shape)

            model = K.models.load_model(self.savepath + 'modelvlp100step_revise_byepoch/model_25.h5')#saved_model_vlp.h5/modelvlp6_byepoch/model_7.h5 /modelvlp100step_byepoch/model_35.h5
            # print(model.summary())
            y_pred = model.predict(x)

            print('RMSE: ', cal_avg_err(y_pred, y))
        
        pd.DataFrame(y_pred).to_csv(self.savepath + 'Y_pred.csv')
        pd.DataFrame(y).to_csv(self.savepath + 'Y_test.csv')

