from __future__ import absolute_import, division, print_function, unicode_literals

from DEEPWIF.wif_deep_env import *
from DEEPWIF.utils import *
from DEEPWIF.imu_en import *
from bleimu_adver_ionet import define_encoder, define_regressor, combine_regr

class RoVLP:
    def __init__(self, dataargs, resolver):
        print("[INFO] Initializing RoVLP...")
        '''Hyper Parameters'''
        self.RE = RoVLPEnv(dataargs)
        self.savepath = dataargs['save_path']
        self.ionet = custom_IONet(dataargs, resolver)

    def rss_nml(self, rss):
        rss[:,:] -= -110
        rss[:,:] /= 10
        rss[rss<0.6] = 0
        return rss

    def build_network(self,batch_size):
        TRAINIMU = False
        TRAINRSS = True
        rssact = 'selu'
        # Inertial Encoder      
        imu_in = K.layers.Input(shape=(self.RE.n_steps, self.RE.io_indim))
        imu = K.layers.Bidirectional(K.layers.LSTM(128, return_sequences=True, dropout=0.25, trainable=TRAINIMU), name='bilstm1')(imu_in)
        imu = K.layers.Bidirectional(K.layers.LSTM(128, dropout=0.25, trainable=TRAINIMU), name='bilstm2')(imu)

        # Wireless Positioning Encoder
        rss_in = K.layers.Input(shape=(100,20,1))
        rss = K.layers.Conv2D(filters=16, kernel_size=7, strides=(2,2), activation=rssact, trainable=TRAINRSS, name='cnn_conv1', padding='same')(rss_in)
        rss = K.layers.Conv2D(filters=32, kernel_size=5, strides=(2,2), activation=rssact, trainable=TRAINRSS, name='cnn_conv2', padding='same')(rss)
        rss = K.layers.Conv2D(filters=64, kernel_size=3, strides=(5,1), activation=rssact, trainable=TRAINRSS, name='cnn_conv3', padding='same')(rss)
        rss = K.layers.MaxPooling2D(pool_size=2, trainable=TRAINRSS, name='cnn_mp2d')(rss)    
        rss = K.layers.GlobalAveragePooling2D(trainable=TRAINRSS, name='cnn_gap2d')(rss)

        # Asymmetric attention
        c0 = K.layers.Dense(rss.shape[-1], activation='selu', name='att_dense1_2')(rss)
        c1 = rss * c0
        c2 = K.layers.Concatenate()([imu, c1])
        pr = K.backend.reshape(c2, (batch_size, 1, imu.shape[-1]+rss.shape[-1])) 

        # Fusion
        pr = K.layers.Bidirectional(K.layers.LSTM(imu.shape[-1]+rss.shape[-1], batch_size = batch_size, 
        stateful=True, name='fu_bilstm1', dropout=0.2))(pr)
        pr = K.layers.Dropout(0.2)(pr)
        o1 = K.layers.Dense(self.RE.fu_outdim, name='fu_o1_1')(pr)  # Position x, y ,z
        o2 = K.layers.Dense(self.RE.fu_outdim, name='fu_o1_2')(pr)  # Direction vector (normalized) d, e, f
        out = K.layers.Concatenate()([o1, o2])

        model = K.models.Model(inputs=[imu_in, rss_in], outputs=out)

        model.compile(optimizer=self.RE.optiz, loss='mse', metrics=['mse'])

        return model

    def train_model(self, nml=False):
        print("[INFO] Training MODEL...")
        imu = np.load(self.savepath+'trainx_imu.npy')
        rss = np.load(self.savepath+'trainx_rss.npy')
        if nml:
            rss = self.rss_nml(rss)
        y_loc = np.load(self.savepath+'trainy_loc.npy')
        y_dir = np.load(self.savepath+'trainy_dir.npy')     

        batchsize = imu.shape[0]

        print('loaded trainset shape: ', imu.shape, rss.shape, y_loc.shape, y_dir.shape)
        imu = np.moveaxis(imu, 0, 1)
        rss = np.moveaxis(rss, 0, 1)
        y_loc = np.moveaxis(y_loc, 0, 1)
        y_dir = np.moveaxis(y_dir, 0, 1)
        print('trainset swapaxis: ', imu.shape, rss.shape, y_loc.shape, y_dir.shape)
        imu = imu.reshape(-1, imu.shape[2], imu.shape[3])
        rss = rss.reshape(-1, rss.shape[2], rss.shape[3], 1)
        y_loc = y_loc.reshape(-1, y_loc.shape[2])
        y_dir = y_dir.reshape(-1, y_dir.shape[2])
        print('trainset flattened: ', imu.shape, rss.shape, y_loc.shape, y_dir.shape)
        
        x = [imu, rss]
        y = np.append(y_loc, y_dir, axis=-1)

        model = self.build_network(batchsize)
        print(model.summary())
        
        model_ionet = K.models.load_model(self.savepath+'model13_byepoch/model_ionet_287.h5', custom_objects={'loss_func': self.ionet.loss_func})
        weights_ionet = []
        for layer in model_ionet.layers:
            if layer.name.startswith("bilstm"):
                weights_ionet.append(layer.get_weights())
            if layer.name=="output":
                weights_ionet.append(layer.get_weights())
        k = 0
        for layer in model.layers:
            if layer.name.startswith("bilstm"):
                layer.set_weights(weights_ionet[k])
                k += 1
            if layer.name=="output":
                layer.set_weights(weights_ionet[k])
                k += 1

        model_vlp = K.models.load_model(self.savepath+'/modelvlp/model_25.h5')       
        weights_vlp = []
        for layer in model_vlp.layers:
            if layer.name.startswith("cnn"):
                weights_vlp.append(layer.get_weights())
            if layer.name.startswith("densev"):
                weights_vlp.append(layer.get_weights())
            if layer.name.startswith("outputv"):
                weights_vlp.append(layer.get_weights())
        k = 0
        for layer in model.layers:
            if layer.name.startswith("cnn"):
                layer.set_weights(weights_vlp[k])
                k += 1
            if layer.name.startswith("densev"):
                layer.set_weights(weights_vlp[k])
                k += 1
            if layer.name.startswith("outputv"):
                layer.set_weights(weights_vlp[k])
                k += 1
                
        epoch = self.RE.epoch

        # Load validation set
        imu_v_all,rss_v_all,y_loc_v_all,y_dir_v_all = [],[],[],[]
        for i in range(7):
            imu_v = np.load(self.savepath+'testx_imu{}.npy'.format(i+1))
            rss_v = np.load(self.savepath+'testx_rss{}.npy'.format(i+1))
            if nml:
                rss_v = self.rss_nml(rss_v)
            y_loc_v = np.load(self.savepath+'testy_loc{}.npy'.format(i+1))
            y_dir_v = np.load(self.savepath+'testy_dir{}.npy'.format(i+1))
            print('loaded testset shape: ', imu_v.shape, rss_v.shape, y_loc_v.shape, y_dir_v.shape)
            imu_v_all.append(imu_v)
            rss_v_all.append(rss_v)
            y_loc_v_all.append(y_loc_v)
            y_dir_v_all.append(y_dir_v)
        

        saver = CustomSaver(self.savepath+self.RE.epochmodeldest, frequency=1)

        for i in range(epoch):
            print('epoch: ', i+1, '/', epoch)

            saver.epoch = i+1
            model.fit(x , y, epochs=1, shuffle=False, batch_size=batchsize, verbose=2, callbacks=[saver]) #, callbacks=[print_on_end()]
       
            model.reset_states()
            # model.save(self.savepath+'saved_model_temp.h5')
            if (i+1)%1==0:
                # Validation
                # model_train = K.models.load_model(self.savepath + 'saved_model_temp.h5')
                for j in range(7):
                    imu_v,rss_v,y_loc_v = imu_v_all[j],rss_v_all[j],y_loc_v_all[j]
                    model_v = self.build_network(1)
                    model_v.set_weights(model.get_weights())
                    yp_v = model_v.predict([imu_v, rss_v], batch_size=1)
                    y_pred_v = yp_v[:, :self.RE.fu_outdim]

                    if j == 3 or j == 6:
                        endstr = '\n'
                    else:
                        endstr = ' '
                    print('Val RMSE: %.3f' % cal_avg_err(y_pred_v[20:], y_loc_v[20:]),end=endstr)

        # save model
        print("[INFO] saving MODEL to file 'saved_model_bleimu.h5'...")
        model.save(self.savepath+'saved_model_bleimu.h5')

    def test_model_batch(self):
        ADVER = True
        print("[INFO] Predicting on test set...")
        fpidx = '3'
        imu = np.load(self.savepath+'testx_imu{}.npy'.format(fpidx))
        rss = np.load(self.savepath+'testx_rss{}.npy'.format(fpidx))
        y_loc = np.load(self.savepath+'testy_loc{}.npy'.format(fpidx))
        y_dir = np.load(self.savepath+'testy_dir{}.npy'.format(fpidx))

        print(imu.shape, rss.shape, y_loc.shape, y_dir.shape)
        modelname = self.savepath + 'saved_model_temp.h5'
        
        if self.RE.withmodelepoch >= 0:
            modelname = self.RE.epochmodeldest + "model_{}.h5".format(self.RE.withmodelepoch)
        print('model name: {}'.format(modelname))
        model_train = K.models.load_model(self.savepath + modelname)
        
        model = self.build_network(1)
        model.set_weights(model_train.get_weights())
        # print(model.summary())

        if ADVER:
            model_ionet = combine_regr(define_encoder(), define_regressor())
            model_ionet.load_weights('./RoV/save2/ionetadver_8/saved_model_ionet9.h5')           
            weights_vlp = []
            for layer in model_ionet.layers:              
                if layer.name == 'model_6':           
                    weights_vlp.append(layer.get_weights())
            k = 0
            for layer in model.layers:
                if layer.name == 'bilstm1':
                    layer.set_weights(weights_vlp[k])
                    k += 1

            # model_vlp = K.models.load_model(self.savepath+'bleadver_8/saved_model4.h5')    
            # print(model_vlp.summary())
            # weights_vlp = []
            # for layer in model_vlp.layers:
            #     if layer.name.startswith("cnn_conv"):
            #         weights_vlp.append(layer.get_weights())
            # k = 0
            # for layer in model.layers:
            #     if layer.name.startswith("cnn_conv"):
            #         layer.set_weights(weights_vlp[k])
            #         k += 1

        yp = model.predict([imu, rss], batch_size=1)
        print(yp.shape)
        y_pred_arr = yp[:, :self.RE.fu_outdim]
        y_pred_dir_arr = yp[:, self.RE.fu_outdim:]

        print('Validation set RMSE: %.3f' % cal_avg_err(y_pred_arr[20:], y_loc[20:]))

        print("[INFO] saving results...")
        
        pd.DataFrame(y_pred_arr).to_csv(self.savepath + 'Y_pred_rov_{}.csv'.format(fpidx))
        pd.DataFrame(y_loc).to_csv(self.savepath + 'Y_test_rov_{}.csv'.format(fpidx))
        pd.DataFrame(y_pred_dir_arr).to_csv(self.savepath + 'Y_pred_dir_rov_{}.csv'.format(fpidx))
        pd.DataFrame(y_dir).to_csv(self.savepath + 'Y_test_dir_rov_{}.csv'.format(fpidx))
        



