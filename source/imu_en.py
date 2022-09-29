from DEEPWIF.utils import *
from scipy.ndimage import gaussian_filter1d
from IPython import display as ids

class custom_IONet:
    def __init__(self, dataargs, resolver):
        print("[INFO] Initializing IONet...")
        '''Hyper Parameters'''
        # Data directory and save directory
        self.train_files = [path for path in dataargs['data_dir'].glob(dataargs['trainio_dir_globmatchpattern'])]    
        self.test_files = [path for path in dataargs['data_dir'].glob(dataargs['testio_dir_globmatchpattern'])]
        self.testall = bool(dataargs['testall'])
        self.test_fileno = dataargs['test_fileno']
        self.savepath = dataargs['save_path']
        # Raw data details
        self.acce_col = dataargs['acce_col']
        self.gyro_col = dataargs['gyro_col']
        self.game_rv_col = dataargs['game_rv_col']
        self.ble_col = dataargs['ble_col']
        self.vi_col = dataargs['vi_col']

        # Fine data details
        self.acce_comp = dataargs['acce_component']
        self.gyro_comp = dataargs['gyro_component']
        self.game_rv_comp = dataargs['game_rv_component']
        self.gt_comp = ['x', 'z']
        self.gt_quat_comp = dataargs['gt_quat_component']
    
        # Training factors
        self.resolver = resolver
        self.stride = dataargs['stride']
        self.n_steps = dataargs['n_steps']
        self.epoch = dataargs['ionet_epoch']
        self.optiz = optimizerparser(dataargs['ionet_optimizer'])
        self.outlier_angle = dataargs['outlier_angle']
        self.modelnamesuffix = dataargs['modelnamesuffix']
        self.feat_sigma = dataargs['feat_sigma']
        self.targ_sigma = dataargs['targ_sigma']
        # Test factors
        self.test_sp, self.test_ep = dataargs['ionet_test_start_at'], dataargs['ionet_test_end_at']
        # Save by epoch
        self.savebyepoch = dataargs['savebyepoch']
        self.epochmodeldest = dataargs['epochmodeldest']
        self.withmodelepoch = dataargs['withmodelepoch']

       
    def model_trainset(self, start_frame=20000, end_frame=-40000):
        print("[INFO] Setting training set for IONet...")
        trainx, trainy = [], []


        for fp in self.train_files:
            if fp.match('imu_acce*'):
                print("Loading data: ", fp)
                imu, rss, vi = datasetcompilecommon(self, fp, imuonly=True, align=False, aligngrvonly=True)

                imu = imu[start_frame:end_frame]
                vi = vi[start_frame:end_frame]

                # Gaussian Smooth
                if self.feat_sigma>0:
                    imu = np.array([gaussian_filter1d(feat, sigma=self.feat_sigma, axis=0) for feat in imu])
                if self.targ_sigma>0:
                    vi = np.array([gaussian_filter1d(targ, sigma=self.targ_sigma, axis=0) for targ in vi])

                # Sampling window by stride
                siss = np.arange(self.n_steps, len(vi)-self.n_steps, self.stride)
                print(fp, fp.name, len(siss)) 
                num_after_correction = len(siss)          
                for sis in siss:
       
                    win_0, win_1, win_n = sis, sis + 1, sis + self.n_steps
                    trainx.append(imu[win_1:win_n+1, :])
                    
                    y_t, y_t_1, y_t_2 = vi[win_n], vi[win_0], vi[win_0-self.n_steps] 
                    trainy.append(calculate_dl_and_dpsi(y_t, y_t_1, y_t_2, dimension=2)) 
                    
                print('after correction: ', fp.name, num_after_correction, end='\n')
        trainx_arr = np.array(trainx, dtype=object)
        trainy_arr = np.array(trainy, dtype=object)
        print(trainx_arr.shape)

        print("[INFO] Saving Training set to file 'trainx_ionet.npy', 'trainy_ionet.npy'...")
        np.save(self.savepath + 'trainx_ionet.npy', trainx_arr)
        np.save(self.savepath + 'trainy_ionet.npy', trainy_arr)


    def model_testset(self):
        print("[INFO] Setting test set for IONet...")
        testx, testy = [], []
        gens = (fps for fps in self.test_files if fps.match('imu_acce*'))
        for gen in gens:
            fp = gen
        print('Loading data: ', fp)
        imu, rss, vi = datasetcompilecommon(self, fp, imuonly=True, align=False, aligngrvonly=True)
        if self.feat_sigma>0:
            imu = np.array([gaussian_filter1d(feat, sigma=self.feat_sigma, axis=0) for feat in imu])

        print(imu.shape)
        lenimu = len(imu)
        self.test_sp = lenimu-40000
        self.test_ep = lenimu-2000
        seqlen = (self.test_ep - self.test_sp) // self.n_steps
        
        pt_0, pt_0_1 = self.test_sp, self.test_sp-self.n_steps 
        y_0 = vi[pt_0]
        y_0_1 = vi[pt_0_1]
        test_pose0 = np.append(y_0, cal_psi(y_0, y_0_1, dimension=2))

        print('seqlen: ', seqlen)
        
        for i in range(1, seqlen+1):
            pt, pt_1, pt_2 = self.test_sp+i*self.n_steps, self.test_sp+(i-1)*self.n_steps, self.test_sp+(i-2)*self.n_steps 
            testx.append(imu[pt_1+1:pt+1, :]) 
            y_t, y_t_m, y_t_mm = vi[pt], vi[pt_1], vi[pt_2]
            testy.append(calculate_dl_and_dpsi(y_t, y_t_m, y_t_mm, dimension=2)) 

        print(np.array(testx).shape, np.array(testy).shape)
        testx_arr = np.array(testx)
        testy_arr = np.array(testy)

        print("[INFO] Saving Training set to file 'testx_ionet.npy', 'testy_ionet.npy'...")
        np.save(self.savepath + 'testx_ionet.npy', testx_arr)
        np.save(self.savepath + 'testy_ionet.npy', testy_arr)
        np.save(self.savepath + 'testy_ionet_pose0.npy', test_pose0)

    def build_network(self):
        model = K.models.Sequential()
        model.add(K.layers.Bidirectional(K.layers.LSTM(128, dropout=0.25, return_sequences=True), input_shape=(self.n_steps,6), name='bilstm1'))
        model.add(K.layers.Bidirectional(K.layers.LSTM(128, dropout=0.25), name='bilstm2'))   
        model.add(K.layers.Dense(2, name='output'))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=self.loss_func, metrics=['accuracy'])

        self.model = model

    def build_network_hp(self, hp):
        print('building....')
        model = K.models.Sequential()
        
        model.add(K.layers.Bidirectional(K.layers.LSTM(units=hp.Int(name='lstm1units',min_value=64,max_value=256,default=128,step=4), 
        dropout=hp.Float(name='lstm1dp',min_value=0.1,max_value=0.6,default=0.25,step=0.1), 
        return_sequences=True), input_shape=(self.n_steps,6), name='bilstm1'))

        model.add(K.layers.Bidirectional(K.layers.LSTM(units=hp.Int(name='lstm2units',min_value=64,max_value=256,default=128, step=4), 
        dropout=hp.Float(name='lstm2dp',min_value=0.1,max_value=0.6,default=0.25,step=0.1)), name='bilstm2'))   
        
        model.add(K.layers.Dense(2, name='output'))

        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float(
            name='learning_rate',
            min_value=1e-6,
            max_value=1e-2,
            default=0.0015,
            sampling='log')), loss='mse', metrics=['accuracy'])

        self.model = model
        return model

    def loss_func(self, y_true, y_pred, k=2, res=0):
        return tf.keras.losses.MSE(y_true[:,0], y_pred[:,0]) + k*tf.keras.losses.MSE(y_true[:,1], y_pred[:,1])

    def train_model(self):
        x = np.load(self.savepath + 'trainx_ionet.npy', allow_pickle=True).astype('float64')
        y = np.load(self.savepath + 'trainy_ionet.npy', allow_pickle=True).astype('float64')
        print('x mean: ', np.mean(np.mean(x, axis=1), axis=0))
        print('x max: ', np.max(np.max(x, axis=1), axis=0), 'x min: ', np.min(np.min(x, axis=1), axis=0))
        print('y mean: ', np.mean(y, axis=0))
        print('y max: ', np.max(y, axis=0), 'y min: ', np.min(y, axis=0))

        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        self.build_network()

        if self.savebyepoch>0:
            Path(self.savepath+self.epochmodeldest).mkdir(exist_ok=True)
            saver = CustomSaver(self.savepath+self.epochmodeldest, self.savebyepoch)
            history = self.model.fit(x, y, validation_split=0.05, validation_freq=1, callbacks=[saver], 
            epochs=self.epoch, batch_size=128, verbose=2)
        else:
            history = self.model.fit(x, y, validation_split=0.05, validation_freq=1, 
            epochs=self.epoch, batch_size=128, verbose=2)
       
        
        # save model
        print("[INFO] saving MODEL to file 'saved_model_ionet{}.h5'...".format(self.modelnamesuffix))
        self.model.save(self.savepath + 'saved_model_ionet{}.h5'.format(self.modelnamesuffix))

        # save history
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = self.savepath+'history.csv'
        if self.epochmodeldest:
            hist_csv_file = self.savepath+self.epochmodeldest+'history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

    def train_model_hp(self):
        x = np.load(self.savepath + 'trainx_ionet.npy', allow_pickle=True).astype('float64')
        y = np.load(self.savepath + 'trainy_ionet.npy', allow_pickle=True).astype('float64')
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        # print(self.build_network_hp(kt.HyperParameters()))
        build_model = partial(self.build_network_hp)
        tuner = kt.tuners.Hyperband(
            # self.build_network_hp,
            build_model,
            project_name='rovlp_hp',
            directory='./RoV/save2/tuner',
            objective='val_loss',
            allow_new_entries=True,
            tune_new_entries=True,
            hyperband_iterations=3,
            max_epochs=self.epoch)

        tuner.search_space_summary()
        
        start_search = 0
        if start_search:
            tuner.search(x=x, y=y, shuffle=True, validation_split=0.05, validation_freq=1,
                batch_size=128, epochs=10, 
                callbacks=[ 
                    ClearTrainingOutput(),
                    # callbacks.EarlyStopping(
                    #     monitor='val_loss', 
                    #     patience=2,
                    #     verbose=1,
                    #     mode='min',
                    #     restore_best_weights=True),
                    callbacks.TerminateOnNaN()
                ],
                verbose=2)
        
        tuner.results_summary(num_trials=1)
        best_params = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.model = tuner.hypermodel.build(best_params)
        endname = '_hp'
        print("[INFO] saving MODEL to file 'saved_model_ionet{}.h5'...".format(endname))
        self.model.save(self.savepath + 'saved_model_ionet{}.h5'.format(endname))

    def test_model(self):
        x = np.load(self.savepath + 'testx_ionet.npy') 
        y = np.load(self.savepath + 'testy_ionet.npy') 

        modelname = 'saved_model_ionet{}.h5'.format(self.modelnamesuffix)
        print(self.savepath+modelname)
        if self.withmodelepoch >= 0:
            modelname = self.epochmodeldest + "model_{}.h5".format(self.withmodelepoch)
        print('model name: {}'.format(modelname))
        model = K.models.load_model(self.savepath + modelname, custom_objects={'loss_func': self.loss_func})
        print("[INFO] Predict on test set...")
        model.evaluate(x, y, verbose=2)
        y_pred = model.predict(x, verbose=2)

        
        pd.DataFrame(y_pred).to_csv(self.savepath + 'Y_pred_dltheta.csv')
        pd.DataFrame(y).to_csv(self.savepath + 'Y_test_dltheta.csv')
        dl_acc = sqrt(mean_squared_error(y_pred[:,0], y[:,0]))
        dtheta_acc = sqrt(mean_squared_error(y_pred[:,1], y[:,1]))        
        print('dl_acc: %.4f  dtheta_acc: %.4f' % (dl_acc, dtheta_acc))
        
        dl_acmu = np.sum(y_pred[:,0] - y[:,0])
        dtheta_acmu = np.sum(y_pred[:,1] - y[:,1])
        print('accumulated dl: %.4f  accumulated dtheta: %.4f' % (dl_acmu, dtheta_acmu))
        
        Pose_0 = np.load(self.savepath + 'testy_ionet_pose0.npy') 
        n = len(y)+1
        print(y_pred.shape, y.shape)
        Y_pose_pred = np.zeros((n, 3))
        Y_pose_test = np.zeros((n, 3))
        Y_pose_pred[0] = Pose_0
        Y_pose_test[0] = Pose_0
        dim = 2
        # Method 1
        for i in range(n-1):
            if i == 0:
                Y_pose_pred[i+1] = cal_pose_from_polar_vec(Pose_0, y_pred[i], dimension=dim)
                Y_pose_test[i+1] = cal_pose_from_polar_vec(Pose_0, y[i], dimension=dim)
            else:
                Y_pose_pred[i+1] = cal_pose_from_polar_vec(Y_pose_pred[i], y_pred[i], dimension=dim)
                Y_pose_test[i+1] = cal_pose_from_polar_vec(Y_pose_test[i], y[i], dimension=dim)
        Y_xy_pred = Y_pose_pred[:, :2]
        Y_xy_test = Y_pose_test[:, :2]

        print('RMSE: ', cal_avg_err(Y_xy_pred, Y_xy_test))

        Y_xyz_pred = np.concatenate((Y_xy_pred, np.zeros((n, 1))), axis=1)
        Y_xyz_test = np.concatenate((Y_xy_test, np.zeros((n, 1))), axis=1)      
        
        pd.DataFrame(Y_xyz_pred).to_csv(self.savepath + 'Y_pred.csv')
        pd.DataFrame(Y_xyz_test).to_csv(self.savepath + 'Y_test.csv')

    def test_model_stride(self):
        x = np.load(self.savepath + 'testx_ionet.npy') 
        y = np.load(self.savepath + 'testy_ionet.npy') 

        model = K.models.load_model(self.savepath+'iomodel_epoch/saved_model_ionet_an_epoch5.hd5', custom_objects={'loss_func': self.loss_func})
        print("[INFO] Predict on test set...")
        model.evaluate(x, y, verbose=2)
        y_pred = model.predict(x, verbose=2)


        dl_acc = sqrt(mean_squared_error(y_pred[:,0], y[:,0]))
        dtheta_acc = sqrt(mean_squared_error(y_pred[:,1], y[:,1]))        
        print('Before average: dl_acc: %.4f  dtheta_acc: %.4f' % (dl_acc, dtheta_acc))
        
        avg_u = self.n_steps // self.stride
        y_pred_st_dl = np.mean(y_pred[:, 0].reshape(-1, avg_u), axis=1)
        y_pred_st_dtheta = np.mean(y_pred[:, 1].reshape(-1, avg_u), axis=1)
        y_test_st_dl = y[((self.n_steps-self.stride)//self.stride)::avg_u, 0]
        y_test_st_dtheta = y[((self.n_steps-self.stride)//self.stride)::avg_u, 1]

        y_pred = np.array([y_pred_st_dl, y_pred_st_dtheta]).T
        y = np.array([y_test_st_dl, y_test_st_dtheta]).T
        dl_acc = sqrt(mean_squared_error(y_pred[:,0], y[:,0]))
        dtheta_acc = sqrt(mean_squared_error(y_pred[:,1], y[:,1]))         
        print('After average: dl_acc: %.4f  dtheta_acc: %.4f' % (dl_acc, dtheta_acc))
        pd.DataFrame(y_pred).to_csv(self.savepath + 'Y_pred_dltheta.csv')
        pd.DataFrame(y).to_csv(self.savepath + 'Y_test_dltheta.csv')

        dl_acmu = np.sum(y_pred[:,0] - y[:,0])
        dtheta_acmu = np.sum(y_pred[:,1] - y[:,1])
        print('accumulated dl: %.4f  accumulated dtheta: %.4f' % (dl_acmu, dtheta_acmu))
        
        Pose_0 = np.load(self.savepath + 'testy_ionet_pose0.npy') # (3)
        n = len(y)+1
        print(y_pred.shape, y.shape)
        Y_pose_pred = np.zeros((n, 3))
        Y_pose_test = np.zeros((n, 3))
        Y_pose_pred[0] = Pose_0
        Y_pose_test[0] = Pose_0
        dim = 2
        for i in range(n-1):
            if i == 0:
                Y_pose_pred[i+1] = cal_pose_from_polar_vec(Pose_0, y_pred[i], dimension=dim)
                Y_pose_test[i+1] = cal_pose_from_polar_vec(Pose_0, y[i], dimension=dim)
            else:
                Y_pose_pred[i+1] = cal_pose_from_polar_vec(Y_pose_pred[i], y_pred[i], dimension=dim)
                Y_pose_test[i+1] = cal_pose_from_polar_vec(Y_pose_test[i], y[i], dimension=dim)
        Y_xy_pred = Y_pose_pred[:, :2]
        Y_xy_test = Y_pose_test[:, :2]
        print(Y_xy_pred.shape)
        print('RMSE: ', cal_avg_err(Y_xy_pred, Y_xy_test))

        Y_xy_pred = np.concatenate((Y_xy_pred, np.zeros((n, 1))), axis=1)
        Y_xy_test = np.concatenate((Y_xy_test, np.zeros((n, 1))), axis=1)      
        
        pd.DataFrame(Y_xy_pred).to_csv(self.savepath + 'Y_pred.csv')
        pd.DataFrame(Y_xy_test).to_csv(self.savepath + 'Y_test.csv')

class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self, destfolder, frequency=1):
        self.dest = destfolder
        self.frequency = frequency
        self.epoch = -1
    def on_epoch_end(self, epoch, logs={}):
        if self.epoch >= 0:
            epoch = self.epoch
        if epoch%self.frequency == 0:  
            self.model.save(self.dest+"model_{}.h5".format(epoch))

class ClearTrainingOutput(callbacks.Callback):
    def on_train_end(*args, **kwargs):
        ids.clear_output(wait=True)
