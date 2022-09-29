from __future__ import absolute_import, division, print_function, unicode_literals

from DEEPWIF.utils import *
from DEEPWIF.imu_en import *
import random

class RoVLPEnv:
    def __init__(self, dataargs):
        print("[INFO] Initializing RoVLP Environment...")
        # Data and save info
        self.train_files = [path for path in dataargs['data_dir'].glob(dataargs['train_dir_globmatchpattern'])]    
        self.test_files = [path for path in dataargs['data_dir'].glob(dataargs['test_dir_globmatchpattern'])]
        self.testall = dataargs['testall'] 
        self.test_fileno = dataargs['test_fileno']
        self.savepath = dataargs['save_path']
        # Raw data details
        self.acce_col = dataargs['acce_col']
        self.gyro_col = dataargs['gyro_col']
        self.game_rv_col = dataargs['game_rv_col']
        self.vi_col = dataargs['vi_col']
        self.rss_col = dataargs['ble_col']
        # Fine data details
        self.acce_comp = dataargs['acce_component']
        self.gyro_comp = dataargs['gyro_component']
        self.game_rv_comp = dataargs['game_rv_component']
        self.gt_comp = ['x', 'z']
        self.gt_quat_comp = dataargs['gt_quat_component']
        self.rss_comp = dataargs['ble_component']     
        # Training factors
        self.dim = len(self.gt_comp)
        self.train_seqlen = dataargs['train_seqlen']
        self.stride = dataargs['stride']  
        self.n_steps = dataargs['n_steps']  
        self.batch_size = dataargs['batchsize']
        self.epoch = dataargs['rovlp_epoch']
        self.optiz = optimizerparser(dataargs['rovlp_optimizer'])
        self.shadow_vec = dataargs['shadowing_vector'] 
        self.train_sd_prob = dataargs['shadowed_probability'] 
        self.min_ned, self.max_ned = dataargs['min_ned'], dataargs['max_ned']  
        # Test factors
        self.test_start_at, self.test_end_at = dataargs['test_start_at'], dataargs['test_end_at']
        self.test_sd_prob = dataargs['test_shadow_probability']
        self.analyze_sd = dataargs['cdf_shadow']
        # Net IONet factors
        self.io_indim = len(dataargs['acce_component']) + len(dataargs['gyro_component'])
        self.io_outdim = len(dataargs['imu_prediction_labels'])  
        # Net RSS factors
        self.rss_indim = len(self.rss_comp)
        # Net fusion factors
        self.fu_indim = 2 + len(self.rss_comp) + self.dim + 1  
        self.fu_outdim = self.dim 
        # Save by epoch
        self.savebyepoch = dataargs['savebyepoch']
        self.epochmodeldest = dataargs['epochmodeldest']
        self.withmodelepoch = dataargs['withmodelepoch']  
        
    def model_trainset(self, sparse=False, stride=10, start_frame=20000, end_frame=-2000):
        print("[INFO] Setting training set for MODEL...")
        
        trainx_imu, trainx_rss, trainy_loc, trainy_dir = [], [], [], []
        tlen = self.train_seqlen*self.n_steps
        iii = 0
        for fp in self.train_files:
            if fp.match('imu_acce*'):
                print("Loading data: ", fp)               
                imu, rss, vi = datasetcompilecommon(self, fp, imuonly=False, align=False)
                print('before split: ',  len(imu))
                istest = False
                if fp in self.test_files:
                    istest = True
                    print('Used in testset ...')             
                if istest:
                    imu = imu[start_frame:-40000]
                    rss = rss[start_frame:-40000]
                    vi = vi[start_frame:-40000]
                else:
                    imu = imu[start_frame:end_frame]
                    rss = rss[start_frame:end_frame]
                    vi = vi[start_frame:end_frame]
                print('after split: ', len(imu))

                if sparse:
                    for i in range((len(imu)-1)//tlen):
                        iii += 1
                        print('traj count: ', iii)                                         
                        tx_imu, tx_rss, ty_loc, ty_dir = [], [], [], []
                        for j in range(self.train_seqlen):
                            begin, end = i*tlen+j*self.n_steps, i*tlen+(j+1)*self.n_steps
                            tx_imu.append(imu[begin+1:end+1])
                            tx_rss.append(rss[end-100+1:end+1])
                            ty_loc.append(vi[end])
                            ty_dir.append(vi[end]-vi[begin])
                        if iii == 1:
                            trainx_imu = tx_imu
                            trainx_rss = tx_rss
                            trainy_loc = ty_loc
                            trainy_dir = ty_dir
                        else:
                            trainx_imu.extend(tx_imu)
                            trainx_rss.extend(tx_rss)
                            trainy_loc.extend(ty_loc)
                            trainy_dir.extend(ty_dir)
                        
                else:
                    fst = 0
                    imulen = imu.shape[0]
                    
                    while (fst+tlen+self.n_steps < imulen):                       
                        tx_imu, tx_rss, ty_loc, ty_dir = [], [], [], []
                        print(iii)
                        iii += 1

                        randf = random.randrange(0,self.n_steps)

                        for j in range(self.train_seqlen):
                            begin, end = fst+(j)*self.n_steps+randf, fst+(j+1)*self.n_steps+randf
                            tx_imu.append(imu[begin+1:end+1])
                            tx_rss.append(rss[end-100+1:end+1])
                            ty_loc.append(vi[end])
                            ty_dir.append(vi[end]-vi[begin])
                        if iii == 1:
                            trainx_imu = tx_imu
                            trainx_rss = tx_rss
                            trainy_loc = ty_loc
                            trainy_dir = ty_dir
                        else:
                            trainx_imu.extend(tx_imu)
                            trainx_rss.extend(tx_rss)
                            trainy_loc.extend(ty_loc)
                            trainy_dir.extend(ty_dir)

                        fst += stride*self.n_steps

        print(np.stack(trainx_imu).shape)

        trainx_imu = np.stack(trainx_imu).reshape(iii,self.train_seqlen,self.n_steps,6,1)
        trainx_rss = np.stack(trainx_rss).reshape(iii,self.train_seqlen,100,20,1)
        trainy_loc = np.stack(trainy_loc).reshape(iii,self.train_seqlen,2)
        trainy_dir = np.stack(trainy_dir).reshape(iii,self.train_seqlen,2)
        print(trainx_imu.shape, trainx_rss.shape,
         trainy_loc.shape, trainy_dir.shape)
        # Normalize direction vectors
        trainy_dir_arr = trainy_dir
        trainy_dir_arr_norm = np.linalg.norm(trainy_dir_arr, axis=-1, keepdims=True)
        trainy_dir_arr = np.divide(trainy_dir_arr, trainy_dir_arr_norm)

        print("[INFO] Saving Training set to file 'trainx_imu.npy', 'trainx_rss.npy', 'trainy_loc.npy', 'trainy_dir.npy'...")
        save_numpys(self.savepath, ['trainx_imu', 'trainx_rss', 'trainy_loc', 'trainy_dir'], trainx_imu,
        trainx_rss, trainy_loc, trainy_dir)

    # For stateful
    def model_testset_batch(self, test_end_at=-2000, test_start_at=-40000):
        print("[INFO] Setting test set for MODEL...")
        
        self.test_start_at = test_start_at
        self.test_end_at = test_end_at
        seqlen = (self.test_end_at - self.test_start_at) // self.n_steps
        bw = int(self.n_steps/2)
        
        sfidx = 10
        for fp in self.test_files:
            if fp.match('imu_acce*'):
                testx_imu, testx_rss, testy_loc, testy_dir = [], [], [], []
                print("Loading data: ", fp)
                imu, rss, vi = datasetcompilecommon(self, fp, imuonly=False, align=False)
                print(imu.shape)

                tx_imu, tx_rss, ty_loc, ty_dir = [], [], [], []
                for i in range(seqlen):           
                    begin, end = self.test_start_at+i*self.n_steps, self.test_start_at+(i+1)*self.n_steps
                    tx_imu.append(imu[begin+1:end+1])
                    tx_rss.append(rss[end-100+1:end+1])
                    ty_loc.append(vi[end])
                    ty_dir.append(vi[end]-vi[begin])
                
                testx_imu = tx_imu
                testx_rss = tx_rss
                testy_loc = ty_loc
                testy_dir = ty_dir

                print(np.stack(testx_imu).shape)
                testx_imu = np.stack(testx_imu).reshape(seqlen,self.n_steps,6,1)
                testx_rss = np.stack(testx_rss).reshape(seqlen,100,20,1)
                testy_loc = np.stack(testy_loc).reshape(seqlen,2)
                testy_dir = np.stack(testy_dir).reshape(seqlen,2)
                print(testx_imu.shape, testx_rss.shape,
                testy_loc.shape, testy_dir.shape)

                # Normalize direction vectors
                testy_dir_arr = testy_dir
                testy_dir_arr_norm = np.linalg.norm(testy_dir_arr, axis=-1, keepdims=True)
                testy_dir_arr = np.divide(testy_dir_arr, testy_dir_arr_norm)

                print("[INFO] Saving Test set to file 'testx_imu.npy', 'testx_rss.npy', 'testy_loc.npy', 'testy_dir.npy'...")
                save_numpys(self.savepath, ['testx_imu'+str(sfidx), 'testx_rss'+str(sfidx), 'testy_loc'+str(sfidx), 'testy_dir'+str(sfidx)], 
                testx_imu, testx_rss, testy_loc, testy_dir)
                sfidx += 1
