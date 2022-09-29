from __future__ import absolute_import, division, print_function, unicode_literals

# from bleimu_RoVLP_cnn import *
from DEEPWIF.wif_deep import *
from DEEPWIF.wl_en import *

import argparse
'''
Usage: python3 ./RoV/wif_main.py -m trainset -M io
python3 ./RoV/wif_main.py -m train -M io -se 1 -emd model1_byepoch/
python3 ./RoV/wif_main.py -m test -M io --epochmodeldest model1_byepoch/ --withmodelepoch 1 (-1 not use epoch model)

python3 ./RoV/wif_main.py -m test -M vlp

python3 ./RoV/plot_estgt.py
'''

class RoVLPParams:
    def __init__(self, argsdict, argsparsed):
        self.dataargs = {
            # Raw Data column details
            'acce_col': ['timestamp','unixtime','x','y','z'],
            'gyro_col': ['timestamp','unixtime','x','y','z'],
            'game_rv_col': ['timestamp','unixtime','x','y','z','w'],
            'ble_col': ['timestamp']+['b'+str(i) for i in range(1,21)],
            'vi_col': ['timestamp','x','y','z','qx','qy','qz','qw'],
            # Fine Data colunm details
            'acce_component': ['x', 'y', 'z'],
            'gyro_component': ['x', 'y', 'z'],
            'game_rv_component': ['x', 'y', 'z','w'],
            'gt_component': ['x', 'y', 'z'],
            'gt_quat_component': ['qx', 'qy', 'qz', 'qw'],
            'ble_component': ['b'+str(i) for i in range(1,21)],
            'gt_latentcomponent': ['dir_x', 'dir_y', 'dir_z'],
            'imu_prediction_labels': ['vx', 'vy'],
            # Data directory and save directory
            'data_dir': Path(argsdict['rootdir']+argsdict['data_dir']),
            'train_dir_globmatchpattern': argsdict['train_dir_glob'],
            'test_dir_globmatchpattern': argsdict['test_dir_glob'],
            'trainio_dir_globmatchpattern': argsdict['trainio_dir_glob'],
            'testio_dir_globmatchpattern': argsdict['testio_dir_glob'],
            'save_path': argsdict['rootdir']+argsdict['save_path'],
            # Training factors
            'train_seqlen': argsdict['train_seqlen'],
            'train_seqnum_eachdata': argsdict['train_seqnum_eachdata'],
            'shadowing_vector': argsdict['shadowing_vector'],
            'shadowed_probability': argsdict['shadowed_probability'],
            'stride': argsdict['stride'],  
            'n_steps': argsdict['n_steps'], 
            'min_ned': argsdict['min_ned'], 'max_ned': argsdict['max_ned'], 
            'rovlp_epoch': argsdict['rovlp_epoch'],
            'rovlp_optimizer': argsdict['rovlp_optimizer'], 
            # Test factors
            'testall': argsdict['testall'],
            'test_fileno': argsdict['test_fileno'],
            'test_start_at': argsdict['test_start_at'], 'test_end_at': argsdict['test_end_at'],
            'batchsize': argsdict['batchsize'],
            'test_shadow_probability': argsdict['test_shadow_probability'], 
            'cdf_shadow': argsdict['cdf_shadow'],
            # IONet factors
            'ionet_optimizer': argsdict['ionet_optimizer'],
            'ionet_epoch': argsdict['ionet_epoch'],
            'ionet_test_start_at': argsdict['ionet_test_start_at'], 'ionet_test_end_at': argsdict['ionet_test_end_at'],
            'outlier_angle': argsdict['outlier_angle'], 
            'feat_sigma': argsdict['feat_sigma'],
            'targ_sigma': argsdict['targ_sigma'],
            # Wireless Positioning
            'VLP_lr': argsdict['VLP_lr'],
            'VLP_epoch': argsdict['VLP_epoch'],
            # 
            'modelnamesuffix': args.modelnamesuffix,
            'savebyepoch': argsparsed.saveepoch,
            'epochmodeldest': argsparsed.epochmodeldest,
            'withmodelepoch': argsparsed.withmodelepoch
        }
        
    def print_hyper_ionet(self):
        print('-----IONET HYPERPARAMETERS-----')
        print('IMU: ', self.dataargs['imu_component'])
        print('Ground truth: ', self.dataargs['gt_component'])
        print('Labels: ', self.dataargs['imu_prediction_labels'])
        print('Stride: ', self.dataargs['stride'])
        print('Steps/window: ', self.dataargs['n_steps'])
        print('Learning rate: ', self.dataargs['ionet_lr'])
        print('Epoch: ', self.dataargs['ionet_epoch'])

        print('Test start: ', self.dataargs['test_start_at'], ', end: ', self.dataargs['test_end_at'])
        print('Test set: data5/', self.dataargs['test_fileno'])


if __name__ == '__main__':
        tf.get_logger().setLevel('ERROR')
        resolver = 1  
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--mode', type=str, metavar='', required=True, help='run mode (trainset/testset/train/test)')
        parser.add_argument('-M', '--model', type=str, metavar='', required=True, help='run model (io/rov/vlp)')
        parser.add_argument('-sh', '--showhyperparameter', type=bool, metavar='', required=False, default=False, help='show hyperparameters (True/False)')
        parser.add_argument('-se', '--saveepoch', type=int, metavar='', required=False, default=0, help='save model every * epochs (0 = not save)')
        parser.add_argument('-emd', '--epochmodeldest', type=str, metavar='', required=False, help='epoch model saved path')
        parser.add_argument('-wme', '--withmodelepoch', type=int, metavar='', required=False, help='test with model of epoch *')
        parser.add_argument('-mns', '--modelnamesuffix', type=str, metavar='', required=False, default="", help='name model with suffix')
        args = parser.parse_args()
        
        if args.saveepoch>0 and not args.epochmodeldest:
            parser.error("--saveepoch required --epochmodeldest")
        if args.withmodelepoch and not args.epochmodeldest:
            parser.error("--withmodelepoch required --epochmodeldest")

        with open("./RoV/run2.json",'r') as load_f:
            argsdict = json.load(load_f)

        '''Load Hyper Parameters'''
        params = RoVLPParams(argsdict, args)
        if args.showhyperparameter:
            params.print_hyper_ionet()

        if args.model == 'rov':
            RoVLP_sub = RoVLP(params.dataargs, resolver)
            if args.mode == 'trainset':
                RoVLP_sub.RE.model_trainset()
            elif args.mode == 'train':
                RoVLP_sub.train_model()
            elif args.mode == 'testset':
                RoVLP_sub.RE.model_testset_batch()
            elif args.mode == 'test':
                RoVLP_sub.test_model_batch()
            elif args.mode == 'build':
                RoVLP_sub.build_network(301)
            
        elif args.model == 'io':
            IONet_sub = custom_IONet(params.dataargs, resolver)
            if args.mode == 'trainset':
                IONet_sub.model_trainset()
            elif args.mode == 'train':
                IONet_sub.train_model()
            elif args.mode == 'train_hp':
                IONet_sub.train_model_hp()
            elif args.mode == 'testset':
                IONet_sub.model_testset()
            elif args.mode == 'test':
                IONet_sub.test_model()

        elif args.model == 'vlp':
            VLP_sub = VLP(params.dataargs)       
            if args.mode == 'train':
                VLP_sub.train()
            elif args.mode == 'test':
                VLP_sub.test()

