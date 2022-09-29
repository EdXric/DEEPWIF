# -*- coding: utf-8 -*-
import pandas as pd
from os import path


def blepre(accfp, blefp, bleorifp, savepath):
    print('Compiling data: ', blefp)

    accsynfile = pd.read_csv(accfp, header=0, delimiter=',')
    
    blerawfile = pd.read_csv(blefp, skiprows=2, 
                          names=['timestamp', 'unixtime', 'address', 'rssi', 'priphy', 'secphy'], 
                          delimiter=' ')
    
    strs = ["E1", "DC", "E2", "DD", "E3", 
            "E5", "EA", "DB", "DF", "DE", 
            "E6", "DA", "EB", "E9", "E4", 
            "D9", "EC", "E7", "E8", "E0"]
    blestr = ["AC:23:3F:8A:38:" + s for s in strs]
    bleset = {}
    for i in range(len(blestr)):
        bleset[blestr[i]] = blerawfile[blerawfile['address']==blestr[i]]
        
    reftime = pd.to_timedelta(accsynfile['Android_Time_s_'], unit='s')

    # Test
    # blematfile = pd.read_csv(bleorifp, 
    #                          header=0, delimiter=',')
    # num = 1
    
    # bletest = bleset[blestr[num]].loc[:,['timestamp', 'rssi']]
    # bletest.rename(columns = {'rssi':strs[num]}, inplace = True)
    # bletest['timestamp'] = (bletest['timestamp']/1e9).round(3)
    # bletest['timestamp'] = pd.to_timedelta(bletest['timestamp'], unit='s')  
       
    # ans = bletest.set_index('timestamp').resample('0.001S').interpolate('linear')
    
    # finalans = ans.reindex(reftime).bfill().ffill();
    
    # blemattest = blematfile.loc[:,['Android_Time_s_', strs[num]]]
    # blemattest.rename(columns = {'Android_Time_s_':'timestamp', strs[num]:strs[num]+'_ori'}, 
    #                   inplace = True)
    # blemattest['timestamp'] = pd.to_timedelta(blemattest['timestamp'], unit='s')
    # blemattest = blemattest.set_index('timestamp')
    
    # finalans2 = finalans.resample('0.5S').mean().resample('0.005S').interpolate('linear')
    # finalans2.rename(columns = {strs[num]:strs[num]+'_2'}, 
    #                  inplace = True)
    
    # a = finalans.join(blemattest).join(finalans2)
    
    # a.plot()

    for i in range(len(blestr)):
        blefinal = bleset[blestr[i]].loc[:,['timestamp', 'rssi']]
        blefinal.rename(columns = {'rssi':strs[i]}, inplace = True)
        blefinal['timestamp'] = (blefinal['timestamp']/1e9).round(3)
        blefinal['timestamp'] = pd.to_timedelta(blefinal['timestamp'], unit='s')  
        blefinal = blefinal.set_index('timestamp').resample('0.001S').interpolate('linear')
        blefinal = blefinal.reindex(reftime).bfill().ffill()
        blefinal = blefinal.resample('0.2S').mean().resample('0.005S').interpolate('spline', order=3) # default 0.5S
        blefinal = blefinal.reindex(reftime).bfill().ffill()
        if i == 0:
            blesetfinal = blefinal
        else:
            blesetfinal = blesetfinal.join(blefinal)
    blesetfinal = blesetfinal.reset_index()
    t = blesetfinal['Android_Time_s_'].astype('timedelta64[ms]').astype(int)
    t = t/1e3
    blesetfinal['Android_Time_s_'] = t
    
    blesetfinal.to_csv(savepath, index=False)
    print('Compiling done.')


if __name__ == '__main__':
    
    rt = './RoV/data2'

    dd = 'Test20220314'
    imuds = ['test/20220514193640','test/20220514210232']
    sds = ['syn_rov/test/test/' + s for s in  ['1','2']]

    for i in range(len(imuds)):
        imud = imuds[i]
        sd = sds[i]
        accfp = path.join(rt,dd,sd,'syn/imu_acce.txt')
        blefp = path.join(rt,dd,imud,'ble.txt')
        bleorifp = path.join(rt,dd,sd,'syn/imu_ble.txt')
        savefp = path.join(rt,dd,sd,'syn/imu_ble2.txt')
        print(accfp)
        print(blefp)
        print(bleorifp)
        print(savefp)

    
    
    