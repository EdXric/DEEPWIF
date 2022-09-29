# ---------For the inertial encoder-----------
# python3 ./DEEPWIF/wif_main.py -m trainset -M io
# python3 ./DEEPWIF/wif_main.py -m train -M io -se 1 -emd **
# python3 ./DEEPWIF/wif_main.py -m testset -M io 
# python3 ./DEEPWIF/wif_main.py -m test -M io --epochmodeldest ** --withmodelepoch 1

# ---------For the wireless positioning encoder-----------
# python3 ./DEEPWIF/bt_prep.py
# python3 ./DEEPWIF/wif_main.py -m train -M vlp
# python3 ./DEEPWIF/wif_main.py -m test -M vlp

# ---------For DEEPWIF-----------
# python3 ./RoV/wif_main.py -m trainset -M rov 
# python3 ./RoV/wif_main.py -m train -M rov -se 1 -emd **
# python3 ./RoV/wif_main.py -m testset -M rov
# python3 ./RoV/wif_main.py -m test -M rov --epochmodeldest ** --withmodelepoch 2