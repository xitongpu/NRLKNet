import pickle
import numpy as np


def load_data(filename=r'datasets/RML2016.10b.dat'):
    Xd = pickle.load(open(filename,'rb'),encoding='iso-8859-1')  # Xd(120W,2,128) 10calss*20SNR*6000samples
    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ] # mods['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    X = []
    lbl = []
    train_idx = []
    # val_idx = []
    np.random.seed(2016)
    a = 0

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])     # ndarray(6000,2,128)
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
            train_idx += list(np.random.choice(range(a*6000,(a+1)*6000), size=4800, replace=False))
            # val_idx += list(np.random.choice(list(set(range(a*6000,(a+1)*6000)) - set(train_idx)), size=1200, replace=False))
            a+=1
    X = np.vstack(X)
    n_examples = X.shape[0]
    test_idx = list(set(range(0,n_examples)) - set(train_idx)) #- set(val_idx))
    
    X_train = X[train_idx]
    # X_val = X[val_idx]
    X_test =  X[test_idx]
    
    Y_train = np.array(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    # Y_val = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = np.array(list(map(lambda x: mods.index(lbl[x][0]),test_idx)))

    return (mods, snrs, lbl), (X_train,Y_train), (X_test,Y_test), (train_idx, test_idx)

if __name__ == '__main__':
    (mods, snrs, lbl), (X_train,Y_train), (X_test,Y_test), (train_idx,test_idx) = load_data()
