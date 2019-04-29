import numpy as np
from skmultiflow.drift_detection import DDM
from tqdm import tqdm

def ddm_test():
    ddm = DDM()
    true_occur_position = 4443
    data_stream = np.load("data/stream_acc.npy")
    for i in tqdm(range(data_stream.shape[0])):
        # print(data_stream[i])
        # print(i)
        ddm.add_element(data_stream[i])
        if ddm.detected_warning_zone():
            print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
        if ddm.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))

if __name__ == '__main__':
    ddm_test()
