from keras.layers import Dense, Activation, Dropout, LSTM, Input, BatchNormalization, Embedding, Bidirectional, Masking, Lambda, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential, Model
from keras.losses import binary_crossentropy
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, CSVLogger
from read_data import *
import os


class lstm():
    def __init__(self, model_weight_path):
        self.data_dim = 1
        self.timesteps = 966
        self.lstmModel = None
        self.model_weight_path = model_weight_path
        self.inputShape = (self.timesteps,)
        self.lstmModel = self.getLSTM()

    def getLSTM(self):
        # expected input data shape: (batch_size, timesteps, data_dim)
        if self.lstmModel:
            return self.lstmModel
        self.lstmModel = Sequential()
        self.lstmModel.add(Masking(mask_value=0.0, input_shape=self.inputShape))
        self.lstmModel.add(Embedding(input_dim=66378, output_dim=500, input_length=966))
        self.lstmModel.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        self.lstmModel.add(Bidirectional(LSTM(12, dropout=0.2, recurrent_dropout=0.2)))
        self.lstmModel.add(Dense(1, activation='sigmoid'))
        rms = RMSprop()
        self.lstmModel.compile(loss=binary_crossentropy, optimizer=rms, metrics=["acc"])
        if os.path.exists(self.model_weight_path):
            self.lstmModel.load_weights(self.model_weight_path)
        return self.lstmModel

    def train(self, epoch=20, batch_size=32):
        x, y = get_encoded_data()
        saveBest = ModelCheckpoint(self.model_weight_path, save_best_only=True, monitor='val_loss')
        model = self.getLSTM()
        if os.path.exists(self.model_weight_path):
            model.load_weights(self.model_weight_path)
        model.summary()
        csv_logger = CSVLogger('log/training.csv')
        model.fit(x, y, batch_size=batch_size, validation_split=0.2, callbacks=[saveBest, csv_logger], epochs=epoch)


    def get_stream_accuarcy(self, batch_size=32):
        x1, y1 = get_encoded_data()
        x1 = x1.tolist()
        y1 = y1.tolist()
        x2, y2 = get_encoded_data("data/encoded_minor.csv")
        x2 = x2.tolist()
        y2 = y2.tolist()
        print("Drift occur at: %d" % (len(x1)))
        x1.extend(x2)
        y1.extend(y2)
        x = np.asarray(x1)
        y = y1
        predY = self.getLSTM().predict(x, batch_size=32)
        predY = predY.tolist()
        accuracy_list = []
        for i in range(int(len(y)/batch_size)):
            right_count = 0
            for j in range(batch_size):
                index = i * batch_size + j
                if y[index] == 1 and predY[index][0] >= 0.5:
                    right_count += 1
                elif y[index] == 0 and predY[index][0] < 0.5:
                    right_count += 1
            accuracy = float(right_count) / float(batch_size)
            accuracy_list.append(accuracy)
        accuracy_list = np.asarray(accuracy_list)
        np.save("data/stream_acc.npy", accuracy_list)
if __name__ == '__main__':

    s = lstm('model/model.h5')
    # s.train(20, 32)
    s.get_stream_accuarcy()