from keras.layers import Dense, Activation, Dropout, LSTM, Input, BatchNormalization, Embedding, Bidirectional, Masking, Lambda, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential, Model
from keras.losses import binary_crossentropy
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.callbacks import Callback
from keras.layers.merge import concatenate
import numpy.random as rng
import keras.backend as K
from read_data import *
import os
import time

class siameseLSTM():
    def __init__(self, model_weight_path):
        self.data_dim = 1
        self.timesteps = 966
        self.model = None
        self.lstmModel = None
        self.model_weight_path = model_weight_path
        self.inputShape = (self.timesteps,)
        self.lstmModel = self.getLSTM()
        self.model = self.getModel()

    def getLSTM(self):
        # expected input data shape: (batch_size, timesteps, data_dim)
        if self.lstmModel:
            return self.lstmModel
        self.lstmModel = Sequential()
        self.lstmModel.add(Masking(mask_value=0.0, input_shape=self.inputShape))
        self.lstmModel.add(Embedding(input_dim=66378, output_dim=500, input_length=966))
        self.lstmModel.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        self.lstmModel.add(Bidirectional(LSTM(12, dropout=0.2, recurrent_dropout=0.2)))
        # self.lstmModel.add(Conv1D(16, 3, strides=2))
        # self.lstmModel.add(BatchNormalization())
        # self.lstmModel.add(Activation('relu'))
        # #
        # self.lstmModel.add(Conv1D(32, 2, strides=2))
        # self.lstmModel.add(BatchNormalization())
        #
        # self.lstmModel.add(MaxPooling1D(2, strides=2))
        # self.lstmModel.add(Activation('relu'))
        #
        # self.lstmModel.add(Conv1D(64, 3, strides=2))
        # self.lstmModel.add(BatchNormalization())
        # self.lstmModel.add(Activation('relu'))
        #
        # self.lstmModel.add(GlobalMaxPooling1D())
        # self.lstmModel.add(Dropout(0.2))
        # self.lstmModel.summary()
        return self.lstmModel

    def getModel(self):
        if self.model:
            return self.model

        leftInput = Input(self.inputShape)
        rightInput = Input(self.inputShape)
        leftOutput = self.lstmModel(leftInput)
        rightOutput = self.lstmModel(rightInput)

        distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([rightOutput, leftOutput])
        self.model = Model(inputs=[leftInput, rightInput], outputs=distance)
        rms = RMSprop()
        self.model.compile(loss=self.contrasttive_loss, optimizer=rms)
        if os.path.exists(self.model_weight_path):
            self.model.load_weights(self.model_weight_path)
        return self.model

    def euclidean_distance(self,vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    def eucl_dist_output_shape(self,shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def contrasttive_loss(self, y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    def soft_margin_loss(self, y_true, y_pred):
        return K.mean((1 - y_true) * K.square(K.maximum(-10. * y_pred + 6.4, 0.24/(y_pred + 0.0001))) + y_true * K.square(K.maximum(10. * y_pred - 3.6, y_pred)))

    def train(self, epoch=30, batch_size=32):


        data_list, sentence_list = get_encoded_data("processed_data/encoded_major.csv", "raw_data/picked_data.csv")
        index_list = get_index_list("processed_data/train_index.csv")
        val_index_list = get_index_list("processed_data/test_index.csv")
        train_generator = generaterator(batch_size, data_list, index_list)
        val_generator = generaterator(batch_size, data_list, val_index_list)
        train_size = len(index_list)
        val_size = len(val_index_list)
        saveBest = ModelCheckpoint(self.model_weight_path, save_best_only=True, monitor='val_loss')
        model = self.getModel()
        if os.path.exists(self.model_weight_path):
            model.load_weights(self.model_weight_path)
        model.summary()
        csv_logger = CSVLogger('log/training.csv')
        model.fit_generator(generator=train_generator, validation_data=val_generator,
                            steps_per_epoch=int(train_size / batch_size),
                            validation_steps=int(val_size / batch_size), callbacks=[saveBest, csv_logger], epochs=epoch)
        lstm_model = self.getLSTM()
        lstm_model.save("model/lstm_model.h5")

    def evaluate_acc(self, batch_size, test_size=1000):
        data_list, sentence_list = get_encoded_data("processed_data/encoded_major.csv", "raw_data/picked_data.csv")
        val_index_list = get_index_list("processed_data/test_index.csv")
        val_generator = generaterator(batch_size, data_list, val_index_list)
        model = self.getModel()
        right_count = 0
        total_count = 0
        while(True):
            batch_data = val_generator.next()
            if len(batch_data) < 2 or total_count > test_size:
                break
            predict_data = model.predict_on_batch(batch_data[0])
            batch_len = batch_data[1].shape[0]
            total_count += batch_len
            for i in range(batch_len):
                if predict_data[i][0] > 0.5 and batch_data[1][i] == 0:
                    right_count += 1
                elif predict_data[i][0] <= 0.5 and batch_data[1][i] == 1:
                    right_count += 1

        print ("Accuracy on validation data: %.4f" % (float(right_count)/float(total_count)))

    def get_extractor(self):
        model = self.getModel()
        input_layer = model.layers[2].get_input_at(0)
        output_layer = model.layers[2].get_output_at(0)
        lstm_model = Model(inputs=input_layer, outputs=output_layer)
        lstm_model.compile(loss=binary_crossentropy, optimizer=RMSprop())
        lstm_model.summary()
        return lstm_model

    def presentation(self, n=10, length=24):
        data_list, sentence_list = get_encoded_data("processed_data/encoded_major.csv", "raw_data/picked_data.csv")
        val_index_list = get_index_list("processed_data/test_index.csv")

        extractor = self.get_extractor()
        index_list = random.sample(val_index_list, n)
        X1 = []
        X2 = []
        y = []
        sentences_pair = []

        for i in index_list:
            X1.append(data_list[i[0][0]])
            X2.append(data_list[i[0][1]])
            y.append(i[1])
            sentences_pair.append([sentence_list[i[0][0]], sentence_list[i[0][1]]])
        distance_list = []
        X1 = np.array(X1)
        X2 = np.array(X2)
        pred1 = extractor.predict_on_batch(X1)
        pred2 = extractor.predict_on_batch(X2)
        for i in range(n):
            distance = 0
            for j in range(length):
                distance += (pred1[i][j] - pred2[i][j]) ** 2.
            distance_list.append(distance)
        return X1, X2, pred1, pred2, distance_list, sentences_pair, y

    def encoded_2_features(self, input_file="processed_data/encoded_minor.csv", features_file="processed_data/features_minor.csv"):
        f = open(input_file, 'r')
        reader = csv.reader(f)
        index = []
        texts = []
        for row in reader:
            index.append(int(row[2]))
            texts.append(np.asarray(ast.literal_eval(row[0])))
        texts = np.asarray(texts)
        model = self.get_extractor()
        result = model.predict(texts)
        print result.shape
        features_list = []
        for i in range(result.shape[0]):
            features_list.append(result[i])
        f.close()
        f = open(features_file, 'w')
        writer = csv.writer(f)
        for i in range(len(features_list)):
            writer.writerow([index[i], features_list[i].tolist()])
        f.close()

if __name__ == '__main__':

    s = siameseLSTM('model/model.h5')
    # s.train(20, 32)
    # s.encoded_2_features("data/spam_minor.csv", "data/features_minor.csv")
    #
    s.evaluate_acc(32, 1000)
    # plot_logs()
    s.presentation()
    X1, X2, pred1, pred2, distance_list, sentences_pair, y = s.presentation()
    for i in range(10):
        print("=======================================================")
        print("Sentence 1")
        print(sentences_pair[i][0])
        print("-------------------------------------------------------")
        print("Sentence 2")
        print(sentences_pair[i][1])
        print("-------------------------------------------------------")
        print("distance: ", distance_list[i])
        print("=======================================================\n")
    # s.encoded_2_features()