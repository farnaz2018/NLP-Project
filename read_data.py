import csv
from tqdm import tqdm
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import ast
import numpy as np
import random
from matplotlib import pyplot as plt
import pickle

def read_data_raw():
    catid_list = []
    subcatid_list = []
    value_list = []
    original_list = []
    cat_dict = {}
    sub_cat_dict_2 = {}
    sub_cat_dict_3 = {}
    f = open('ads_en_us.csv', 'r')
    reader = csv.reader(f)
    i = 0
    for line in reader:
        if i == 0:
            print line
            i += 1
        else:
            if line[-1] not in value_list:
                if line[1] not in cat_dict.keys():
                    cat_dict[line[1]] = 1
                else:
                    cat_dict[line[1]] += 1
                if line[1] == '2':
                    if line[3] not in sub_cat_dict_2.keys():
                        sub_cat_dict_2[line[3]] = 1
                    else:
                        sub_cat_dict_2[line[3]] += 1
                elif line[1] == '3':
                    if line[3] not in sub_cat_dict_3.keys():
                        sub_cat_dict_3[line[3]] = 1
                    else:
                        sub_cat_dict_3[line[3]] += 1
                catid_list.append(line[1])
                subcatid_list.append(line[3])
                value_list.append(line[-1])
            original_list.append(line[-1])

    print len(value_list)
    print cat_dict
    print sub_cat_dict_2
    print sub_cat_dict_3
    return catid_list, subcatid_list, value_list

def write_data():

    data = []
    label = []

    f = open('ads_en_us.csv', 'r')
    reader = csv.reader(f)
    wf = open('picked_data.csv', 'w')
    writer = csv.writer(wf)

    i = 0
    for line in reader:
        if i == 0:
            print line
            i += 1
        else:
            if line[1] == '2':
                if line[3] == '16':
                    data.append(line[-1])
                    label.append('c1_main')
                elif line[3] == '28':
                    data.append(line[-1])
                    label.append('c1_minor')
            elif line[1] == '3':
                if line[3] == '2':
                    data.append(line[-1])
                    label.append('c2_main')
                elif line[3] == '51':
                    data.append(line[-1])
                    label.append('c2_minor')
    f.close()

    for i in tqdm(range(len(data))):
        data_item = data[i].split('<')[0]
        stop_punctuation = ['\r', '\n', '\t']
        for item in stop_punctuation:
            while item in data_item:
                data_item = data_item.replace(item, '')
        row = [data_item, label[i]]
        writer.writerow(row)
    wf.close()

def read_data_from_processed_data():
    f = open('picked_data.csv', 'r')
    reader = csv.reader(f)
    for line in reader:
        print len(line)
        print line[1]
        print line[0]

def make_word_dict(word_dict, max_length, sentence):

    word_list = sentence.split(' ')
    if len(word_list) > max_length:
        max_length = len(word_list)
    for word in word_list:
        if word in word_dict.keys():
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    return word_dict, max_length

def major_minor_split_index(root_dir="raw_data", target_dir="processed_data"):
    f = open(os.path.join(root_dir, 'picked_data.csv'), 'r')
    reader = csv.reader(f)
    i = 0
    major_index = []
    minor_index = []
    sentence_list = []
    word_dict = {}
    max_length = 0
    labels = []
    for line in tqdm(reader):
        if line[1].endswith('main'):
            major_index.append(i)
        elif line[1].endswith('minor'):
            minor_index.append(i)
        sentence_list.append(line[0])
        labels.append(line[1].split('_')[0])
        i += 1
        word_dict, max_length = make_word_dict(word_dict, max_length, line[0])

    max_words = len(word_dict.keys())
    print max_words
    t = Tokenizer(num_words=max_words)
    t.fit_on_texts(sentence_list)
    encoded_docs = t.texts_to_sequences(sentence_list)
    encoded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    f.close()
    f = open(os.path.join(target_dir, 'encoded_minor.csv'), 'w')
    writer = csv.writer(f)
    for i in minor_index:
        writer.writerow([encoded_docs[i].tolist(), labels[i], i])

    f.close()
    f = open(os.path.join(target_dir, 'encoded_major.csv'), 'w')
    writer = csv.writer(f)
    for i in major_index:
        writer.writerow([encoded_docs[i].tolist(), labels[i], i])
    f.close()

def make_pairs_index(root_dir="processed_data", unit_length=200, split=0.2):
    f = open(os.path.join(root_dir, 'encoded_major.csv'), 'r')
    reader = csv.reader(f)
    c1_index_list = []
    c2_index_list = []
    i = 0
    for row in reader:
        label = row[1]
        if label == "c1":
            c1_index_list.append(i)
        elif label == "c2":
            c2_index_list.append(i)
        i += 1

    c1_picked = random.sample(c1_index_list, unit_length)
    c2_picked = random.sample(c2_index_list, unit_length)

    c1_part_1 = c1_picked[0: int(unit_length * 0.5)]
    c1_part_2 = c1_picked[int(unit_length * 0.5):]
    c2_part_1 = c2_picked[0: int(unit_length * 0.5)]
    c2_part_2 = c2_picked[int(unit_length * 0.5):]

    rows = []

    for i in range(int(unit_length * 0.5)):
        for j in range(i, int(unit_length * 0.5)):
            rows.append([[c1_part_1[i], c1_part_2[j]], 1])
            rows.append([[c2_part_1[i], c2_part_2[j]], 1])
            rows.append([[c1_part_1[i], c2_part_1[j]], 0])
            rows.append([[c2_part_2[i], c1_part_2[j]], 0])

    random.shuffle(rows)
    total_length = len(rows)
    train_length = int(total_length * (1 - split))
    train_rows = rows[0: train_length]
    val_rows = rows[train_length:]
    f = open(os.path.join("processed_data", 'train_index.csv'), 'w')
    writer = csv.writer(f)
    for row in train_rows:
        writer.writerow(row)
    f.close()

    f = open(os.path.join("processed_data", "test_index.csv"), 'w')
    writer = csv.writer(f)
    for row in val_rows:
        writer.writerow(row)
    f.close()

def get_encoded_data(encoded_file, picked_data):
    f1 = open(encoded_file, 'r')
    reader1 = csv.reader(f1)
    f2 = open(picked_data, 'r')
    reader2 = csv.reader(f2)
    sentenc_list_raw = []
    for row in reader2:
        sentenc_list_raw.append(row[0])
    data_list = []
    sentence_list = []
    for row in reader1:
        data_list.append(np.asarray(ast.literal_eval(row[0])))
        sentence_list.append(sentenc_list_raw[int(row[2])])
    return data_list, sentence_list

def get_index_list(index_file):
    index_list = []
    f = open(index_file, 'r')
    reader = csv.reader(f)
    for row in reader:
        pair = ast.literal_eval(row[0])
        label = int(row[1])
        index_list.append([pair, label])
    return index_list

def generaterator(batch_size, data_list, index_list):

    i = 0
    while True:
        x_batch_1 = []
        x_batch_2 = []
        y_batch = []
        for b in range(batch_size):
            # print len(index_list)
            if i >= (len(index_list) - 1):
                i = 0
            try:
                x1 = data_list[index_list[i][0][0]]
                x2 = data_list[index_list[i][0][1]]
            except:
                print(i)
                print len(index_list)
            y = index_list[i][1]
            i += 1
            x_batch_1.append(x1)
            x_batch_2.append(x2)
            y_batch.append(y)
        x_batch_1 = np.asarray(x_batch_1)
        x_batch_2 = np.asarray(x_batch_2)
        yield ([x_batch_1, x_batch_2], np.array(y_batch))

def plot_logs(csv_dir="result/training.csv"):
    f = open(csv_dir, 'r')
    reader = csv.reader(f)
    i = 0
    index = []
    train_loss = []
    test_loss = []

    for row in reader:
        if i > 0:
            train_loss.append(float(row[1]))
            test_loss.append(float(row[2]))
            index.append(i)
        i += 1
    plt.subplot(121)
    plt.plot(index, train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.xticks(index)

    plt.subplot(122)
    plt.plot(index, test_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.xticks(index)

    plt.show()

if __name__ == '__main__':
    # write_data()
    # major_minor_split_index()
    # make_pairs_index()
    data_list, sentence_list = get_encoded_data("processed_data/encoded_major.csv", "raw_data/picked_data.csv")
    index_list = get_index_list("processed_data/test_index.csv")
    gen = generaterator(32, data_list, index_list)
    i = 0
    while(True):
        batch_test = gen.next()
        # print batch_test[0]
        # print batch_test[1]
        # print i
        i += 1
    print "OK"