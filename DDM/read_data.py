import ast
import numpy as np
import csv
from matplotlib import pyplot as plt

def get_encoded_data(encoded_file="data/encoded_major.csv"):
    f1 = open(encoded_file, 'r')
    reader1 = csv.reader(f1)

    data_list = []
    label_list = []
    for row in reader1:
        data_list.append(np.asarray(ast.literal_eval(row[0])))
        if row[1] == "c1":
            label_list.append(1)
        elif row[1] == "c2":
            label_list.append(0)
    random_index = np.random.permutation(len(data_list))
    x = np.asarray(data_list)
    y = np.asarray(label_list)
    x = x[random_index]
    y = y[random_index]
    return x, y

def plot_logs(csv_dir="result/training.csv"):
    f = open(csv_dir, 'r')
    reader = csv.reader(f)
    i = 0
    index = []
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for row in reader:
        if i > 0:
            train_loss.append(float(row[2]))
            test_loss.append(float(row[4]))
            train_acc.append(float(row[1]))
            test_acc.append(float(row[3]))
            index.append(i)
        i += 1
    plt.subplot(221)
    plt.plot(index, train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.xticks(index)

    plt.subplot(222)
    plt.plot(index, test_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.xticks(index)

    plt.subplot(223)
    plt.plot(index, train_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.xticks(index)

    plt.subplot(224)
    plt.plot(index, test_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.xticks(index)

    plt.show()

if __name__ == '__main__':
    # get_encoded_data()
    plot_logs()