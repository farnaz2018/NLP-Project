import csv
from matplotlib import pyplot as plt
import numpy as np


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

    print cat_dict
    main_id_dict = {3:'Real Estate', 2:'Jobs'}
    sub_id_dict_1 = {11:'Lawyers', 12:'Administrative - Secretary', 14:'Call cente', 15:'Building',
                     16:'Accounting finance', 17:"Education - Teachers", 19:'Customer Support', 20:'Bar and Restaurant',
                     21:'Biotechnology', 22:'Retail', 23:'Technical support', 24:'Work from home', 26:'Transport',
                     27:'Medicine - Health', 28:'fashion', 29:'Advertising - Marketing', 30:'Human Resources',
                     31:'Public relations', 32:'Sellers', 33:'Engineers - Architects', 34:'software', 35:'Wholesales', 122:'Other offers',
                     132:'Travels and tourism', 134:'Administration - Executives'
                     }
    sub_id_dict_2 = {2:'Apartment House for sale', 51:'Apartment - House for rent'}
    print sub_cat_dict_2
    print sub_cat_dict_3
    # dict_2_plot(cat_dict, main_id_dict)
    # dict_2_plot(sub_cat_dict_2, sub_id_dict_1)
    # dict_2_plot(sub_cat_dict_3, sub_id_dict_2)
    stacked_id_majority = {16:'Accounting finance', 2:'Apartment House for sale'}
    stacked_id_minority = {28: 'fashion', 51: 'Apartment - House for rent'}
    dict_2_plot_stacked(sub_cat_dict_2, sub_cat_dict_3, stacked_id_majority, stacked_id_minority)

def dict_2_plot(dict, id_dict):
    label_list = []
    values = []
    for key in dict.keys():
        label_list.append(id_dict[int(key)])
        values.append(dict[str(key)])
    ind = np.arange(len(label_list))

    plt.bar(ind, values)

    plt.ylabel('number')
    plt.xlabel('category')
    rotation = 0
    if len(dict.keys()) > 10:
        rotation=270
    plt.xticks(ind, label_list, rotation=rotation)
    plt.show()

def dict_2_plot_stacked(dict1, dict2, id_dict1, id_dict2):
    label_list = []
    values1 = []
    values2 = []
    dict1.update(dict2)
    for key in id_dict1.keys():
        # label_list.append(id_dict1[int(key)])
        values1.append(dict1[str(key)])
    for key in id_dict2.keys():
        # label_list.append(id_dict2[int(key)])
        values2.append(dict1[str(key)])


    label_list = ["Jobs", "Real Estate"]
    ind = np.arange(len(label_list))
    values1 = np.array(values1)
    values2 = np.array(values2)
    p1 = plt.bar(ind, values1,)
    p2 = plt.bar(ind, values2, bottom=values1)

    plt.ylabel('number')
    plt.xlabel('category')
    plt.legend((p1[0], p2[0]), ('major', 'minor'), loc='upper right')
    rotation = 0
    if len(dict1.keys()) > 10:
        rotation=270
    plt.xticks(ind, label_list, rotation=rotation)
    plt.show()



if __name__ == '__main__':
    read_data_raw()
