from numpy import *
import ast
import csv
import numpy as np
from scipy import stats

def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))

def initCentroids(dataSet, k, index_list):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    centroids_index = []
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
        centroids_index.append(index_list[index])
    return centroids, centroids_index


def kmeans(dataSet, k, index_list):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 3)))
    clusterChanged = True

    ## step 1: init centroids
    centroids, centroids_index = initCentroids(dataSet, k, index_list)

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in xrange(numSamples):
            minDist = 100000.0
            minIndex = 0
            minOriginalIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, 0:2] = minIndex, minDist ** 2
            clusterAssment[i, 2] = index_list[i]

        ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)

    print 'Congratulations, cluster complete!'
    return centroids, clusterAssment

def kmeans_detector(dataSet, k, index_list, given_centroids):
    numSamples = dataSet.shape[0]
    detected_number = 0
    centroids = given_centroids

    ## for each sample

    for i in xrange(numSamples):
        if i > k:
            sample_window = []
            for index in range(i - k, i):
                sample_window.append(dataSet[index, :])

        minDist = 100000.0
        minIndex = 0

        ## for each centroid
        distance_centroids = []
        for j in range(k):
            distance = euclDistance(centroids[j, :], dataSet[i, :])
            person_score = stats.pearsonr(centroids[j, :], dataSet[i, :])
            distance_centroids.append(1 - abs(person_score[0]))
            if distance < minDist:
                minDist = distance
                minIndex = j
        if i > k:
            status, internel_distance, std_among_centroids = monitor(sample_window, distance_centroids)
            if status > 623896:
                detected_number += 1
                print "Drift Detected at : %d"%i
                print status, internel_distance, std_among_centroids
                print "+++++++++++++++++++++++++++++++++++"
            # elif status > 6:
            #     print "Drift Warning found at : %d"%i

    print 'Detected: %d drifts' % detected_number


def monitor(sample_window, distance_centroids):
    sample_shape = sample_window[0].shape
    sample_sum = np.zeros(sample_shape)
    for sample in sample_window:
        sample_sum += sample
    sample_mean = sample_sum/float(len(sample_window))
    internel_distance = 0
    for sample in sample_window:
        internel_distance += euclDistance(sample_mean, sample)

    std_among_centroids = np.std(distance_centroids)
    sum_centroids = (np.sum(distance_centroids) * 1000000.) ** 2

    status = sum_centroids - internel_distance - std_among_centroids * 1000000000.
    return status, internel_distance, std_among_centroids

def evaluate_clustering_acc(clusterAssment, labels):
    dictonary = {}
    for i in range(clusterAssment.shape[0]):
        category = clusterAssment[i, 0]
        if category not in dictonary.keys():
            dictonary[category] = [clusterAssment[i, 2]]
        else:
            dictonary[category].append(clusterAssment[i, 2])

    print dictonary
def load_text_2_array(featrues_file):
    f = open(featrues_file, 'r')
    reader = csv.reader(f)
    index = []
    features = []
    labels = []
    for row in reader:
        index.append(int(row[0]))
        features.append(np.asarray(ast.literal_eval(row[1])))
        # if row[1] == "c1":
        #     labels.append(0)
        # elif row[1] == "c2":
        #     labels.append(1)
    features = np.asarray(features)
    return index, features, labels

if __name__ == '__main__':
    index, features, labels = load_text_2_array("processed_data/features_major.csv")
    centroids, clusterAssment = kmeans(features, 2, index)

    index, features, labels = load_text_2_array("processed_data/features_minor.csv")
    kmeans_detector(features, 2, index, centroids)
