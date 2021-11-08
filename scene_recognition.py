import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from pathlib import Path, PureWindowsPath
import copy

## Start provided code

def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list

def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()

## End provided code


def get_tiny_image(img, output_size):
    shape_y, shape_x = img.shape

    mean = np.sum(img) / shape_x / shape_y
    img_zero_mean = img - mean
    img_zero_variance = img_zero_mean / np.std(img_zero_mean)

    output_x, output_y = output_size
    x_pad = shape_x % output_x
    y_pad = shape_y % output_y

    img_pad = np.pad(img_zero_variance, ((0, y_pad), (0, x_pad)), constant_values=(0))

    step_x = shape_x // output_x
    step_y = shape_y // output_y
    i = 0
    j = 0

    feature = np.zeros(shape=(output_x, output_y))
    while i < output_x:
        while j < output_y:
            pix = img_pad[j*step_y:(j+1)*step_y, i*step_x:(i+1)*step_x]
            mean_pix = np.sum(pix) / step_x / step_y
            feature[j, i] = mean_pix
            j += 1
        i += 1

    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(feature_train, label_train)
    label_test_pred = classifier.predict(feature_test)
    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, k=24, output_size=(16,16)):
    tiny_w, tiny_h = output_size
    tiny_img_train = np.zeros(shape=(len(img_train_list), tiny_h * tiny_w))
    tiny_img_test = np.zeros(shape=(len(img_test_list), tiny_w * tiny_h))

    ## Get tiny image of training images
    for index, img_path in enumerate(img_train_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        tiny_img_train[index, :] = get_tiny_image(img, (tiny_w, tiny_h)).flatten()

    ## Get tiny image of testing images
    for index, img_path in enumerate(img_test_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        tiny_img_test[index, :] = get_tiny_image(img, (tiny_w, tiny_h)).flatten()

    ## Do KNN stuff
    label_test_pred = predict_knn(tiny_img_train, label_train_list, tiny_img_test, k)

    ## Map labels to indices because confusion needs indices but prediction gives labels
    label_dict = {}
    for i in range(len(label_classes)):
        label_dict[label_classes[i]] = i

    ## Compute accuracy and confusion from predictions
    confusion = np.zeros(shape=(15, 15))
    accuracy_lst = np.zeros(len(label_classes))
    num_samples = np.zeros(len(label_classes))
    for index, prediction in enumerate(label_test_pred):
        predicted = label_dict[prediction]
        actual = label_dict[label_test_list[index]]
        confusion[actual, predicted] += 1
        num_samples[predicted] = num_samples[predicted] + 1
        if predicted == actual:
            accuracy_lst[actual] = accuracy_lst[actual] + 1

    ## Normalize confusion matrix by total observations for each label
    for row in range(confusion.shape[0]):
        confusion[row, :] = confusion[row, :] / num_samples[row]

    ## accuracy = SUM(% correctly labeled images in label i/ total images within label i) = % correctly labeled images / total images
    accuracy = np.sum(accuracy_lst / num_samples) / len(label_classes)
    # visualize_confusion_matrix(confusion, accuracy , label_classes)
    return confusion, accuracy


def compute_dsift(img, stride, size):
    kp = []
    x, y = img.shape
    for i in range(0, y, stride):
        for j in range(0, x, stride):
            kp.append(cv2.KeyPoint(j, i, size))

    sift = cv2.SIFT.create()
    keypoints, dense_feature = sift.compute(img, kp)
    return dense_feature


def build_visual_dictionary(dense_feature_list, dic_size, n_init=10, max_iter=300):
    kmeans = KMeans(n_clusters=dic_size, n_init=n_init, max_iter=max_iter)
    kmeans.fit(dense_feature_list)
    vocab = kmeans.cluster_centers_
    return vocab


def compute_bow(feature, vocab, dic_size=50):
    histogram = np.zeros(shape=(1,dic_size))
    NN = NearestNeighbors(n_neighbors=1).fit(vocab)
    d, ind = NN.kneighbors(feature)
    for i in ind:
        histogram[0, i[0]] += 1

    bow_feature = histogram / np.sum(histogram)
    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, k=24, dic_size=50, n_init=10, max_iter=300, stride=20, size=20):
    dsift_train = np.zeros(shape=(0,128))
    try:
        dsift_train = np.loadtxt('dsift_train.txt')
    except OSError:
        print('computing dsift')
        for index, im_path in enumerate(img_train_list):
            print('computing dsift ', index, '/', len(img_train_list))
            im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            dsift = compute_dsift(im, stride,size)
            dsift_train = np.append(dsift_train, dsift, axis=0)
        np.savetxt('dsift_train.txt', dsift_train)
    try:
        vocab = np.loadtxt('vocab.txt')
    except OSError:
        print('bulding vocab')
        vocab = build_visual_dictionary(dsift_train, dic_size, n_init, max_iter)
        np.savetxt('vocab.txt',vocab)
    bow_train = np.zeros(shape=(0, dic_size))
    try:
        bow_train = np.loadtxt('bow_train.txt')
        print('found bow_train.txt')
    except OSError:
        print('beginning bow computation')
        for index, im_path in enumerate(img_train_list):
            print('computing bow ', index, '/', len(img_train_list))
            im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            feature = compute_dsift(im, stride, size)
            bow = compute_bow(feature, vocab, dic_size)
            bow_train = np.append(bow_train, bow, axis=0)
        np.savetxt('bow_train.txt', bow_train)

    bow_test = np.zeros(shape=(0, dic_size))
    try:
        bow_test = np.loadtxt('bow_test.txt')
        print('found bow_test.txt')
    except OSError:
        print('beginning dense feature test computation')
        for index, im_path in enumerate(img_test_list):
            print('computing bow test ', index, '/', len(img_test_list))
            im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            feature = compute_dsift(im, stride, size)
            bow = compute_bow(feature, vocab, dic_size)
            bow_test = np.append(bow_test, bow, axis=0)
        np.savetxt('bow_test.txt', bow_test)

    print('beginning prediction')
    label_test_pred = predict_knn(bow_train, label_train_list, bow_test, k)

    ## Map labels to indices because confusion needs indices but prediction gives labels
    label_dict = {}
    for i in range(len(label_classes)):
        label_dict[label_classes[i]] = i

    ## Compute accuracy and confusion from predictions
    confusion = np.zeros(shape=(15, 15))
    accuracy_lst = np.zeros(len(label_classes))
    num_samples = np.zeros(len(label_classes))
    for index, prediction in enumerate(label_test_pred):
        predicted = label_dict[prediction]
        actual = label_dict[label_test_list[index]]
        confusion[actual, predicted] += 1
        num_samples[actual] = num_samples[actual] + 1
        if predicted == actual:
            accuracy_lst[predicted] = accuracy_lst[predicted] + 1

    ## Normalize Confusion matrix by total observations for each label
    for row in range(confusion.shape[0]):
        confusion[row, :] = confusion[row, :] / num_samples[row]

    ## accuracy = SUM(accurately predicted samples per class / total samples per class) / total classes
    accuracy = np.sum(accuracy_lst / num_samples) / len(label_classes)
    # visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes, C=1):

    ## Construct list of all labels in training set
    label_arr = []
    for label in label_train:
        if label not in label_arr:
            label_arr += [label]

    ## Initialize and fit n_classes SVM classifiers
    classifiers = []
    for i in range(n_classes):
        temp_label_train = label_train[:]
        for index, label in enumerate(temp_label_train):
            if label != label_arr[i]:
                temp_label_train[index] = 'not ' + label_arr[i]
        classifier = SVC(C=C,gamma='scale', probability=True)
        classifier.fit(feature_train, temp_label_train)
        classifiers += [copy.deepcopy(classifier)]

    scores = np.zeros(shape=(n_classes, len(img_test_list)))
    for clss in range(n_classes):
        res = classifiers[clss].predict_proba(feature_test)[:,0]
        scores[clss, :] = res


    max_ind = np.argmax(scores, axis=0)
    label_test_pred = np.array(label_arr)[max_ind]
    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, n_classes=15, n_init=10, max_iter=300,dic_size=50, C=1):
    dsift_train = np.zeros(shape=(0, 128))
    stride, size = 15, 15
    try:
        dsift_train = np.loadtxt('dsift_train.txt')
        print('found dsift train'
              '')
    except OSError:
        print('computing dsift')
        for index, im_path in enumerate(img_train_list):
            print('computing dsift ', index, '/', len(img_train_list))
            im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            dsift = compute_dsift(im, stride, size)
            dsift_train = np.append(dsift_train, dsift, axis=0)
        np.savetxt('dsift_train.txt', dsift_train)
    try:
        vocab = np.loadtxt('vocab.txt')
    except OSError:
        print('bulding vocab')
        vocab = build_visual_dictionary(dsift_train, dic_size, n_init, max_iter)
        np.savetxt('vocab.txt', vocab)
    bow_train = np.zeros(shape=(0, dic_size))
    try:
        bow_train = np.loadtxt('bow_train.txt')
        print('found bow_train.txt')
    except OSError:
        print('beginning bow computation')
        for index, im_path in enumerate(img_train_list):
            print('computing bow ', index, '/', len(img_train_list))
            im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            feature = compute_dsift(im, stride, size)
            bow = compute_bow(feature, vocab, dic_size)
            bow_train = np.append(bow_train, bow, axis=0)
        np.savetxt('bow_train.txt', bow_train)

    bow_test = np.zeros(shape=(0, dic_size))
    try:
        bow_test = np.loadtxt('bow_test.txt')
        print('found bow_test.txt')
    except OSError:
        print('beginning dense feature test computation')
        for index, im_path in enumerate(img_test_list):
            print('computing bow test ', index, '/', len(img_test_list))
            im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            feature = compute_dsift(im, stride, size)
            bow = compute_bow(feature, vocab, dic_size)
            bow_test = np.append(bow_test, bow, axis=0)
        np.savetxt('bow_test.txt', bow_test)

    print('beginning prediction')
    label_test_pred = predict_svm(bow_train, label_train_list, bow_test, n_classes, C=C)

    ## Map labels to indices because confusion needs indices but prediction gives labels
    label_dict = {}
    for i in range(len(label_classes)):
        label_dict[label_classes[i]] = i


    ## Compute accuracy and confusion from predictions
    confusion = np.zeros(shape=(15, 15))
    accuracy_lst = np.zeros(n_classes)
    num_samples = np.zeros(n_classes)
    for index, prediction in enumerate(label_test_pred):
        predicted = label_dict[prediction]
        actual = label_dict[label_test_list[index]]
        confusion[actual, predicted] += 1
        num_samples[actual] = num_samples[actual] + 1
        if predicted == actual:
            accuracy_lst[predicted] = accuracy_lst[predicted] + 1

    ## Normalize Confusion matrix by total observations for each label
    for row in range(confusion.shape[0]):
        confusion[row, :] = confusion[row, :] / num_samples[row]

    ## accuracy = SUM(accurately predicted samples per class / total samples per class) / total classes
    accuracy = np.sum(accuracy_lst / num_samples) / n_classes
    # visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    _, acc = classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    print(acc)
    _, acc = classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    print(acc)
    _, acc = classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    print(acc)





