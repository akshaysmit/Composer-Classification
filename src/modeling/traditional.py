import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import seaborn as sb

#these parameters are found by running the tuning functions for each type of classifier
#see out/hyperparams for the corresponding accuracy vs. hyperparameter plots
SVM_Best_C = 5.5
Softmax_Best_C = 0.025
knn_Best_k = 10

def get_y_labels():
    extraction_arg_path = "../data_cleaning/extraction_arguments.json"
    f = open(extraction_arg_path)
    return json.load(f)["composer_names"]

def run_svm(train_X, train_y, dev_X, dev_y, composer_list):
    print("\n")
    print("SVM 1v1 on all composers:")

    clf = SVC(kernel='rbf', gamma='auto', C=SVM_Best_C)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(train_X)
    accuracy = accuracy_score(train_y, y_pred)
    print('Train Accuracy: ', accuracy)
    confusion_m = confusion_matrix(train_y, y_pred)
    save_confusion_m(confusion_m, composer_list, "SVM_train_C={}".format(SVM_Best_C), accuracy)

    y_pred = clf.predict(dev_X)
    accuracy = accuracy_score(dev_y, y_pred)
    print('Dev Accuracy: ', accuracy)
    confusion_m = confusion_matrix(dev_y, y_pred)
    save_confusion_m(confusion_m, composer_list, "SVM_dev_C={}".format(SVM_Best_C), accuracy)

    print("Metrics on dev set: ")
    metrics = print_additional_metrics(dev_y, y_pred, composer_list)
    
def run_softmax(train_X, train_y, dev_X, dev_y, composer_list):
    clf = LogisticRegression(multi_class = 'multinomial', solver='lbfgs', penalty='l2', max_iter=1000, C=Softmax_Best_C)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(train_X)
    print("Softmax on all composers:\n")
    accuracy = accuracy_score(train_y, y_pred)
    print('Train Accuracy: ', accuracy)
    confusion_m = confusion_matrix(train_y, y_pred)
    save_confusion_m(confusion_m, composer_list, "softmax_train_C={}".format(Softmax_Best_C), accuracy)
    
    y_pred = clf.predict(dev_X)
    accuracy = accuracy_score(dev_y, y_pred)
    print('Dev Accuracy: ', accuracy)
    confusion_m = confusion_matrix(dev_y, y_pred)
    save_confusion_m(confusion_m, composer_list, "softmax_dev_C={}".format(Softmax_Best_C), accuracy)

    print("Metrics on dev set: ")
    metrics = print_additional_metrics(dev_y, y_pred, composer_list)
    
def run_knn(train_X, train_y, dev_X, dev_y, composer_list):
    clf = KNeighborsClassifier(n_neighbors=knn_Best_k)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(train_X)
    print("knn on all composers:\n")
    accuracy = accuracy_score(train_y, y_pred)
    print('Train Accuracy: ', accuracy)
    confusion_m = confusion_matrix(train_y, y_pred)
    save_confusion_m(confusion_m, composer_list, "knn_train_k={}".format(knn_Best_k), accuracy)
    
    y_pred = clf.predict(dev_X)
    accuracy = accuracy_score(dev_y, y_pred)
    print('Dev Accuracy: ', accuracy)
    confusion_m = confusion_matrix(dev_y, y_pred)
    save_confusion_m(confusion_m, composer_list, "knn_dev_k={}".format(knn_Best_k), accuracy)

    print("Metrics on dev set: ")
    metrics = print_additional_metrics(dev_y, y_pred, composer_list)
    
def run_lda(train_X, train_y, dev_X, dev_y, composer_list):
    clf = LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')
    clf.fit(train_X, train_y)
    y_pred = clf.predict(train_X)
    print("LDA on all composers:\n")
    accuracy = accuracy_score(train_y, y_pred)
    print('Train Accuracy: ', accuracy)
    confusion_m = confusion_matrix(train_y, y_pred)
    save_confusion_m(confusion_m, composer_list, "lda_train", accuracy)

    y_pred = clf.predict(dev_X)
    accuracy = accuracy_score(dev_y, y_pred)
    print('Dev Accuracy: ', accuracy)
    confusion_m = confusion_matrix(dev_y, y_pred)
    save_confusion_m(confusion_m, composer_list, "lda_dev", accuracy)

    print("Metrics on dev set: ")
    metrics = print_additional_metrics(dev_y, y_pred, composer_list)
    
def softmax_C_tuning(train_X, train_y, dev_X, dev_y):
    print("\n")
    print("Softmax C parameter tuning with l2 penalty:")
    C_list = np.arange(0.005, 0.5, 0.005)
    acc = []
    for C in C_list:
        print('C = ', C)
        clf = LogisticRegression(multi_class = 'multinomial', solver='lbfgs', penalty='l2', max_iter=1000, C=C)
        clf.fit(train_X, train_y)
        y_pred =  clf.predict(dev_X)
        acc.append(accuracy_score(dev_y, y_pred))
        
    ind = np.argmax(np.array(acc))
    best_C = C_list[ind]
    print("best C value is: ", best_C)

    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.title('C parameter tuning for softmax with l2 penalty')
    plt.plot(C_list, acc, 'r-x')
    plt.savefig('out/hyperparams/C_softmax_all_features.png')
    plt.clf()
    return best_C
    
def svm_C_tuning(train_X, train_y, dev_X, dev_y):
    print("\n")
    print("SVM 1v1 C parameter tuning with rbf kernel:")
    C_list = np.arange(2, 15, 0.5)
    acc = []
    for C in C_list:
        clf = SVC(kernel='rbf', gamma='auto', C=C)
        clf.fit(train_X, train_y)
        y_pred =  clf.predict(dev_X)
        acc.append(accuracy_score(dev_y, y_pred))

    ind = np.argmax(np.array(acc))
    best_C = C_list[ind]
    print("best C value is: ", best_C)

    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.title('C parameter tuning for SVM with rbf kernel')
    plt.plot(C_list, acc, 'r-x')
    plt.savefig('out/hyperparams/C_svm_all_features.png')
    plt.clf()
    return best_C

def knn_k_tuning(train_X, train_y, dev_X, dev_y):
    print("\n")
    print("Tuning k for knn classifier:")
    k_list = np.arange(1, 50, 1)
    acc = []
    for k in k_list:
        print("k = ", k)
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(train_X, train_y)
        y_pred =  clf.predict(dev_X)
        acc.append(accuracy_score(dev_y, y_pred))

    ind = np.argmax(np.array(acc))
    best_k = k_list[ind]
    print("best k value is: ", best_k)

    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('k parameter tuning for knn classifier with l2 distance')
    plt.plot(k_list, acc, 'r-x')
    plt.savefig('out/hyperparams/k_knn_all_features.png')
    plt.clf()

palette = sb.color_palette("Paired") + sb.color_palette("Set2")

def print_additional_metrics(y_true, y_pred, composer_list):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    accuracies = []    
    for i in range(len(composer_list)):
        mask = y_true == i
        numer = 0
        for j in range(len(y_true)):
            if y_true[j] == i and y_pred[j] == i:
                numer += 1
        accuracies.append(numer/np.sum(mask))
        
    metrics = {'precision': precision, 'recall': recall, 'f1_score': f1, 'per-class accuracy': accuracies}
    
    for key, value in metrics.items():
        print("\n")
        print(key, ":")
        for i in range(len(composer_list)):
            print(composer_list[i], " : ", value[i])
    
    return metrics

def save_confusion_m(confusion_m, y_labels, out_name, accuracy):
    plt.figure()
    ax = sb.heatmap(pd.DataFrame(confusion_m, columns=y_labels,
                                 index=y_labels), annot=True, cbar=False, fmt="d")
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)
    plt.gcf().subplots_adjust(bottom=0.25, left=0.20)
    plt.title("{}, accuracy = {}".format(out_name, accuracy))
    plt.savefig("out/traditional/{}.png".format(out_name))
    
def main():
    train_path = "../data_set_generator/train.csv"
    dev_path = "../data_set_generator/dev.csv"
    test_path = "../data_set_generator/test.csv"

    #drop Note_Density_per_Quarter_Note_Variability as it has too many bad values
    train = pd.read_csv(train_path)
    train_y = train[['y']]
    train_y = np.ndarray.flatten(train_y.to_numpy())
    train_X = train.drop(["y", "Note_Density_per_Quarter_Note_Variability"], axis=1)
    
    dev = pd.read_csv(dev_path)
    dev_y = dev[['y']]
    dev_y = np.ndarray.flatten(dev_y.to_numpy())
    dev_X = dev.drop(["y", "Note_Density_per_Quarter_Note_Variability"], axis=1)

    #NaN removal from remaining dataset (this removes very few examples)
    train_X = train_X.to_numpy()
    train_X = train_X.astype(np.float64)
    train_y = train_y[~np.isnan(train_X).any(axis=1)]
    train_X = train_X[~np.isnan(train_X).any(axis=1)]
    dev_X = dev_X.to_numpy()
    dev_X = dev_X.astype(np.float64)
    dev_y = dev_y[~np.isnan(dev_X).any(axis=1)]
    dev_X = dev_X[~np.isnan(dev_X).any(axis=1)]

    train_X, dev_X = [StandardScaler().fit_transform(data)
                          for data in [train_X, dev_X]]

    #Uncomment these to do hyperparameter tuning
    #svm_C_tuning(train_X, train_y, dev_X, dev_y)
    #softmax_C_tuning(train_X, train_y, dev_X, dev_y)
    #knn_k_tuning(train_X, train_y, dev_X, dev_y)

    composer_list = get_y_labels()
    
    run_svm(train_X, train_y, dev_X, dev_y, composer_list)
    #run_softmax(train_X, train_y, dev_X, dev_y, composer_list)
    #run_knn(train_X, train_y, dev_X, dev_y, composer_list)
    #run_lda(train_X, train_y, dev_X, dev_y, composer_list)
    
if __name__ == "__main__":
    main()
