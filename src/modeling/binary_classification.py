from load_data import read_and_clean
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import json
from sklearn.metrics import confusion_matrix
from pca import get_y_labels
import matplotlib.pyplot as plt
import seaborn as sb
import os

BEST_C = 5.5
Softmax_Best_C = 0.025
OUT_BASE = "./out/binary_classifier/"

def save_matrix(m, y_labels, out_name, avg_score):
    plt.figure()
    mask = np.zeros_like(m)
    mask[np.triu_indices_from(mask)] = True
    ax = sb.heatmap(pd.DataFrame(m, columns=y_labels,
                                 index=y_labels), annot=True, cbar=False, fmt=".2f", mask=mask, cmap='inferno', vmin=0.8, vmax=1.0)
    b, t = plt.ylim()
    b += 0.5 
    t -= 0.5 
    plt.ylim(b, t)
    plt.gcf().subplots_adjust(bottom=0.25, left=0.20)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    text  = "mean accuracy = {:.3f}".format(avg_score)
    ax.text(0.6, 0.95, text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
    plt.title("{}".format(out_name))
    plt.savefig("{}/{}.png".format(OUT_BASE, out_name))


def binary_all(clf, train_X, train_y, dev_X, dev_y, name, headers):
    y_labels = get_y_labels()
    k = len(y_labels)
    result = np.zeros([k,k])

    total_score = 0
    c = 0
    #dictionary to store # occurences of a feature within
    #top 20 pos and top 20 neg features for each pair
    top_feature_occurences = dict.fromkeys(headers, 0)
    for i in range(k):
        for j in range(i + 1, k):
            c += 1
            indices_train = (train_y == i) | (train_y == j)
            indices_dev = (dev_y == i) | (dev_y == j)

            #make index j labels 1 and index i labels 0 for easy coef_ interpretation
            new_train_y = train_y[indices_train]
            new_train_y = np.array([1 if y==j else 0 for y in new_train_y])
            new_dev_y = dev_y[indices_dev]
            new_dev_y = np.array([1 if y==j else 0 for y in new_dev_y])
            
            clf.fit(train_X[indices_train], new_train_y)
            result[j][i] = clf.score(dev_X[indices_dev], new_dev_y)
            print(result[j][i])
            total_score += result[j][i]
            #plot top features for softmax
            if type(clf) == LogisticRegression:
                top_features = importance_plot(clf, headers, "top_features_{}_vs_{}".format(y_labels[j], y_labels[i]))
                for f in top_features:
                    top_feature_occurences[f] += 1
                
    avg_score = total_score / c
    save_matrix(result, y_labels, name, avg_score)

    #print 50 most frequently occuring features
    if type(clf) == LogisticRegression:
        print("Here are the 50 features which occur most frequently as the top cofficients for all composer pairs")
        sorted_d = sorted(top_feature_occurences.items(), key=lambda x: x[1])
        sorted_d = sorted_d[::-1]
        for i in range(min(50, train_X.shape[1])):
            print(sorted_d[i])

#plot the features corresponding to 20 most positive and 20 most negative
#coefficients in the already-fitted classifier clf
def importance_plot(clf, feature_list, out_name, n_features=20):
    coef = clf.coef_.ravel()
    if coef.shape[0] >= 80:
        top_pos_coef = np.argsort(coef)[-n_features:]
        top_neg_coef = np.argsort(coef)[:n_features]
        top_coef = np.hstack([top_neg_coef, top_pos_coef])
    else:
        top_coef = np.argsort(coef)
        
    plt.figure(figsize=(15, 10))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coef]]
    plt.bar(np.arange(2 * n_features), coef[top_coef], color=colors)
    feature_list = np.array(feature_list)
    plt.xticks(np.arange(1, 1 + 2 * n_features), feature_list[top_coef],
               rotation=60, ha='right', fontsize='medium')
    plt.title(out_name)
    plt.tight_layout()
    plt.savefig("{}/{}.png".format(OUT_BASE, out_name))
    plt.clf()
    plt.close()

    #return the top feature names
    return feature_list[top_coef]
    
def main():
    if not os.path.exists(OUT_BASE):
        os.makedirs(OUT_BASE)
    train_X, train_y, dev_X, dev_y, headers = read_and_clean(standardize=True)
    
    clf_svm = SVC(gamma='auto', kernel='rbf', C=BEST_C)
    clf_softmax = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000, C=Softmax_Best_C)

    classifiers = [(clf_svm, "SVM"), (clf_softmax, "Softmax")]

    for clf, name in classifiers:
        binary_all(clf, train_X, train_y, dev_X, dev_y, name, headers)

if __name__ == "__main__":
    main()
