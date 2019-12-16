import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
import os

from load_data import read_and_clean


OUT_BASE = "./out/pca"
BEST_C = 5.5 #chosen using validation set with all features, see the plot in out/hyperparams


def get_y_labels():
    extraction_arg_path = "../data_cleaning/extraction_arguments.json"
    f = open(extraction_arg_path)
    return json.load(f)["composer_names"]


def save_confusion_m(confusion_m, y_labels, out_name, accuracy):
    plt.figure()
    ax = sb.heatmap(pd.DataFrame(confusion_m, columns=y_labels,
                                 index=y_labels), annot=True, cbar=False, fmt="d")
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)
    plt.gcf().subplots_adjust(bottom=0.25, left=0.20)
    plt.title("{}, accuracy = {:3f}".format(out_name, accuracy))
    plt.savefig("{}/{}.png".format(OUT_BASE, out_name))


def eval_classifier(clf, data, clf_name, y_labels):
    train_X, train_y, dev_X, dev_y = data
    for X, y, name in [(train_X, train_y, "train"), (dev_X, dev_y, "dev")]:
        y_pred = clf.predict(X)
        confusion_m = confusion_matrix(y, y_pred)
        accuracy = clf.score(X, y)
        save_confusion_m(confusion_m, y_labels,
                         "{}_{}".format(clf_name, name), accuracy)


palette = sb.color_palette("Paired") + sb.color_palette("Set2")

def save_scatter(X, y, y_labels, name):
    plt.figure()
    scatter_data_frame = pd.DataFrame(
        {'X': X[:, 0], 'Y': X[:, 1], "label": [y_labels[elem] for elem in y]})
    sb.scatterplot(x='X', y='Y', data=scatter_data_frame)

    plot = sb.scatterplot(x="X", y="Y",
                    hue="label",
                    legend='full',
                    palette=palette[0:np.unique(y).shape[0]],
                    data=scatter_data_frame)
    plt.title(name)
    left,right = plt.xlim()
    plt.xlim(left, right + (right - left) * 0.3)
    plt.savefig("{}/{}.png".format(OUT_BASE,name))
    plt.close()

def save_scatter_3d(X,y, y_labels, name):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #points = [X[y == k] for k, _ in enumerate(y_labels)]
    colors = [palette[y_l] for y_l in y]
    #for i, Xi in enumerate(points):
    ax.scatter(X[:,0], X[:,1], X[:,2], c = colors, s=60)
    ax.view_init(30, 185)
    plt.savefig("{}/{}.png".format(OUT_BASE,name))
    plt.title(name)
    plt.close()



def main():

    if not os.path.exists(OUT_BASE):
        os.makedirs(OUT_BASE)

    group_composers = True

    train_X, train_y, dev_X, dev_y, headers = read_and_clean(standardize=True,group_composers=group_composers)


    y_labels = get_y_labels()
    if group_composers:
        #[["bach", "handel", "vivaldi","telemann"], ["haydn","mozart"], ["brahms","chopin", "debussy","liszt","mendelssohn", "dvorak"]]
        y_labels = ["baroque", "classical", "romantic"]
    ########## Visualize the high dim features with PCA ##########
    pca = PCA(n_components=2)
    pca.fit(train_X)
    PCA_train_X = pca.transform(train_X)
    PCA_dev_X = pca.transform(dev_X)

    #print(pca.explained_variance_)
    #print(pca.components_.shape)
    #component_lengths = np.linalg.norm(pca.components_, axis=0)
    #arg_sorted = np.argsort(component_lengths)[::-1]
    #print(component_lengths[arg_sorted])
    #print(arg_sorted)
    #print(headers[arg_sorted])

    save_scatter(PCA_train_X, train_y, y_labels, "PCA")

    for i, y_label in enumerate(y_labels):
        X_y = (train_y == i).astype(int)
        arr_sort = X_y.argsort()
        save_scatter(PCA_train_X[arr_sort], X_y[arr_sort], ["other", y_label], "PCA_{}_vs_all".format(y_label))

    k = len(y_labels)
    for i in range(k):
        for j in range(i + 1,k):
            n1 = y_labels[i]
            n2 = y_labels[j]
            indices = (train_y == i) | (train_y == j)
            save_scatter(PCA_train_X[indices], train_y[indices], y_labels, "PCA_{}_vs_{}".format(n1, n2))
            

        

    #### 3d ##########
    pca = PCA(n_components=3)
    pca.fit(train_X)
    PCA_train_X = pca.transform(train_X)
    PCA_dev_X = pca.transform(dev_X)

    save_scatter_3d(PCA_train_X, train_y, y_labels, "PCA_3d")


    ########## Visualize the high dim features with T-SNE ##########
    pca = PCA(n_components=50)
    pca.fit(train_X)
    PCA_train_X = pca.transform(train_X)

    PCA_dev_X = pca.transform(dev_X)
    X_embedded = TSNE(n_components=2).fit_transform(PCA_train_X)

    for i, y_label in enumerate(y_labels):
        X_y = (train_y == i).astype(int)
        arr_sort = X_y.argsort()
        save_scatter(X_embedded[arr_sort], X_y[arr_sort], ["other", y_label], "t-SNE_{}_vs_all".format(y_label))

    k = len(y_labels)
    for i in range(k):
        for j in range(i + 1,k):
            n1 = y_labels[i]
            n2 = y_labels[j]
            indices = (train_y == i) | (train_y == j)
            save_scatter(X_embedded[indices], train_y[indices], y_labels, "t-SNE_{}_vs_{}".format(n1, n2))
    #X_embedded_3 = TSNE(n_components=3).fit_transform(PCA_train_X)

    save_scatter(X_embedded, train_y, y_labels, "t-SNE")
    #save_scatter_3d(X_embedded_3, train_y, y_labels, "t-SNE_3d")

    ########## Predict reduced number of features ##########
    K = [2, 5, 10, 20, 50, 100, 200, 500]
    for k in K:
        print("predict with {} features".format(k))
        pca = PCA(n_components=k)
        pca.fit(train_X)
        PCA_train_X = pca.transform(train_X)
        PCA_dev_X = pca.transform(dev_X)

        clf_pca = SVC(gamma='auto', kernel='rbf', C=BEST_C)
        clf_pca.fit(PCA_train_X, train_y)
        out_name =  "{}-features-group".format(k) if group_composers else "{}-features".format(k)
        eval_classifier(clf_pca, (PCA_train_X, train_y,
                                  PCA_dev_X, dev_y), out_name, y_labels)

    ########## Predict with all the features ##########
    clf = SVC(gamma='auto', kernel='rbf', C=BEST_C)
    clf.fit(train_X, train_y)
    eval_classifier(clf, (train_X, train_y, dev_X, dev_y), "all_features",y_labels)


if __name__ == "__main__":
    main()
