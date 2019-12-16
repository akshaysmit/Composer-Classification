import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sb
from load_data import read_and_clean
from traditional import get_y_labels

#Uses gradient boosting with decision stumps, and sample bagging 
#The following parameters are found by manual search
application = 'multiclass'
bagging_fraction = 0.6
bagging_freq = 1
lambda_l2 = 10.0
n_estimators = 700

palette = sb.color_palette("Paired") + sb.color_palette("Set2")

#n_features is the number of features we want to plot
#feature_importances is the corresponding attribute of lgbclf
#imp_measure is 'split' or 'gain' depending on lgbclf
def plot_feature_importance(feature_importances, headers, n_features, imp_measure):
    feature_imp = pd.DataFrame(sorted(zip(feature_importances, headers))[-n_features:], columns=['Value','Feature'])
    plt.figure(figsize=(20, 10))
    sb.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM feature importances: {}'.format(imp_measure))
    plt.tight_layout()
    plt.savefig('out/ensemble/feature_importances_{}.png'.format(imp_measure))
    plt.clf()
    
def print_additional_metrics(y_true, y_pred, composer_list):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    metrics = {'precision': precision, 'recall': recall, 'f1_score': f1}

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
    plt.savefig("out/ensemble/{}.png".format(out_name))

def run_lgb(train_X, train_y, dev_X, dev_y, composer_list, headers):
    print("\n")
    print("Gradient boosting with decision stumps:")
    lgbclf = lgb.LGBMClassifier(n_estimators=n_estimators, application=application, lambda_l2=lambda_l2, bagging_fraction=bagging_fraction,
                                bagging_freq=bagging_freq, num_class=len(np.unique(train_y)))
    lgbclf.fit(train_X, train_y)
    y_pred = lgbclf.predict(train_X)
    accuracy = accuracy_score(train_y, y_pred)
    print("Train accuracy is: ", accuracy)
    confusion_m = confusion_matrix(train_y, y_pred)
    save_confusion_m(confusion_m, composer_list, "train_gbdt_{}_learners".format(n_estimators), accuracy)
    
    y_pred = lgbclf.predict(dev_X)
    accuracy = accuracy_score(dev_y, y_pred)
    print("Dev accuracy is: ", accuracy)
    confusion_m = confusion_matrix(dev_y, y_pred)
    save_confusion_m(confusion_m, composer_list, "dev_gbdt_{}_learners".format(n_estimators), accuracy)

    print("Metrics on dev set: ")
    metrics = print_additional_metrics(dev_y, y_pred, composer_list)
    plot_feature_importance(lgbclf.feature_importances_, headers, min(50, train_X.shape[1]), lgbclf.importance_type)

    #retrain to record gain
    lgbclf = lgb.LGBMClassifier(n_estimators=n_estimators, application=application, lambda_l2=lambda_l2, bagging_fraction=bagging_fraction,
                                bagging_freq=bagging_freq, num_class=len(np.unique(train_y)), importance_type='gain')
    lgbclf.fit(train_X, train_y)
    plot_feature_importance(lgbclf.feature_importances_, headers, min(50, train_X.shape[1]), lgbclf.importance_type)
    
def main():
    #the most general experiment
    train_X, train_y, dev_X, dev_y, headers = read_and_clean()
    composer_list = get_y_labels()

    #grouped composers, subset of features
    #train_X, train_y, dev_X, dev_y, headers = read_and_clean(group_composers=True, selected_features='rhythm')
    #composer_list = ['baroque', 'classical', 'romantic']

    #un-grouped composers, subset of features
    #train_X, train_y, dev_X, dev_y, headers = read_and_clean(selected_features='melodic')
    #composer_list = get_y_labels()
    
    run_lgb(train_X, train_y, dev_X, dev_y, composer_list, headers) 
if __name__ == "__main__":
    main()
