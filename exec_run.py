import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# meus modulos
from lbp_module.texture import base_lbp, improved_lbp, hamming_lbp, completed_lbp, extended_lbp
from utils.features import compute_features
from utils.plotting import plot_results
from utils.metrics import MyGridSearch

# Para referencias aos descritores
ALGORITHM = {
    'base_lbp': base_lbp,
    'improved_lbp': improved_lbp,
    'extended_lbp': extended_lbp,
    'completed_lbp': completed_lbp,
    'hamming_lbp': hamming_lbp
}

# Para titulos dos graficos
VARIANTS = {
    'base_lbp': 'ORIGINAL LBP',
    'improved_lbp': 'IMPROVED LBP',
    'extended_lbp': 'EXTENDED LBP',
    'completed_lbp': 'COMPLETED LPB',
    'hamming_lbp': 'HAMMING LBP'
}

def run(dataset, variant, method, P, R, size_train_percent, save_descriptors=True):
    ################ computing image descriptors #######################################
    print('Experimento: Descritor= {}, P = {}, R = {}'.format(VARIANTS[variant], P, R))

    path_cur = os.path.join('descriptors', variant, method + '_' + str(P) +'_' +str(R)) 
    if save_descriptors:
        print('Computando recursos...')
        x_train, y_train, x_test, y_test = compute_features(path_dataset=dataset, \
                        descriptor=ALGORITHM[variant], P=P, R=R, method=method, size_train=size_train_percent)
        if not os.path.exists(path_cur):
            os.makedirs(path_cur)

        all_descriptors_train = np.concatenate([x_train, y_train.reshape(-1, 1)], axis=1)
        all_descriptors_test = np.concatenate([x_test, y_test.reshape(-1, 1)], axis=1)
        all_descriptors = np.concatenate([all_descriptors_train, all_descriptors_test], axis=0)
        np.savetxt(path_cur + '/data.txt', all_descriptors)

    else:
        print('Carregando recursos...')
        if os.path.exists(path_cur):
            data = np.loadtxt(path_cur + '/data.txt')
            print(data.shape)
        else:
            print("Descritores não computados")
            quit()

    if not os.path.exists('ARR_ROC'):
        os.makedirs('ARR_ROC')
    
    arr_file = 'ARR_ROC/' + '{}_y_true_{}_{}_{}.txt'.format(variant, method, P, R)
    np.savetxt(arr_file, y_test)
    
    n_features = x_train[0].shape[0]
    print('Comprimento do vetor de recursos: ', n_features)

    if os.path.exists('results.csv'):
        df = pd.read_csv('results.csv')
    else:
        columns = ['classifier', 'variant', 'method', '(P, R)', 'parameters', \
                   'best_matthews','fscore', 'accuracy', 'confusion_matrix','auc_roc', 'n_features']
        df = pd.DataFrame(columns=columns)

    ############### Defining classifiers and its parameters for GridSearch ############
    svm = SVC()
    mlp = MLPClassifier()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    knn = KNeighborsClassifier()

    svm_parameters = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [1, 10, 100],
        'gamma': [0.0001, 0.00001, 0.000001]
    }
    mlp_parameters = {
        'hidden_layer_sizes': [(10, 10), (50, 50), (100, 100)],
        'solver': ['adam', 'sgd'],
        'activation': ['relu', 'identity', 'logistic', 'tanh'],
        'max_iter': [50, 100, 200],
    }
    dt_parameters = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [3, 5, 10, 50],
    }
    rf_parameters = {
        'n_estimators': [5, 11, 51, 101],
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 50, 100, 200],
    }
    knn_parameters = {
        'n_neighbors': [1, 5, 9],
        'weights' : ['uniform', 'distance'],
        'algorithm': ['kd_tree', 'ball_tree'],
        'p': [1, 2] # Manhatan and Euclidian distance, respectivity
    }

    classifiers = [['SVM', svm, svm_parameters], ['MLP', mlp, mlp_parameters], \
                ['Decision Trees', dt, dt_parameters], ['Random Forest', rf, rf_parameters], \
                ['K-Nearest Neighbor', knn, knn_parameters]]
    
    classifiers = [['Random Forest', rf, rf_parameters], ['SVM', svm, svm_parameters], ['MLP', mlp, mlp_parameters],]
    classifiers = [['SVM', svm, svm_parameters], ['MLP', mlp, mlp_parameters],]
    ################### Starting the train of models ##################################
    for _id, clf, parameters in classifiers:
        np.random.seed(10)
        res_curr = {}
        print(35 * ' * ')
        print('Classificando com {}...'.format(_id))

        # CROSS-VALIDATION HOLD OUT --> train: 0.8, test:0.2
        
        ################# Executing GridSearch ###################################

        clf_search = MyGridSearch(classifier=clf, grid_parameters=parameters,)
        
        res_search = clf_search.fit(x_train, y_train, x_test, y_test)

        ################### Computing Performance #################################
        # Compute ROC curve and area the curve
        plot_results(_id, res_search['best_clf'], x_test, y_test, method.upper(), VARIANTS[variant], P, R)

        # Get AUC, F1Score and Accuracy metrics
        print(35 * '# ')
        print('Melhor Parametro: ', res_search['best_parameter']) 
        auc_roc = res_search['auc']
        f1score = res_search['f1score']
        accuracy = res_search['accuracy']
        mathews = res_search['best_matthews']
        matrix = res_search['confusion_matrix']

        print('F1Score:' , f1score)
        print('Accuracy:', accuracy)
        print('AUC:', auc_roc)
        print('Mathews:', mathews)

        res_curr['classifier'] = _id
        res_curr['variant'] = VARIANTS[variant]
        res_curr['method'] = method
        res_curr['(P, R)'] = (P, R)
        res_curr['parameters'] = res_search['best_parameter']
        res_curr['best_matthews'] = np.round(mathews*100, 2)
        res_curr['fscore'] = np.round(f1score * 100, 2)
        res_curr['accuracy'] = np.round(accuracy * 100, 2)
        res_curr['confusion_matrix'] = matrix
        res_curr['auc_roc'] = np.round(auc_roc * 100, 2)
        res_curr['n_features'] = n_features

        df = df.append(res_curr, ignore_index=True)

    df.to_csv('results.csv', index=False)


if __name__ == '__main__':

    run(dataset='SAMPLE_10_DATASET_GASTRIC_256', variant='base_lbp', method='uniform', P=24, R=3, size_train_percent=.8)
