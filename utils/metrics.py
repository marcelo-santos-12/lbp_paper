'''
Modulo que implementa o GridSearch
'''
import numpy as np
import os
from itertools import product
from sklearn.metrics import (roc_curve, f1_score, auc,
                            accuracy_score, matthews_corrcoef,
                            confusion_matrix)

from sklearn.metrics._plot.base  import _get_response

from sklearn.base import clone
from sklearn.neural_network import MLPClassifier


class MyGridSearch():

    def __init__(self, classifier, grid_parameters):
        self.classifier = classifier
        self.grid_parameters = grid_parameters
        self.all_parameter_comb = self._get_all_parameter_comb()

    def fit(self, x_train, y_train, x_test, y_test):
        self.results = {}
        self.results['best_matthews'] = -1

        for (i, parameters_i) in enumerate(self.all_parameter_comb):

            if isinstance(self.classifier, MLPClassifier) and len(parameters_i)==5: # gambiarra
                parameters_i[0] = (parameters_i[0], parameters_i[1])
                parameters_i.pop(1)

            parameters_to_clf = dict(zip(self.grid_parameters.keys(), parameters_i))

            clf_i = clone(self.classifier)

            clf_i.set_params(**parameters_to_clf)

            clf_i = clf_i.fit(X=x_train, y=y_train)

            y_pred = clf_i.predict(x_test)
            
            matthews = matthews_corrcoef(y_test, y_pred)

            if self.results['best_matthews'] < matthews:

                y_pred_roc, _ = _get_response(x_test, clf_i, 'auto', pos_label=None)
                fpr, tpr, _ = roc_curve(y_test, y_pred_roc, pos_label=None, sample_weight=None, drop_intermediate=True)

                self.results['best_matthews'] = matthews
                self.results['f1score'] = f1_score(y_test, y_pred)
                self.results['auc'] = auc(fpr, tpr)
                self.results['accuracy'] = accuracy_score(y_test, y_pred)
                self.results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
                self.results['best_parameter'] = dict(zip(self.grid_parameters.keys(), parameters_i))
                self.results['best_clf'] = clf_i
        return self.results

    def _get_all_parameter_comb(self,):
        list_comb = []
        for i, k in enumerate(self.grid_parameters.keys()):
            if i == 0:
                list_comb = self.grid_parameters[k]
                continue
            list_comb = [(x,y) for x, y in product(list_comb, self.grid_parameters[k])]

        def formated(tuple_arr):

            def just_list(_list):
                if isinstance(_list, (list, tuple)):
                    return [sub_elem for elem in _list for sub_elem in just_list(elem)]
                else:
                    return [_list]

            _list_formated = []
            for _list in tuple_arr:
                _list_formated.extend(just_list(_list))
            return _list_formated

        return list(map(formated, list_comb))


if __name__ == '__main__':

    
    mlp_parameters = {
        'hidden_layer_sizes': [(5, 5), (10, 10), (20, 20)],
        'solver': ['adam', 'sgd'],
        'activation': ['relu', 'identity', 'logistic', 'tanh'],
        'max_iter': [50, 100, 200]
    }

    gs = MyGridSearch(classifier=(MLPClassifier()), grid_parameters=mlp_parameters)
