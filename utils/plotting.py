'''
Modulo que implementa as funcoes de plot da curva roc
'''
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import plot_roc_curve

plt.style.use('ggplot')


def plot_results(_id, best_clf, x_test, y_test, method, variant, P, R):

    if not os.path.exists('ARR_ROC/'+variant+'/y_pred'):
        os.makedirs('ARR_ROC/'+variant+'/y_pred')
    
    if not os.path.exists('ARR_ROC/'+variant+'/y_pred_roc'):
        os.makedirs('ARR_ROC/'+variant+'/y_pred_roc')
    
    from sklearn.metrics._plot.base  import _get_response
    y_pred_roc, _ = _get_response(x_test, best_clf, 'auto', pos_label=None)
    
    arr_roc = 'ARR_ROC/{}/y_pred_roc/{}_{}_{}_{}.txt'.format(variant, method, _id.replace(' ', ''), str(P), str(R))
    np.savetxt(arr_roc, y_pred_roc)

    y_predict = best_clf.predict(x_test)
    arr_roc = 'ARR_ROC/{}/y_pred/{}_{}_{}_{}.txt'.format(variant, method, _id.replace(' ', ''), str(P), str(R))
    np.savetxt(arr_roc, y_predict)

    plot_roc_curve(best_clf, x_test, y_test)
    
    #PLOTANDO LINHA DIAGONAL --> y = x
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='g', alpha=.8)

    # PLOTANDO INFORMACOES BASICA DO GRAFICO
    plt.title('{} - {}/{} - P: {}, R:{}'.format(_id, method, variant, P, R))
    plt.legend(loc="lower right")
    
    if not os.path.exists('ROC_'+variant):
        os.makedirs('ROC_'+variant)

    plt.savefig('ROC_{}/{}_{}_{}_{}_{}.png'.format(variant, variant, method, _id.replace(' ', ''), P, R))
    plt.close()