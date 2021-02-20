import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn')

def plot_facetgrid(output, measure = ['best_matthews', 'accuracy', 'fscore', 'auc_roc']):

    df_res = pd.read_csv(output + '/results.csv')
    df_res.pop('n_features')

    variants = df_res['variant'].unique()
    
    for variant in variants:

        variant_res = df_res[df_res['variant'] == variant]

        fig, axs = plt.subplots(2, 3)

        fig.suptitle('Results of ' + variant)

        for i, method in enumerate(df_res['method'].unique()):
            for j, clf in enumerate(df_res['classifier'].unique()):
                splot = axs[i, j]

                height = variant_res[variant_res['method'] == method][variant_res['classifier'] == clf]

                height.plot(kind='bar', x='(P, R)', y=measure, rot=0, ax=splot, legend='')

                splot.set_xlabel('')
                
                splot.set_ylim([0, 100])
                
                if i == 0 and j == 0 or j == 0 and i == 1:
                    splot.set_ylabel(method)
                if i == 1:
                    splot.set_xlabel(clf)

        plt.legend(['MCC', 'Acuracy', 'Fscore', 'AUC'], bbox_to_anchor=(0., 2.22, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)

        plt.savefig(output+'/'+variant + '_facetgrid.png', dpi=200)
        
if __name__ == '__main__':

    main()
