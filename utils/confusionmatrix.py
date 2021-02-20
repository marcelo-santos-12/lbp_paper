import pandas as pd
import numpy as np

def matstr_arr(confusion_str):

    mat_str = confusion_str.replace('[', '').replace(']', '').replace('\n', '').split(' ')

    mat_str.remove('')
    mat_str.remove('')

    mat_str = np.array(list(map(int, mat_str))).reshape(2, 2)

    return mat_str

def main():

    mat = pd.read_csv('results.csv')['confusion_matrix']

    all_confusion = list(map(matstr_arr, mat))

    print (all_confusion)

if __name__ == '__main__':

    main()
