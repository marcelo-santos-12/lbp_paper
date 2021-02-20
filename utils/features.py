'''
Modulo que contem a funcao de calculo dos descritores do dataset, cujo retorno
sera os arrays contendo os dados escolhido para treinamento e teste
'''
import cv2
import numpy as np
from tqdm import tqdm
import glob
from sklearn.preprocessing import normalize
import os

def compute_features(path_dataset, descriptor, P, R, method, size_train, norm='l2'):
    '''
    Calcula os descritores de todas as imagens inseridas no diretorio do dataset

    Parameters
    ----------
        path_dataset: (str)
            Diretorio que contem as imagens a serem classificadas
        
        descriptor: (function)
            Funcao que calcula os descritores das imagens

        P: (int)
            Numero de pontos dentro de uma vizinhanca

        R:  (int)
            Raio de localizacao dos pontos da vizinhanca

        method: (string)
            Versao do LBP a ser calculada
    '''

    x_train, y_train, x_test, y_test = [], [], [], []
    for index, path_class in enumerate(glob.glob(path_dataset + '/*')):
        print('Classe {}: indice {}...'.format(path_class, index))

        dir_iter = glob.glob(path_class + '/*')
        length_dir = len(dir_iter)
        ind_max_train = int(size_train * length_dir)

        cont = 0

        ind_max_train = 80
    
        for i, name_img in enumerate(tqdm(dir_iter)):

            if cont == 100:
                break
            
            try:
                img = cv2.imread(name_img, cv2.IMREAD_GRAYSCALE)
            except:
                print('Não é possível ler arquivo:', name_img)
            
            # garante que as imagens terao as mesma dimensoes
            try:
                raise Exception(img.shape != (460, 700))

            except:
                img = cv2.resize(img, (460, 700))
            
            # calcula o array descritor da imagem atual
            feature = descriptor(img, P=P, R=R, block=(1, 1), method=method)

            feature = normalize(feature.reshape(1, -1), norm='l1').reshape(-1,)

            # adiciona ao conjunto de treinamento
            if i < ind_max_train:
                x_train.append(list(feature))
                y_train.append(index)

            # adiciona ao conjunto de teste
            else:
                x_test.append(list(feature))
                y_test.append(index)

            cont += 1

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)
