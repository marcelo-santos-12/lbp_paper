from exec_run import run
import time

if __name__ == '__main__':

    ################ defining useful parameters #######################################
    DATASET = 'SAMPLE_10_DATASET_GASTRIC_256'
    size_train = .8

    t0 = time.time()

    print('Inicio dos experimentos...')
    for VARIANT in ['base_lbp', 'completed_lbp', 'extended_lbp']:
        for METHOD in ['nri_uniform','uniform']:
 
            for P,R in [(8, 1), (16, 2), (24, 3)]:
                # inicia treinamento
                run(dataset=DATASET, variant=VARIANT, method=METHOD, P=P, R=R, size_train_percent=size_train)

    print('Fim dos Experimentos')

    runtime = (time.time() - t0) / 60 / 60
    print('Tempo Total: {}h'.format(round(runtime, 2)))
