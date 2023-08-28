import os
import pdb
import pickle
import warnings
import numpy as np
import pandas as pd
# from tsne import bh_sne
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

from model.Mymodel import *
from model.NeuralNetwork import *
import copy, sys

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_dataset(task):
    print("task: ", task)

    A_us, A_uu,A_uu0,A_uu1,A_up,A_pu,A_up_NR,A_up_FR,g_up= pickle.load(open("dataset/"+task+"/relations.pkl", 'rb'))
    X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, word_embeddings = pickle.load(open("dataset/"+task+"/train.pkl", 'rb'))
    X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev = pickle.load(open("dataset/"+task+"/dev.pkl", 'rb'))
    X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test = pickle.load(open("dataset/"+task+"/test.pkl", 'rb'))
    config['maxlen'] = len(X_train_source_wid[0])
    if task == 'twitter15':
        config['n_heads'] = 10
    elif task == 'twitter16':
        config['n_heads'] = 8
    else:
        config['n_heads'] = 7
        config['batch_size'] = 128
        config['num_classes'] = 2
        config['target_names'] = ['NR', 'FR']
    # print(config)

    config['embedding_weights'] = word_embeddings
    config['A_up'] = A_up
    config['A_pu'] = A_pu
    config['A_up_NR'] = A_up_NR
    config['A_up_FR'] = A_up_FR
    config['g_up'] = g_up

    return X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, \
           X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, \
           X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test

def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)

    outputs = outputs.cpu().detach().numpy()
    tsne = TSNE(random_state=0,n_iter=1000,perplexity=15)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    flatui = ["#0000FF", "#90EE90", "#FF0000"]

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette(flatui),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir,'tsne.png'), bbox_inches='tight')
    print('done!')

def train_and_test(model, task,i):
    model_suffix = model.__name__.lower().strip("text")
    config['save_path'] = 'checkpoint/weights.best.' + str(i) + task + "." + model_suffix
    config['save_user_path'] = 'checkpoint/weights.best.user.' + str(i) + task + "." + model_suffix
    config['load_user_path'] = 'checkpoint/weights.best.user.' + task + "." + model_suffix
    config['load_path'] = 'load/weights.best.'+ task + "." + model_suffix

    X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, \
    X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, \
    X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test = load_dataset(task)

    nn = model(config)



    # nn.fit(user_model,X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred,
    #        X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev,X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid,y_test,i)  #


    nn.load_state_dict(torch.load('weibo/weights.best'))

    W = torch.load('weibo/weights.best.user')
    y_pred = nn.predict(W,X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid)
    print(classification_report(y_test, y_pred, target_names=config['target_names'], digits=3))





config = {
    'lr':0.0005,
    'reg':1e-6,
    'embeding_size': 100,
    'batch_size':16,
    'nb_filters':100,
    'kernel_sizes':[3, 4, 5],
    'dropout':0.5,
    'epochs':20,
    'num_classes':4,
    'target_names':['NR', 'FR', 'TR', 'UR']
}


if __name__ == '__main__':
    # task = 'twitter15'
    # task = 'twitter16'
    task = 'weibo'
    i = 1
    if i == 0:
        for i in range(300):
            model = MCFN_weibo
            train_and_test(model, task,i)
    else:
        model = MCFN_weibo
        train_and_test(model, task, i)

