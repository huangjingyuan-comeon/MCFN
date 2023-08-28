import os
import pickle
import warnings

from model.Mymodel import *
from model.NeuralNetwork import *

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_dataset(task):
    print("task: ", task)

    A_us, A_uu,A_uu0,A_uu1,A_uu2,A_uu3,A_up,A_pu,A_up_NR,A_up_FR,A_up_TR,A_up_UR,g_up= pickle.load(open("dataset/"+task+"/relations.pkl", 'rb'))
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
    config['A_up_TR'] = A_up_TR
    config['A_up_UR'] = A_up_UR
    config['g_up'] = g_up

    return X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, \
           X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, \
           X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test



def train_and_test(model, task,i):
    model_suffix = model.__name__.lower().strip("text")
    config['save_path'] = 'checkpoint/weights.best.' + str(i) + task + "." + model_suffix
    config['save_user_path'] = 'checkpoint/weights.best.user.' + str(i) + task + "." + model_suffix
    config['load_user_path'] = 'checkpoint/weights.best.user.' + task + "." + model_suffix
    config['load_path'] = 'load2/weights.best.'+ task + ".sbag"

    X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, \
    X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, \
    X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test = load_dataset(task)


    nn = model(config)




    # nn.fit(user_model,X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred,
    #        X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev,i)  #


    nn.load_state_dict(torch.load('t16/weights.best'))
    W = torch.load('t16/weights.best.user')
    y_pred = nn.predict(W,X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid)
    print(classification_report(y_test, y_pred, target_names=config['target_names'], digits=3))







config = {
    'lr':0.001,
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
    task = 'twitter16'
    # task = 'weibo'
    i = 1
    if i == 0:
        for i in range(20):
            model = MCFN
            train_and_test(model, task,i)
    else:
        model = MCFN
        train_and_test(model, task, i)


# Twitter15
#               precision    recall  f1-score   support
#
#           NR      0.865     0.988     0.922        84
#           FR      0.975     0.917     0.945        84
#           TR      0.938     0.893     0.915        84
#           UR      0.951     0.917     0.933        84
#
#     accuracy                          0.929       336
#    macro avg      0.932     0.929     0.929       336
# weighted avg      0.932     0.929     0.929       336



# Twitter16
#               precision    recall  f1-score   support
#
#           NR      0.936     0.957     0.946        46
#           FR      0.976     0.870     0.920        46
#           TR      0.857     0.933     0.894        45
#           UR      0.979     0.979     0.979        47
#
#     accuracy                          0.935       184
#    macro avg      0.937     0.935     0.935       184
# weighted avg      0.938     0.935     0.935       184


# Weibo  head=7
#               precision    recall  f1-score   support
#
#           NR      0.967     0.936     0.951       529
#           FR      0.937     0.967     0.952       521
#
#     accuracy                          0.951      1050
#    macro avg      0.952     0.952     0.951      1050
# weighted avg      0.952     0.951     0.951      1050

