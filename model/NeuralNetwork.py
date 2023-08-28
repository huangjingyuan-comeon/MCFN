import torch
import torch.nn as nn
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
import copy

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)



class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self):
        raise NotImplementedError

    def fit(self,user_model,X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred,
            X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev,X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid,y_test,num):

        if torch.cuda.is_available():
            self.cuda()

        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['reg'], amsgrad=True) #
        # self.optimizer = torch.optim.Adadelta(self.parameters(), weight_decay=self.config['reg'])

        X_train_source_wid = torch.LongTensor(X_train_source_wid)
        X_train_source_id = torch.LongTensor(X_train_source_id)
        X_train_user_id = torch.LongTensor(X_train_user_id)
        X_train_ruid = torch.LongTensor(X_train_ruid)
        y_train = torch.LongTensor(y_train)
        y_train_cred = torch.LongTensor(y_train_cred)
        y_train_rucred = torch.LongTensor(y_train_rucred)

        dataset = TensorDataset(X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


        loss_func = nn.CrossEntropyLoss()


        for epoch in range(1, self.config['epochs'] + 1):


            self.train()
            avg_loss = 0
            avg_acc = 0

            for i, data in enumerate(dataloader):

                meta_model.load_state_dict(self.state_dict())
                meta_W = user_model

                with torch.no_grad():
                    X_source_wid, X_source_id, X_user_id, X_ruid, batch_y, batch_y_cred, batch_y_rucred = (
                        item.cuda(device=self.device) for item in data)
                    num = int(len(X_source_id) * 0.5)
                    support_source_wid, support_source_id, support_user_id, support_ruid, support_y = X_source_wid[:num], X_source_id[:num], X_user_id[:num], X_ruid[:num], batch_y[:num]
                    query_source_wid, query_source_id, query_user_id, query_ruid, query_y = X_source_wid[num:], X_source_id[ num:], X_user_id[num:], X_ruid[ num:], batch_y[num:]

                inneroutput = meta_model.forward(meta_W, support_source_wid, support_source_id, support_user_id,
                                                 support_ruid)
                innerloss = loss_func(inneroutput, support_y)

                grads = torch.autograd.grad(innerloss, meta_model.parameters(), retain_graph=True)



                for grad, (n, p) in zip(grads, meta_model.named_parameters()):

                    if n in stop_dict:
                        continue
                    else:
                        self._parameters[n] = p - config['lr'] * grad

                outeroutput = self.forward(user_model,
                                           query_source_wid,
                                           query_source_id,
                                           query_user_id,
                                           query_ruid)

                outerloss = loss_func(outeroutput, query_y)



                u_grad = torch.autograd.grad(outerloss, user_model, retain_graph=True, allow_unused=True)[0]

                user_model = user_model - u_grad * config['lr']

                torch.autograd.grad(innerloss, meta_model.parameters(), only_inputs=True, retain_graph=False,
                                    create_graph=False)
                torch.cuda.empty_cache()

                loss = outerloss

                loss.backward()



                self.optimizer.step()

                logit = torch.cat([inneroutput, outeroutput], dim=0)

                corrects = (torch.max(logit, 1)[1].view(batch_y.size()).data == batch_y.data).sum()
                accuracy = 100 * corrects / len(batch_y)
                print('Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(i, loss.item() + innerloss.item(), accuracy, corrects, batch_y.size(0)))

                avg_loss += loss.item() + innerloss.item()
                avg_acc += accuracy

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

            cnt = y_train.size(0) // batch_size + 1
            print("Average loss:{:.6f} average acc:{:.6f}%".format(avg_loss/cnt, avg_acc/cnt))
            if epoch > self.config['epochs'] // 2 and self.patience > 2:  #
                print("Reload the best model...")
                self.load_state_dict(torch.load(self.config['save_path']))
                now_lr = self.adjust_learning_rate(self.optimizer)
                print(now_lr)
                self.patience = 0

            self.evaluate(user_model, X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, epoch)



    def adjust_learning_rate(self, optimizer, decay_rate=.5):
        now_lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            now_lr = param_group['lr']
        return now_lr


    def evaluate(self, W,X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev,epoch):
        y_pred= self.predict(W,X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid)
        acc = accuracy_score(y_dev, y_pred)
        # print("Val set acc:", acc)
        # print("Best val set acc:", self.best_acc)

        if epoch >= self.config['epochs']//2 and acc > self.best_acc:  #
            self.best_acc = acc
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            torch.save(W, self.config['save_user_path'])
            # print(classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5))
            # print("save model!!!")
        else:
            self.patience += 1


    def predict(self,W ,X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid):
        if torch.cuda.is_available():
            self.cuda()

        self.eval()
        y_pred = []
        X_dev_source_wid = torch.LongTensor(X_dev_source_wid)
        X_dev_source_id = torch.LongTensor(X_dev_source_id)
        X_dev_user_id = torch.LongTensor(X_dev_user_id)
        X_dev_ruid = torch.LongTensor(X_dev_ruid)

        dataset = TensorDataset(X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid)
        dataloader = DataLoader(dataset, batch_size=32)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                X_source_wid, X_source_id, X_user_id, \
                X_ruid = (item.cuda(device=self.device) for item in data)

            logits,_= self.forward(W,X_source_wid, X_source_id, X_user_id, X_ruid)
            predicted = torch.max(logits, dim=1)[1]
            y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred
