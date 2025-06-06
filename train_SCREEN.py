import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from EC_contrastive import *
import time
import torch.nn as nn
from random import choice
from protein_init import *


class EarlyStopping:
    def __init__(self, patience=7, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_aupr = None
        self.aupr_max = 0
        self.early_stop = False

    def __call__(self, result_AUPRC, model):

        if self.best_val_aupr is None:
            self.best_val_aupr = result_AUPRC
            self.save_checkpoint(result_AUPRC, model)

        elif result_AUPRC < self.best_val_aupr:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_aupr = result_AUPRC
            self.save_checkpoint(result_AUPRC, model)
            self.counter = 0
        return

    def save_checkpoint(self, AUPRC, model):
        if self.verbose:
            print(f'Validation auprc increased ({self.aupr_max:.4f} --> {AUPRC:.4f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join("./Model/", 'SCREEN_EC_contrastive.pkl' ))
        self.aupr_max= AUPRC


def train_one_epoch(model, data_loader, epoch, class_center, data_path):
    epoch_loss_train = 0.0
    n = 0
    update_ecfeas = {}
    for i, data in enumerate(data_loader):
        model.optimizer.zero_grad()
        sequence_name, sequence, labels, EC1, graphs, evo_feature = data

        EC1 = int(EC1.cpu().detach().numpy()[0])
        ecnums = [1,2,3,4,5,6,7]
        positive_out = class_center[EC1]

        ecnums.remove(EC1)
        EC_index = choice(ecnums)
        negative_out = class_center[EC_index]

        if torch.cuda.is_available():

            graphs = Variable(graphs.cuda())
            evo_feature = Variable(evo_feature.cuda())
            y_true = Variable(labels.cuda())
            positive_out = Variable(positive_out.cuda())
            negative_out = Variable(negative_out.cuda())
        else:

            graphs = Variable(graphs)
            evo_feature = Variable(evo_feature)
            y_true = Variable(labels)
            positive_out = Variable(positive_out)
            negative_out = Variable(negative_out)

        graphs = torch.squeeze(graphs)
        evo_feature = torch.squeeze(evo_feature)
        y_true = torch.squeeze(y_true)

        y_pred, ec_fea = model(graphs, evo_feature)
        #print(ec_fea.size(),positive_out.size(),negative_out.size())
        #ec_fea = torch.unsqueeze(ec_fea, 0)
        update_ecfeas[sequence[0]] = ec_fea

        funsite_loss = model.criterion(y_pred, y_true)
        triplet_loss = nn.functional.triplet_margin_loss(ec_fea, positive_out, negative_out,reduction='mean')

        T = 200
        w_m = 1.0 if epoch > T else epoch/T
        w_a = 0.0 if epoch > T else (1-epoch/T)
        loss = w_m * funsite_loss + w_a * triplet_loss

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        epoch_loss_train += funsite_loss.item()
        n += 1

    epoch_loss_train_avg = epoch_loss_train / n

    if (epoch+1) % 10 == 0:
        print("this epoch is ", epoch)
        f_save = open(data_path + 'updated_enzfeas/dict_enzfeas.pkl', 'wb')
        pickle.dump(update_ecfeas, f_save)
        f_save.close()
    return epoch_loss_train_avg


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, labels,_, graphs, evo_feature = data

            if torch.cuda.is_available():
                graphs = Variable(graphs.cuda())
                evo_feature = Variable(evo_feature.cuda())
                y_true = Variable(labels.cuda())
            else:
                graphs = Variable(graphs)
                evo_feature = Variable(evo_feature)
                y_true = Variable(labels)

            graphs = torch.squeeze(graphs)
            evo_feature = torch.squeeze(evo_feature)
            y_true = torch.squeeze(y_true)
            y_pred,_ = model(graphs, evo_feature)#
            loss = model.criterion(y_pred, y_true)

            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred/8)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n
    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred,best_threshold=None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def train_model(train_dataframe, valid_dataframe, data_path, ec2pro, protein_dict):

    print("Train on", str(train_dataframe.shape[0]), "samples, validate on", str(valid_dataframe.shape[0]), "samples")
    train_loader = DataLoader(dataset=ProDataset(train_dataframe, protein_dict), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe, protein_dict), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = SCREEN(NLAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT)
    if torch.cuda.is_available():
        model.cuda()

    early_stopping = EarlyStopping(patience=7, verbose=True)

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()
        epoch_time = time.time()

        if epoch % 10==0:
            print("========== update enzfeas ==========")
            class_center = get_cluster_center(ec2pro, data_path, epoch)

        epoch_loss_train_avg = train_one_epoch(model, train_loader, epoch, class_center, data_path)
        print("Epoch: {} cost time: {}, train_loss: {}".format(epoch + 1, time.time() - epoch_time, epoch_loss_train_avg))

        if epoch>200:
            print("========== Evaluate Valid set ==========")
            epoch_loss_valid_avg, valid_true, valid_pred, _ = evaluate(model, valid_loader)
            result_valid = analysis(valid_true, valid_pred)

            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} ".format(epoch + 1, epoch_loss_train_avg,  epoch_loss_valid_avg))
            print("Valid binary acc: ", result_valid['binary_acc'])
            print("Valid precision: ", result_valid['precision'])
            print("Valid recall: ", result_valid['recall'])
            print("Valid f1: ", result_valid['f1'])
            print("Valid AUC: ", result_valid['AUC'])
            print("Valid AUPRC: ", result_valid['AUPRC'])
            print("Valid mcc: ", result_valid['mcc'])

            early_stopping(result_valid['AUPRC'], model)
            if early_stopping.early_stop:
                print("Early stopping")
                break


def main():
    #loading enzyme IDs with the first-level EC number annotation for training
    IDs, sequences, labels, EC1 = [], [], [], []
    data_path = "./Dataset/training_data/"
    with open(data_path + "training_id_withEC.txt", "r") as f:
        IDs = [line.strip()[:4].lower() + "-" + line.strip()[5].upper() for line in f]

    # loading enzyem sequences and lables according to IDs
    prot_seq = {}
    with open(data_path + "training_label.txt", "r") as f:
        data = f.readlines()

    for i in range(0, len(data), 3):
        if not data[i].startswith('>'):
            continue
        protein = data[i][1:].strip()
        parts = protein.replace('_', '-').split('-')
        if len(parts) != 2:
            continue
        pdb_id = f"{parts[0].lower()}-{parts[1].upper()}"

        if pdb_id not in IDs or pdb_id in prot_seq.keys():
            continue

        seq = data[i + 1].strip()[:1000]
        label = data[i + 2].strip()[:1000]

        prot_seq[pdb_id] = seq
        sequences.append(seq)
        labels.append([int(x) for x in label])

    #loading the first level enzyme function (EC1)
    PDB2EC1 = np.load(data_path + 'training_PDB2EC1.npy', allow_pickle=True).item()
    ec2pro = {}
    for prot in IDs:
        EC_index = get_EC1_num(PDB2EC1[prot])
        EC1.append(EC_index)

        if EC_index in ec2pro.keys():
            ec2pro[EC_index].add(prot_seq[prot])
        else:
            ec2pro[EC_index] = set()
            ec2pro[EC_index].add(prot_seq[prot])

    data_dic = {"ID":IDs, "sequence":sequences, "label":labels, "EC1":EC1}
    data_dataframe = pd.DataFrame(data_dic)

    print('Extracting enzyme features')
    Enz_sequence = [data_dataframe['sequence'][ind] for ind in data_dataframe.index]
    #protein_dict = protein_init(Enz_sequence)
    #torch.save(protein_dict, 'protein_dict.pt')
    protein_dict = torch.load('protein_dict.pt')

    train_dataframe = data_dataframe.iloc[:int(len(IDs) * 0.9)]
    valid_dataframe = data_dataframe.iloc[int(len(IDs) * 0.9):]
    train_model(train_dataframe, valid_dataframe, data_path, ec2pro, protein_dict)


if __name__ == "__main__":
    main()









