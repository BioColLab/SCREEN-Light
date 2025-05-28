import pandas as pd
from torch.autograd import Variable
from EC_contrastive import *
#from feature_extract import *
import argparse
from protein_init import *
Model_Path = "./Model/"

def read_fasta(filepath):
    sequences = {}
    with open(filepath, 'r') as file:
        current_id = ""
        current_seq = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:]  # remove ">"
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    return sequences

class EnzCatDataset(Dataset):
    def __init__(self, dataframe,protein_dict):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.enzfea = protein_dict

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]

        evo_feature = self.enzfea[sequence]['seq_feat']
        graph = self.enzfea[sequence]['contact_map'].to(torch.float32)
        return sequence_name, sequence, graph, evo_feature

    def __len__(self):
        return len(self.names)


def evaluate(model, data_loader):
    model.eval()
    every_valid_pred = []
    pred_dict = {}
    Enz_names = []
    Sequences = []
    binary_pred = []

    for data in data_loader:
        with torch.no_grad():
            sequence_names, sequence, graphs, evo_feature = data
            if torch.cuda.is_available():
                graphs = Variable(graphs.cuda())
                evo_feature = Variable(evo_feature.cuda())
            else:
                graphs = Variable(graphs)
                evo_feature = Variable(evo_feature)

            graphs = torch.squeeze(graphs)
            evo_feature = torch.squeeze(evo_feature)

            y_pred, _ = model(graphs, evo_feature)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred/8)
            y_pred = y_pred.cpu().detach().numpy()

            every_valid_pred.append([pred[1] for pred in y_pred])
            binary_pred.append( [1 if pred[1] >= 0.5 else 0 for pred in y_pred])
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]
            Sequences.append(sequence[0])
            Enz_names.append(sequence_names[0])

    return pred_dict,Sequences,binary_pred,Enz_names


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--fastafile', type=str, default='./Example/PDB_id.fa', help='file containing enzyme sequences')
    args = parser.parse_args()

    fasta_dict = read_fasta(args.fastafile)
    print('Extracting enzyme features')
    Enz_sequence = [sequence if len(sequence) < 1000 else sequence[:1000] for sequence in fasta_dict.values()]
    profea_dict = protein_init(Enz_sequence)

    print("starting to predict the catalytic residue")
    test_dic = {"ID": [pro for pro in fasta_dict.keys()], "sequence": Enz_sequence}
    test_dataframe = pd.DataFrame(test_dic)
    test_loader = DataLoader(dataset=EnzCatDataset(test_dataframe, profea_dict), batch_size=BATCH_SIZE, shuffle=True,num_workers=2)

    #load the pretrained SCREEN predictor
    model = SCREEN(NLAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(Model_Path + 'SCREEN_EC_contrastive.pkl', map_location='cuda:0'))
    pred_dict,sequences,binary_preds,enz_names = evaluate(model, test_loader)

    #print the predicted catalytic residue for every enzyme
    for i in range(len(enz_names)):
        PDB_id = enz_names[i]
        seq = sequences[i]
        print("For enzyme",PDB_id)
        for i in range(len(pred_dict[PDB_id])):
            if pred_dict[PDB_id][i]>0.5:
                print("The predicted catlytic residue:", seq[i] + str(i+1))

if __name__ == "__main__":
    main()



