import argparse
import csv
import time
from numpy.core.defchararray import asarray
import torch
from torch._C import device
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from cmn.utils import *
import dal.load_dblp_data as dblp
from torch.utils.data import Dataset
from torchsummary import summary
import eval.evaluator as dblp_eval

from loss import listwise_penalty, collaboration_score



class TFDataset(Dataset):
    def __init__(self, fold_id, trainSet=True, embedding_dim = 300):
        self.embedding_dim = embedding_dim
        if dblp.ae_data_exist(file_path='/media/Coherent/dataset/Expert_Skill_dataset.pkl'.format(self.embedding_dim)):
            dataset = dblp.load_ae_dataset(file_path='/media/Coherent/dataset/Expert_Skill_dataset.pkl'.format(self.embedding_dim))
        train_test_indices = dblp.load_train_test_indices(file_path='/media/Coherent/dataset/Train_Test.pkl')
        x_train, y_train, x_test, y_test = dblp.get_fold_data(fold_id, dataset, train_test_indices)

        if trainSet:
            self.x = x_train
            self.y = y_train
        else:
            self.x = x_test
            self.y = y_test

    def __len__(self):
        return len(self.x)

    def input_dim(self):
        return len(self.x[0])

    def output_dim(self):
        return len(self.y[0])

    def __getitem__(self, idx):
        skills = self.x[idx]
        experts = self.y[idx]
        sample = {'skill': skills, 'expert': experts}
        return sample


class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.latent_dim = D_in
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.softmax(self.linear2(x))


class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.latent_dim = decoder.latent_dim
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(100, self.latent_dim)
        self._enc_log_sigma = torch.nn.Linear(100, self.latent_dim)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(device)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-method_name', type=str, default="coherent",)

    args = parser.parse_args()
    # method_name = 'torchSVAEO_listwise'
    latent_dim = 20
    hidden_dim = 300
    batch_size = 64
    epochs = 150
    k_fold = 10
    k_max = 100

    list_loss_thr = 5
    LISTWISE = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    
    result_output_name = "./output/predictions/{}_output.csv".format(args.method_name)
    with open(result_output_name, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Method Name', '# Total Folds', '# Fold Number', '# Predictions', '# Truth', 'Computation Time (ms)',
            'Prediction Indices', 'True Indices'])

    team_member_dict_fp = "/media/Coherent/dataset/team_members_dict.json"
    with open(team_member_dict_fp, 'r') as file_to_read:
        team_member_dict = json.load(file_to_read)

    for fold_counter in range(1, k_fold+1):

        trainData = TFDataset(fold_counter, trainSet=True)
        testData = TFDataset(fold_counter, trainSet=False)

        input_dim = trainData.input_dim()
        output_dim = trainData.output_dim()

        DL_trainDS = DataLoader(trainData, batch_size=batch_size, shuffle=False)
        DL_testDS = DataLoader(testData, batch_size=batch_size, shuffle=False)

        print('Number of samples: ', len(trainData))

        encoder = Encoder(input_dim, 100, 100)
        decoder = Decoder(latent_dim, 100, output_dim)
        encoder.to(device)
        decoder.to(device)
        vae = VAE(encoder, decoder)
        vae.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(vae.parameters(), lr=1e-4)

        for epoch in range(epochs):
            train_loss = 0
            test_loss = 0
            for i, data in enumerate(DL_trainDS):
                inputs = data['skill'].to(device)
                classes = data['expert'].to(device)
                optimizer.zero_grad()
                dec = vae(inputs)
                ll = latent_loss(vae.z_mean, vae.z_sigma)
                if LISTWISE:
                    # print('--------------------listwise training--------------------')
                    loss = criterion(dec, classes.to(torch.float)) \
                           + ll \
                           - collaboration_score(dec, team_member_dict, list_loss_thr)
                else:
                    # print('--------------------regular training--------------------')
                    loss = criterion(dec, classes.to(torch.float)) \
                           + ll
                loss.backward()
                optimizer.step()
                train_loss = +loss.item()
            
            for i, data in enumerate(DL_testDS):
                inputs = data['skill'].to(device)
                classes = data['expert'].to(device)
                dec = vae(inputs)
                ll = latent_loss(vae.z_mean, vae.z_sigma)
                if LISTWISE:
                    loss = criterion(dec, classes.to(torch.float)) \
                           + ll \
                           - collaboration_score(dec, team_member_dict, list_loss_thr)
                else:
                    loss = criterion(dec, classes.to(torch.float)) + ll
                test_loss = +loss.item()        
            
            print('\n EPOCH {}/{} \t train loss {:.5f} \t val loss {:.5f}'.format(epoch + 1, epochs, train_loss, test_loss))

        print("eval on test data fold #{}".format(fold_counter))
        true_indices = []
        pred_indices = []
        with open(result_output_name, 'a+') as file:
            writer = csv.writer(file)
            for item in DL_testDS:
                for sample_x, sample_y in zip(item['skill'].to(device), item['expert']):
                    start_time = time.time()
                    sample_prediction = vae(sample_x)
                    end_time = time.time()
                    elapsed_time = (end_time - start_time)*1000
                    sample_prediction = sample_prediction.cpu().detach().numpy()
                    pred_index, true_index = dblp_eval.find_indices([sample_prediction], [sample_y])
                    true_indices.append(true_index[0])
                    pred_indices.append(pred_index[0])
                    writer.writerow([args.method_name, k_fold, fold_counter, len(pred_index[0][:k_max]), len(true_index[0]),
                                    elapsed_time] + pred_index[0][:k_max] + true_index[0])
