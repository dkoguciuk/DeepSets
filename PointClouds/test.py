from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm, trange
import numpy as np

import classifier
import modelnet


#################### Settings ##############################
#num_epochs = 1000
batch_size = 4
downsample = 10    #For 5000 points use 2, for 1000 use 10, for 100 use 100
network_dim = 512  #For 5000 points use 512, for 1000 use 256, for 100 use 256
#num_repeats = 5    #Number of times to repeat the experiment
data_path = 'ModelNet40_cloud.h5'
model_path = 'model_1.pt'
#################### Settings ##############################


class PointCloudTrainer(object):
    def __init__(self):
        #Data loader
        self.model_fetcher = modelnet.ModelFetcher(data_path, batch_size, downsample, do_standardize=True, do_augmentation=True)

        #Setup network
        self.D = classifier.DTanh(network_dim, pool='max1').cuda()
        self.L = nn.CrossEntropyLoss().cuda()
        #self.optimizer = optim.Adam([{'params':self.D.parameters()}], lr=1e-3, weight_decay=1e-7, eps=1e-3)
        #self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(400,num_epochs,400)), gamma=0.1)
        #self.optimizer = optim.Adamax([{'params':self.D.parameters()}], lr=5e-4, weight_decay=1e-7, eps=1e-3) # optionally use this for 5000 points case, but adam with scheduler also works
        self.D.load_state_dict(torch.load(model_path))

#    def train(self):
#        self.D.train()
#        loss_val = float('inf')
#        for j in trange(num_epochs, desc="Epochs: "):
#            counts = 0
#            sum_acc = 0.0
#            train_data = self.model_fetcher.train_data(loss_val)
#            for x, _, y in train_data:
#                counts += len(y)
#                X = Variable(torch.cuda.FloatTensor(x))
#                Y = Variable(torch.cuda.LongTensor(y))
#                self.optimizer.zero_grad()
#                f_X = self.D(X)
#                loss = self.L(f_X, Y)
#                loss_val = loss.data.cpu().numpy()[0]
#                sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()[0]
#                train_data.set_description('Train loss: {0:.4f}'.format(loss_val))
#                loss.backward()
#                classifier.clip_grad(self.D, 5)
#                self.optimizer.step()
#                del X,Y,f_X,loss
#            train_acc = sum_acc/counts
#            self.scheduler.step()
#            if j%10==9:
#                tqdm.write('After epoch {0} Train Accuracy: {1:0.3f} '.format(j+1, train_acc))

    def test(self):
        self.D.eval()
        counts = 0
        sum_acc = 0.0
        total_time = 0.
        total_batches = 0.
        for x, _, y in self.model_fetcher.test_data():
            counts += len(y)
            X = Variable(torch.cuda.FloatTensor(x))
            Y = Variable(torch.cuda.LongTensor(y))
            start = timer()
            f_X = self.D(X)
            end = timer()
            if total_batches != 0:
                total_time += end - start
            total_batches += 1
            sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()
            del X,Y,f_X
        test_acc = sum_acc/counts
        print('Final Test Accuracy: {0:0.3f}'.format(test_acc))
        print('mean evaluation time of one batch: {0:0.9f}'.format(total_time / (total_batches - 1)))
        print ('parameter number', sum(p.numel() for p in t.D.parameters() if p.requires_grad))
        return test_acc

if __name__ == "__main__":
    t = PointCloudTrainer()
    t.test()
