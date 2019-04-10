import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm, trange
import numpy as np

import classifier
import data_provider

#################### Settings ##############################
num_epochs = 1000
batch_size = 64
#point_cloud_size = 1000
network_dim = 512           #For 5000 points use 512, for 1000 use 256, for 100 use 256
num_repeats = 1             #Number of times to repeat the experiment
#################### Settings ##############################


class PointCloudTrainer(object):
    def __init__(self):
        #Data loader
        #self.model_fetcher = modelnet.ModelFetcher(data_path, batch_size, downsample, do_standardize=True, do_augmentation=True)
#        self.data_provider = data_provider.ModelNet40(16)
        self.data_provider = data_provider.ModelNet40Downsampled(16, distance_metric='equal_agg', pointnet_base='fps')

        #Setup network
        self.D = classifier.DTanh(network_dim, pool='max1').cuda()
        self.L = nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam([{'params':self.D.parameters()}], lr=1e-3, weight_decay=1e-7, eps=1e-3)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(400,num_epochs,400)), gamma=0.1)
        #self.optimizer = optim.Adamax([{'params':self.D.parameters()}], lr=5e-4, weight_decay=1e-7, eps=1e-3) # optionally use this for 5000 points case, but adam with scheduler also works

    def train(self):
        self.D.train()
        loss_val = float('inf')
        for j in trange(num_epochs, desc="Epochs: "):
            counts = 0
            sum_acc = 0.0
            for clouds, labels in self.data_provider.generate_random_batch(True, batch_size=batch_size,
                                                                          shuffle_clouds=True,
                                                                          shuffle_points=True,
                                                                          jitter_points=True,
                                                                          rotate_pointclouds=False,
                                                                          rotate_pointclouds_up=True,
                                                                          sampling_method='fps'):
                counts += len(labels)
                X = Variable(torch.cuda.FloatTensor(clouds))
                Y = Variable(torch.cuda.LongTensor(labels))
                self.optimizer.zero_grad()
                f_X = self.D(X)
                loss = self.L(f_X, Y)
                loss_val = loss.data.cpu().item()
                sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().item()
                #train_data.set_description('Train loss: {0:.4f}'.format(loss_val))
                loss.backward()
                classifier.clip_grad(self.D, 5)
                self.optimizer.step()
                del X,Y,f_X,loss
            train_acc = sum_acc/counts
            self.scheduler.step()
            if j%10==9:
                tqdm.write('After epoch {0} Train Accuracy: {1:0.3f} '.format(j+1, train_acc))

    def test(self):
        self.D.eval()
        counts = 0
        sum_acc = 0.0
        for clouds, labels in self.data_provider.generate_random_batch(False, batch_size=batch_size,
                                                                       shuffle_clouds=True,
                                                                       shuffle_points=True,
                                                                       jitter_points=True,
                                                                       rotate_pointclouds=False,
                                                                       rotate_pointclouds_up=True,
                                                                       sampling_method='fps'):
            counts += len(labels)
            X = Variable(torch.cuda.FloatTensor(clouds))
            Y = Variable(torch.cuda.LongTensor(labels))
            f_X = self.D(X)
            sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().item()
            del X,Y,f_X
        test_acc = sum_acc/counts
        print('Final Test Accuracy: {0:0.3f}'.format(test_acc))
        return test_acc

if __name__ == "__main__":
    test_accs = []
    for i in range(0, num_repeats):
        print('='*30 + ' Start Run {0}/{1} '.format(i+1, num_repeats) + '='*30)
        t = PointCloudTrainer()
        t.train()
        torch.save(t.D.state_dict(), 'model_' + str(i+1) + '.pt')
        acc = t.test()
        test_accs.append(acc)
        print('='*30 + ' Finish Run {0}/{1} '.format(i+1, num_repeats) + '='*30)
    print('\n')
    if num_repeats > 2:
        try:
            print('Test accuracy: {0:0.2f} '.format(np.mean(test_accs)) + unichr(177).encode('utf-8') + ' {0:0.3f} '.format(np.std(test_accs)))
        except:
            print('Test accuracy: {0:0.2f} +/-  {0:0.3f} '.format(np.mean(test_accs), np.std(test_accs)))
