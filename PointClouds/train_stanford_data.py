from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm, trange
import numpy as np

import classifier
import modelnet_pointnet as modelnet
from modelnet_data import ModelnetData


#################### Settings ##############################
num_epochs = 1000
batch_size = 4
downsample = 10    #For 5000 points use 2, for 1000 use 10, for 100 use 100
network_dim = 512  #For 5000 points use 512, for 1000 use 256, for 100 use 256
num_repeats = 5    #Number of times to repeat the experiment
#data_path = 'ModelNet40_cloud.h5'
#model_path = 'model_1.pt'
GPU_IS_ON = True
#################### Settings ##############################


class PointCloudTrainer(object):
    def __init__(self):
        
        #Data loader
        #self.model_fetcher = modelnet.ModelFetcher(data_path, batch_size, downsample, do_standardize=True, do_augmentation=True)
        self.model_fetcher = ModelnetData(pointcloud_size = 1000)

        #Setup network
        if GPU_IS_ON:
            self.D = classifier.DTanh(network_dim, pool='max1').cuda()
            self.L = nn.CrossEntropyLoss().cuda()
        else:
            self.D = classifier.DTanh(network_dim, pool='max1')
            self.L = nn.CrossEntropyLoss()
        
        
        self.optimizer = optim.Adam([{'params':self.D.parameters()}], lr=1e-3, weight_decay=1e-7, eps=1e-3)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(400,num_epochs,400)), gamma=0.1)
        #self.optimizer = optim.Adamax([{'params':self.D.parameters()}], lr=5e-4, weight_decay=1e-7, eps=1e-3) # optionally use this for 5000 points case, but adam with scheduler also works
        
        
#         if GPU_IS_ON:
#             self.D.load_state_dict(torch.load(model_path))
#         else:
#             self.D.load_state_dict(torch.load(model_path, map_location='cpu'))

    def train(self):
        self.D.train()
        loss_val = float('inf')
        for j in trange(num_epochs, desc="Epochs: "):
            counts = 0
            sum_acc = 0.0
#            train_data = self.model_fetcher.train_data(loss_val)
#             for x, _, y in train_data:
            for data, label in self.model_fetcher.generate_random_batch(train=True, batch_size=batch_size,
                                                                        shuffle_clouds=True, shuffle_points=True,
                                                                        jitter_points=True, rotate_pointclouds=False,
                                                                        rotate_pointclouds_up=True, sampling_method='fps'):
                counts += len(label)
                if GPU_IS_ON:
                    X = Variable(torch.cuda.FloatTensor(data))
                    Y = Variable(torch.cuda.LongTensor(label))
                else:
                    X = Variable(torch.FloatTensor(data))
                    Y = Variable(torch.LongTensor(label))
                self.optimizer.zero_grad()
                f_X = self.D(X)
                loss = self.L(f_X, Y)
                loss_val = loss.data.cpu().numpy()
                sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()
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
        total_time = 0.
        total_batches = 0.
        for data, label in self.model_fetcher.generate_random_batch(train=False, batch_size=batch_size,
                                                                    shuffle_clouds=False, shuffle_points=False,
                                                                    jitter_points=False, rotate_pointclouds=False,
                                                                    rotate_pointclouds_up=False, sampling_method='fps'):
        ##for x, _, y in self.model_fetcher.test_data():
            
            counts += len(label)
            #X = Variable(torch.cuda.FloatTensor(x))
            #Y = Variable(torch.cuda.LongTensor(y))
            X = Variable(torch.FloatTensor(data))
            Y = Variable(torch.LongTensor(label))
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
    test_accs = []
    for i in range(8, num_repeats):
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

