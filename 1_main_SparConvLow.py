import os

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import numpy as np
from utils_LDAsparlow import DicInit
from utils_LDAsparlow import func_loss_my, testacc
from arg import args
from utils_loaddata import loaddata
from utils_select_model import select_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # choose GPU
device_ids = range(torch.cuda.device_count())
device = args.device  # 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0
start_epoch = 0

param_name = './DLparam_init/param-' + str(args.model) + '-' + str(args.dataset) + '-' + str(args.lamda1) + '-' + str(
    args.number_perclass_dict) + '-' + str(args.perclass_trainDinit) + '-' + str(args.number_perclass_trainUDP) + '.npy'
print(param_name)

SparConvnet_ckpt_name = './checkpoint/SparConv-ckpt-' + str(args.model) + '-' + str(args.dataset) + '-' + str(
    args.lamda1) + '-' + str(
    args.number_perclass_dict) + '-' + str(args.perclass_trainDinit) + '-' + str(
    args.number_perclass_trainUDP) + '.pth'

SparConvnet_param_name = './DLparam_init/SparConv-param-' + str(args.model) + '-' + str(args.dataset) + '-' + str(
    args.lamda1) + '-' + str(
    args.number_perclass_dict) + '-' + str(args.perclass_trainDinit) + '-' + str(
    args.number_perclass_trainUDP) + '.npy'

net_ckpt_name = './checkpoint/ckpt-' + str(args.model) + '-'+ str(args.net_numclass)+ '-' + str(args.dataset)+ str(args.perclass_trainDinit) + '.pth'

# Data
print('==> Preparing data..')
dataset = args.dataset
_, _, train_loader_UDPtrain, train_loader_Dinit_UDPtest, test_loader_Dinit_UDPtest = loaddata(dataset)

# Model
print('==> Building model..')
net = select_model(args.model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

checkpoint = torch.load(net_ckpt_name)
if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    cudnn.benchmark = True

net = net.to(device)
print(args.model)

net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print('==> Resuming from checkpoint..')
print('before_acc:{:.6f}'.format(best_acc))

train_amount = args.perclass_trainDinit 

# dictionary initialization
with torch.no_grad():
    batch_xtrain = torch.Tensor([])
    batch_ytrain = torch.Tensor([])
    batch_xtest = torch.Tensor([])
    batch_ytest = torch.Tensor([])
    feature_temp = torch.Tensor([]).to(device)
    batch_ytrain_temp = torch.Tensor([]).to(device)
    featuretest_temp = torch.Tensor([]).to(device)
    batch_ytest_temp = torch.Tensor([]).to(device)
    net.eval()

    # CNN features
    for batch_xtrain,batch_ytrain in train_loader_Dinit_UDPtest:
        batch_xtrain, batch_ytrain = batch_xtrain.to(device), batch_ytrain.to(device)
        print('==>train', batch_xtrain.shape, batch_ytrain.shape)

        out_train, feature = net(batch_xtrain)
        _, predicted = out_train.max(1)
        total = batch_ytrain.size(0)
        correct = predicted.eq(batch_ytrain).sum().item()
        print('net_acc:', 100. * correct / total, correct, predicted)
        feature_temp = torch.cat((feature, feature_temp))
        batch_ytrain_temp =  torch.cat((batch_ytrain, batch_ytrain_temp))

    print(feature_temp.shape)

    for batch_xtest, batch_ytest in test_loader_Dinit_UDPtest:
        batch_xtest, batch_ytest = batch_xtest.to(device), batch_ytest.to(device)
        print('==>test', batch_xtest.shape, batch_ytest.shape)
        out_test, featuretest = net(batch_xtest)
        _, predicted = out_test.max(1)
        total = batch_ytest.size(0)
        correct = predicted.eq(batch_ytest).sum().item()
        print('net_acc:', 100. * correct / total, correct, predicted)
        featuretest_temp = torch.cat((featuretest, featuretest_temp))
        batch_ytest_temp =  torch.cat((batch_ytest, batch_ytest_temp))

    print(featuretest_temp.shape)
    feature = feature_temp
    featuretest = featuretest_temp
    batch_ytrain = batch_ytrain_temp
    batch_ytest = batch_ytest_temp

    print('train-', feature.size(), '| trainlabel-', batch_ytrain.size(), '| test-', featuretest.size(),
          '| testlabel-', batch_ytest.size())
    # save features
    # net_array = {
    #     'batch_xtrain': batch_xtrain.cpu().detach().numpy(),
    #     'train_feature': feature.cpu().detach().numpy(),
    #     'train_label': batch_ytrain.cpu().detach().numpy().astype('int64'),
    #     'batch_xtest': batch_xtest.cpu().detach().numpy(),
    #     'test_feature': featuretest.cpu().detach().numpy(),
    #     'test_label': batch_ytest.cpu().detach().numpy().astype('int64')
    # }
    # D_init_net_array_name = 'D_init_net_array-' + str(args.model) + '-' + str(args.dataset) + '-' + str(
    #     args.lamda1) + '-' + str(
    #     args.number_perclass_dict) + '-' + str(args.perclass_trainDinit) + '-' + str(
    #     args.number_perclass_trainUDP) + '.npy'
    # np.save(D_init_net_array_name, net_array)

    param = DicInit(feature.cpu(), batch_ytrain.cpu(), featuretest.cpu(), batch_ytest.cpu())
    # save D and other parameters
    # SparDinit_param_name = './DLparam_init/Dinit-' + str(args.model) + '-' + str(args.dataset) + '-' + str(
    #     args.lamda1) + '-' + str(
    #     args.number_perclass_dict) + '-' + str(args.perclass_trainDinit) + '-' + str(
    #     args.number_perclass_trainUDP) + '.npy'
    # np.save(SparDinit_param_name, param)
#
print('------------DicInit END----------------')

with torch.no_grad():
    batch_xtrain = torch.Tensor([])
    batch_ytrain = torch.Tensor([])
    batch_xtest = torch.Tensor([])
    batch_ytest = torch.Tensor([])
    feature_temp = torch.Tensor([]).to(device)
    batch_ytrain_temp = torch.Tensor([]).to(device)
    featuretest_temp = torch.Tensor([]).to(device)
    batch_ytest_temp = torch.Tensor([]).to(device)

    for batch_xtrain,batch_ytrain in train_loader_Dinit_UDPtest:
        batch_xtrain, batch_ytrain = batch_xtrain.to(device), batch_ytrain.to(device)
        _, featuretrain = net(batch_xtrain)

        feature_temp = torch.cat((featuretrain, feature_temp))
        batch_ytrain_temp = torch.cat((batch_ytrain, batch_ytrain_temp))

    for batch_xtest, batch_ytest in test_loader_Dinit_UDPtest:
        batch_xtest, batch_ytest = batch_xtest.to(device), batch_ytest.to(device)
        _, featuretest = net(batch_xtest)

        featuretest_temp = torch.cat((featuretest, featuretest_temp))
        batch_ytest_temp = torch.cat((batch_ytest, batch_ytest_temp))

    featuretrain = feature_temp
    featuretest = featuretest_temp
    batch_ytrain = batch_ytrain_temp
    batch_ytest = batch_ytest_temp

    svmscore, Accuracy_directLDA, ldascore, knnscore, printsparsity = testacc(featuretrain, featuretest,
                                                                              batch_ytrain, batch_ytest,
                                                                              param)

    print('directLDA: {:.6f}，ldascore: {:.6f}，acc_knnscore: {:.6f}'.format(Accuracy_directLDA, ldascore, knnscore))

print('>>-----------------------------<<')
print(param_name + ' done')

# CNN+SparDR

for epoch in range(10):
    print('----epoch:{:d}----------'.format(epoch))
    acc_lda = 0.0
    acc_ldascore = 0.0
    acc_knnscore = 0.0
    num_batch = 0
    total = 0
    correct = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    for batch_xtrain, batch_ytrain in train_loader_Dinit_UDPtest:
        print('batch:{:d}'.format(num_batch))

        batch_xtrain, batch_ytrain = Variable(batch_xtrain).to(device), Variable(batch_ytrain).to(device)
        _, feature = net(batch_xtrain)

        loss, param = func_loss_my(feature, batch_ytrain, param)
        print(loss.shape, feature.shape)
        optimizer.zero_grad()
        feature.backward(loss.to(device))
        optimizer.step()
        num_batch += 1

        with torch.no_grad():
            batch_xtrain = torch.Tensor([])
            batch_ytrain = torch.Tensor([])
            batch_xtest = torch.Tensor([])
            batch_ytest = torch.Tensor([])
            feature_temp = torch.Tensor([]).to(device)
            batch_ytrain_temp = torch.Tensor([]).to(device)
            featuretest_temp = torch.Tensor([]).to(device)
            batch_ytest_temp = torch.Tensor([]).to(device)

            for batch_xtrain, batch_ytrain in train_loader_Dinit_UDPtest:
                batch_xtrain, batch_ytrain = batch_xtrain.to(device), batch_ytrain.to(device)
                _, featuretrain = net(batch_xtrain)
                feature_temp = torch.cat((featuretrain, feature_temp))
                batch_ytrain_temp = torch.cat((batch_ytrain, batch_ytrain_temp))

            for batch_xtest, batch_ytest in test_loader_Dinit_UDPtest:
                batch_xtest, batch_ytest = batch_xtest.to(device), batch_ytest.to(device)
                _, featuretest = net(batch_xtest)
                featuretest_temp = torch.cat((featuretest, featuretest_temp))
                batch_ytest_temp = torch.cat((batch_ytest, batch_ytest_temp))

            featuretrain = feature_temp
            featuretest = featuretest_temp
            batch_ytrain = batch_ytrain_temp
            batch_ytest = batch_ytest_temp
            print('testacc--',featuretrain.shape, batch_ytrain.shape, featuretest.shape, batch_ytest.shape)
            svmscore, Accuracy_directLDA, ldascore, knnscore, printsparsity = testacc(featuretrain, featuretest,
                                                                                      batch_ytrain, batch_ytest,
                                                                                      param)
            # save model
            # state = {
            #     'net': net.state_dict(),
            #     'epoch': epoch,
            #     'batch': num_batch,
            # }
            # torch.save(state, SparConvnet_ckpt_name)
            # np.save(SparConvnet_param_name, param)
            # print('directLDA: {:.6f}，ldascore: {:.6f}，acc_knnscore: {:.6f}'.format( Accuracy_directLDA, ldascore, knnscore))
