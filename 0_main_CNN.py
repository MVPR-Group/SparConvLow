from utils_select_model import select_model
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from models import *
from utils_LDAsparlow import progress_bar
from utils_loaddata import loaddata
from arg import args


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # # choose GPU
device_ids = range(torch.cuda.device_count())
device = args.device  # 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
dataset = args.dataset
train_loader, test_loader, _, _, _ = loaddata(dataset)

# Model
print('==> Building model..')
net = select_model(args.model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
if device == 'cuda':
    print("divice_ids:", device_ids)
    net = nn.DataParallel(net, device_ids=device_ids)
    cudnn.benchmark = True
net = net.to(device)

net_ckpt_name = './checkpoint/ckpt-' + str(args.model) + '-' + str(
    args.net_numclass) + '-' + str(args.dataset) + str(args.perclass_trainDinit) + '.pth'
print(net_ckpt_name, args.perclass_trainDinit)

if os.path.exists(net_ckpt_name):
    checkpoint = torch.load(net_ckpt_name)
    net.load_state_dict(checkpoint['net'], strict=False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Resuming from checkpoint..', net_ckpt_name, checkpoint['acc'])
else:
    print('no ' + net_ckpt_name + ' found')


# Training
def train(epoch):
    print('\nEpoch: %d ' % epoch + '-' +
          str(args.model) + '-' + str(args.dataset))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


# Test
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            # _, predicted = outputs.topk(5,1,True,True)
            # correct += predicted.eq(targets.view(-1,1)).sum().item()
            total += targets.size(0)

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        best_acc = acc
        # torch.save(state, net_ckpt_name)


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
    scheduler.step()
