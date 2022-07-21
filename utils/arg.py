"""
参数文件
"""
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
parser.add_argument('--device', default='cuda', type=str, help='choose device to run model')
parser.add_argument('--local_rank', default=-1, type=int, help='sparse number')

parser.add_argument('--recordW', default=0, type=int) #fix

parser.add_argument('--dataset', default='ImageNet', type=str)
parser.add_argument('--model', default='ResNet18', type=str, help='be selected model')  #VGG16 MobileNet
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--lamda1', default=0.055, type=int, help='sparse number')

if parser.parse_args().dataset == 'PIE':
    parser.add_argument('--net_numclass', default=68, type=int, help='number of class in CNN model') 
    parser.add_argument('--number_class', default=68, type=int, help='number of class in framework') 
    parser.add_argument('--perclass_trainDinit', default=128, type=int, help='Phi 2 dimension f*[s] / initD _ loader')
    parser.add_argument('--number_perclass_dict', default=128, type=int, help='Dictionary 2 dimension: <= perclass_trainDinit')
    parser.add_argument('--number_perclass_trainUDP', default=200, type=int, help='number of samples for training / update D')
if parser.parse_args().dataset == 'Cifar10':
    parser.add_argument('--net_numclass', default=10, type=int, help='number of class in CNN model') 
    parser.add_argument('--number_class', default=10, type=int, help='number of class in framework')
    parser.add_argument('--perclass_trainDinit', default=1000, type=int, help='Phi 2 dimension f*[s] / initD _ loader')
    parser.add_argument('--number_perclass_dict', default=1000, type=int, help='Dictionary 2 dimension: <= perclass_trainDinit')
    parser.add_argument('--number_perclass_trainUDP', default=100, type=int, help='number of samples for training / update D')
if parser.parse_args().dataset == 'Caltech256':
    parser.add_argument('--net_numclass', default=256, type=int, help='number of class in CNN model') 
    parser.add_argument('--number_class', default=256, type=int, help='number of class in framework') 
    parser.add_argument('--perclass_trainDinit', default=30, type=int, help='Phi 2 dimension： f*[s] / initD _ loader')
    parser.add_argument('--number_perclass_dict', default=30, type=int, help='Dictionary 2 dimension: <= perclass_trainDinit')
    parser.add_argument('--number_perclass_trainUDP', default=50, type=int, help='number of samples for training / update D')

args = parser.parse_args()
