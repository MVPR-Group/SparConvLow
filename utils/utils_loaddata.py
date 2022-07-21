import torch.utils.data.dataloader as Data
import torchvision
import torchvision.transforms as transforms

from arg import args


def loaddata(dataset):
    if dataset == 'PIE':
        size = 32
        transform_train = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.3157, 0.3157, 0.3157),
                                 (0.2177, 0.2177, 0.2177)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.3157, 0.3157, 0.3157),
                                 (0.2177, 0.2177, 0.2177)),
        ])

        train_data = torchvision.datasets.ImageFolder(
            './data/PIE/train_'+str(args.net_numclass), transform=transform_train
        )
        test_data = torchvision.datasets.ImageFolder(
            './data/PIE/test_'+str(args.net_numclass), transform=transform_test
        )
        train_loader = Data.DataLoader(
            dataset=train_data, batch_size=64, shuffle=True)

        test_loader = Data.DataLoader(
            dataset=test_data, batch_size=64, shuffle=True)

        train_data_all = torchvision.datasets.ImageFolder(
            './data/PIE/ALL', transform=transform_train
        )
        train_loader_Dinit_UDPtest = Data.DataLoader(dataset=train_data_all,
                                                     batch_size=args.number_class * 170,
                                                     shuffle=False)

        train_data_ = torchvision.datasets.ImageFolder(
            './data/PIE/ALL', transform=transform_train
        )
        test_data_ = torchvision.datasets.ImageFolder(
            './data/PIE/ALL', transform=transform_test,
        )
        train_loader_UDPtrain = Data.DataLoader(dataset=train_data_,
                                                batch_size=args.number_class * args.number_perclass_trainUDP,
                                                shuffle=True)
        test_loader_Dinit_UDPtest = Data.DataLoader(
            dataset=test_data_, batch_size=args.number_class * 170, shuffle=False)

    elif dataset == 'Cifar10':
        size = 32
        transform_train = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_data = torchvision.datasets.CIFAR10(
            './data/cifar10', train=True, transform=transform_train, download=True
        )
        test_data = torchvision.datasets.CIFAR10(
            './data/cifar10', train=False, transform=transform_test, download=True
        )
        train_data_ = CIFAR10(
            './data/cifar10', train=True, transform=transform_train
        )
        train_data_all = CIFAR10all(
            './data/cifar10', train=True, transform=transform_train
        )
        test_data_ = CIFAR10test(
            './data/cifar10', train=False, transform=transform_test
        )
        train_loader = Data.DataLoader(
            dataset=train_data, batch_size=64, shuffle=True)
        test_loader = Data.DataLoader(
            dataset=test_data, batch_size=64, shuffle=True)

        train_loader_Dinit_UDPtest = Data.DataLoader(dataset=train_data_all, batch_size=args.number_class * args.perclass_trainDinit,
                                                     shuffle=False)
        train_loader_UDPtrain = Data.DataLoader(dataset=train_data_,
                                                batch_size=args.number_class * args.number_perclass_trainUDP,
                                                shuffle=False)
        test_loader_Dinit_UDPtest = Data.DataLoader(
            dataset=test_data_, batch_size=10000, shuffle=False)

    elif dataset == 'Caltech256':
        if args.perclass_trainDinit == 60:
            trainDir = './data/Caltech256/train_60'
            normalize = transforms.Normalize(
                [0.5547, 0.5330, 0.5048], [0.3185, 0.3155, 0.3284])
        if args.perclass_trainDinit == 30:
            trainDir = './data/Caltech256/train_30'
            normalize = transforms.Normalize(
                [0.5552, 0.5336, 0.5056], [0.3192, 0.3162, 0.3289])
        if args.perclass_trainDinit == 15:
            trainDir = './data/Caltech256/train_15'
            normalize = transforms.Normalize(
                [0.5571, 0.5352, 0.5093], [0.3199, 0.3175, 0.3305])

        transform_Cal_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_Cal_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        train_data = torchvision.datasets.ImageFolder(
            trainDir, transform=transform_Cal_train
        )
        test_data = torchvision.datasets.ImageFolder(
            './data/Caltech256/test', transform=transform_Cal_test
        )
        train_loader = Data.DataLoader(
            dataset=train_data, batch_size=64, shuffle=True)

        test_loader = Data.DataLoader(
            dataset=test_data, batch_size=64, shuffle=True)

        train_data_all = torchvision.datasets.ImageFolder(
            trainDir, transform=transform_Cal_train
        )
        train_loader_Dinit_UDPtest = Data.DataLoader(dataset=train_data_all,
                                                     batch_size=args.number_class * 15,
                                                     shuffle=False)

        train_data_ = torchvision.datasets.ImageFolder(
            trainDir, transform=transform_Cal_train
        )
        test_data_ = torchvision.datasets.ImageFolder(
            './data/Caltech256/test_D_2', transform=transform_Cal_test,
        )
        train_loader_UDPtrain = Data.DataLoader(dataset=train_data_,
                                                batch_size=args.number_class * args.number_perclass_dict,
                                                shuffle=True)
        test_loader_Dinit_UDPtest = Data.DataLoader(
            dataset=test_data_, batch_size=args.number_class * 15, shuffle=False)

    #      0net,         0net,        2 update_UDP,          1,2 Dinit/UDPtest, 1,2 Dinit/UDPtest,
    return train_loader, test_loader, train_loader_UDPtrain, train_loader_Dinit_UDPtest, test_loader_Dinit_UDPtest
