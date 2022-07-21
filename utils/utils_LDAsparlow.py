import torch.nn.init as init
import torch.nn as nn
import torch
import time
import sys
import os
import random

import numpy as np
import spams
from scipy.linalg import block_diag
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from arg import args


def obtaintraingtestingsamples(featureMatrix, labelMatrix, numPerClass):
    numClass = labelMatrix.shape[0]
    print('numClass', labelMatrix.shape)
    dimen_fea = featureMatrix.shape[0]
    testPart = np.zeros([dimen_fea, 1])
    HtestPart = np.zeros([numClass, 1])
    trainPart = np.zeros([dimen_fea, 1])
    HtrainPart = np.zeros([numClass, 1])
    for classid in range(numClass):
        col_ids = np.argwhere(labelMatrix[classid, :] == 1)[:, 0]
        data_ids = np.argwhere(colnorms_squared_new(
            featureMatrix[:, col_ids]) > 1e-6)[:, 0]
        perm = np.array(random.sample(
            range(np.size(data_ids, 0)), np.size(data_ids, 0)))
        trainids = col_ids[data_ids[perm[:numPerClass]]]
        testids = np.setdiff1d(col_ids, trainids)
        testPart = np.concatenate(
            (testPart, featureMatrix[:, testids]), axis=1)
        HtestPart = np.concatenate(
            (HtestPart, labelMatrix[:, testids]), axis=1)
        trainPart = np.concatenate(
            (trainPart, featureMatrix[:, trainids]), axis=1)
        HtrainPart = np.concatenate(
            (HtrainPart, labelMatrix[:, trainids]), axis=1)
    return testPart[:, 1:], HtestPart[:, 1:], trainPart[:, 1:], HtrainPart[:, 1:]


def colnorms_squared_new(X):
    blocksize = 6000
    for i in range(0, np.size(X, 1), blocksize):
        blockids = np.array(
            [a for a in range(i, min(i + blocksize - 1, np.size(X, 1)))])
        Y = np.sum(np.square(X[:, blockids]), axis=0)

    return Y


def PCA_eigv(fea, options):
    cov = np.cov(fea, rowvar=0)
    fea_value, fea_vect = np.linalg.eig(cov)
    index = np.argsort(fea_value)
    n_index = index[-1: -(options['ReducedDim'] + 1):-1]
    n_fea_vect = fea_vect[:, n_index]
    return n_fea_vect


def initialDicRandom(row, col):
    M = col
    N = row
    Phi = np.random.randn(N, M)
    Dint = Phi / np.tile(np.sqrt(np.sum(Phi ** 2, axis=0)), (N, 1))
    return Dint


def paramterinitializationMe(training_feats, H_train, para):
    dictsize = para['numBases']
    iterations = para['iterationini']
    numClass = np.size(H_train, 0)
    dimen_fea = np.size(training_feats, 0)
    Dinit = np.zeros([dimen_fea, 1])
    dictLabel = np.zeros([numClass, 1])
    numPerClass = dictsize / numClass
    numPerClass = int(numPerClass)
    for classid in range(numClass):
        labelvector = np.zeros([numClass, 1])
        labelvector[classid] = 1
        dictLabel = np.concatenate(
            (dictLabel, np.tile(labelvector, (1, numPerClass))), axis=1)
    dictLabel = dictLabel[:, 1:]
    for classid in range(numClass):
        col_ids = np.argwhere(H_train[classid, :] == 1)[:, 0]
        data_ids = np.argwhere(colnorms_squared_new(
            training_feats[:, col_ids]) > 1e-6)[:, 0]
        perm = np.array(random.sample(
            range(np.size(data_ids, 0)), np.size(data_ids, 0)))

        Dpart = training_feats[:, col_ids[data_ids[perm[:numPerClass]]]]
        param1 = dict()
        param1['mode'] = 2
        param1['K'] = para['numBases']
        param1['lambda'] = para['lambda']
        param1['lambda2'] = para['lambda2']
        param1['iter'] = iterations
        if para['initialDic_type'] == 'random':
            row, col = Dpart.shape
            Dpart = initialDicRandom(row, col)
            param1['D'] = Dpart.astype(float)
        else:
            param1['D'] = Dpart.astype(float)
        X = np.asfortranarray(
            training_feats[:, col_ids[data_ids]], dtype=float)
        param2 = {'K': param1['K'],  # learns a dictionary with 100 elements
                  'lambda1': param1['lambda'],
                  'lambda2': param1['lambda2'],
                  'mode': param1['mode'],
                  'D': param1['D'],
                  'iter': param1['iter']}
        print('X.shape', X.shape)
        Dpart = spams.trainDL(X, **param2)
        Dinit = np.concatenate((Dinit, Dpart), axis=1)

    Dinit = Dinit[:, 1:]
    return Dinit


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T


def initD(featureMat, labelMat, featureMattest, labelMattest):
    param = dict()
    param['lamda1'] = args.lamda1
    param['lamda2'] = 0.0000001
    # number of samples for training / update D
    param['perclass_trainDinit'] = args.perclass_trainDinit
    # the second dimension of D = classnumber*number_perclass_dict
    param['number_perclass_dict'] = args.number_perclass_dict
    param['reducedMethod'] = 'wavelet'
    param['initialDic_type'] = 'Norandom'
    param['ReducedDim_SPM'] = 80
    param['classnumber'] = args.number_class

    featureMat = featureMat.cpu().detach().numpy()
    labelMat = labelMat.cpu().detach().numpy().astype('int64')
    featureMattest = featureMattest.cpu().detach().numpy()
    labelMattest = labelMattest.cpu().detach().numpy().astype('int64')

    labelMat = convert_to_one_hot(labelMat, len(set(labelMat)))
    featureMat = np.array(featureMat.T)
    labelMat = np.array(labelMat)
    labelMattest = convert_to_one_hot(labelMattest, len(set(labelMattest)))
    featureMattest = np.array(featureMattest.T)
    labelMattest = np.array(labelMattest)
    classnumber = param['classnumber']
    pars = dict()
    pars['lambda'] = param['lamda1']
    pars['lambda2'] = param['lamda2']
    pars['iterationini'] = 100
    pars['initialDic_type'] = param['initialDic_type']
    pars['ntrainsamples'] = param['perclass_trainDinit']
    pars['numBases'] = classnumber * \
        param['number_perclass_dict']  # the second dimension of D

    testing_feats, H_test, training_feats, H_train = obtaintraingtestingsamples(featureMat, labelMat,
                                                                                pars['ntrainsamples'])
    H_test, testing_feats = labelMattest, featureMattest

    if param['reducedMethod'] == 'wavelet':
        pass
    elif param['reducedMethod'] == 'PCA':
        options = dict()
        options['ReducedDim'] = param['ReducedDim_SPM']
        fea = training_feats.T
        eigvector = PCA_eigv(training_feats.T, options)
        Data = np.dot(training_feats.T, eigvector)
        training_feats = Data.T
        Data = np.dot(testing_feats.T, eigvector)
        testing_feats = Data.T

    labelvector_train = np.argwhere(H_train == 1)[:, 0]
    labelvector_test = np.argwhere(H_test == 1)[:, 0]
    trainsampleid = np.argwhere(labelvector_train <= classnumber)[:, 0]
    testsampleid = np.argwhere(labelvector_test <= classnumber)[:, 0]
    trainingsubset = training_feats[:, trainsampleid]
    H_train_subset = H_train[:classnumber, trainsampleid]

    Dinit = paramterinitializationMe(trainingsubset, H_train_subset, pars)

    print('Dinit.shape, training_feats.shape, testing_feats.shape, labelvector_train.shape, labelvector_test.shape',
          Dinit.shape, training_feats.shape, testing_feats.shape, labelvector_train.shape, labelvector_test.shape)
    return Dinit, training_feats, testing_feats, labelvector_train, labelvector_test, param


def directldaFreeTest(TrainingVector, TestingVector, labelvector_train, labelvector_test, NumOfTraining, NumOfClass,
                      dim, DistMeasure):
    print("dim:", dim, TrainingVector.shape)
    MeanVector = np.zeros([dim, NumOfClass])
    for i in range(NumOfClass):
        MeanVector[:, i] = np.mean(
            TrainingVector[:, (i * NumOfTraining):(i + 1) * NumOfTraining], axis=1)
    MeanVectorOfAllVector = np.mean(MeanVector, axis=1)
    print('NumOfClass * NumOfTraining', NumOfClass,
          NumOfTraining, NumOfClass * NumOfTraining)
    MatrixW = np.zeros([dim, NumOfClass * NumOfTraining])
    for i in range(NumOfClass):
        for j in range(NumOfTraining):
            MatrixW[:, i * NumOfTraining + j] = (TrainingVector[:, i * NumOfTraining + j] - MeanVector[:, i]) / np.sqrt(
                NumOfTraining * NumOfClass)

    MatrixB = np.zeros([dim, NumOfClass])
    for i in range(NumOfClass):
        MatrixB[:, i] = (MeanVector[:, i] -
                         MeanVectorOfAllVector) / np.sqrt(NumOfClass)

    MatrixT = np.zeros([dim, NumOfClass * NumOfTraining])
    for i in range(NumOfClass):
        for j in range(NumOfTraining):
            MatrixT[:, i * NumOfTraining + j] = (TrainingVector[:,
                                                 i * NumOfTraining + j] - MeanVectorOfAllVector) / np.sqrt(
                NumOfTraining * NumOfClass)

    MatrixSB = np.dot(MatrixB.T, MatrixB)
    Ub, Sb, Vb = np.linalg.svd(MatrixSB)
    Sb = np.diag(Sb)
    RowOfSb, ColOfSb = Sb.shape
    i = 0
    while (i <= RowOfSb):
        if (Sb[i, i] > 1e-10):
            i = i + 1
        else:
            break
    i = i + 1
    rs = i - 1
    if rs > (NumOfClass - 1):
        rs = NumOfClass - 1
    MatrixY = np.zeros([MatrixB.shape[0], rs])
    for i in range(rs):
        MatrixY[:, i] = np.dot(MatrixB, Ub[:, i]) / \
            np.linalg.norm(np.dot(MatrixB, Ub[:, i]))
    MatrixDb = Sb[:rs, :rs]
    for i in range(rs):
        MatrixDb[i, i] = MatrixDb[i, i] ** (-1 / 2)
    MatrixZ = np.dot(MatrixY, MatrixDb)
    MatrixSwp = np.dot(np.dot(np.dot(MatrixZ.T, MatrixW), MatrixW.T), MatrixZ)
    Uw, Sw, Vw = np.linalg.svd(MatrixSwp)
    Sw = np.diag(Sw)
    RowOfSw, ColOfSw = Sw.shape
    MatrixA = Uw[:, np.diag(Sw) < 1]
    MatrixProjectionV = np.dot(MatrixZ, MatrixA)
    print('MatrixProjectionV', MatrixProjectionV.shape,
          'MeanVector', MeanVector[:, 1].shape)  # (512,14)(512)
    SizeOfSampleVector = np.dot(MatrixProjectionV.T, MeanVector[:, 1]).shape[0]
    SampleVector = np.zeros([SizeOfSampleVector, NumOfClass])
    TempVector = np.zeros([SizeOfSampleVector, NumOfClass * NumOfTraining])
    for i in range(NumOfClass):
        for j in range(NumOfTraining):
            TempVector[:, i * NumOfTraining + j] = np.dot(
                MatrixProjectionV.T, TrainingVector[:, i * NumOfTraining + j])
        SampleVector[:, i] = np.mean(
            TempVector[:, i * NumOfTraining + 1:(i + 1) * NumOfTraining].T, 0).T

    count = 0
    TainingSamplesMean = SampleVector
    VectorForTest = np.dot(MatrixProjectionV.T, TestingVector)
    DistanceArray = np.zeros([NumOfClass])
    if DistMeasure == 'mindist':
        print('labelvector_test', labelvector_test.shape[0])
        for i in range(labelvector_test.shape[0]):
            for k in range(NumOfClass):
                DistanceArray[k] = np.linalg.norm(
                    TainingSamplesMean[:, k] - VectorForTest[:, i])
            I = np.argmin(DistanceArray[:])
            if I == labelvector_test[i]:
                count = count + 1
    Accuracy = count / labelvector_test.shape[0]
    print('directldaFreeTest -> count', count,
          labelvector_test.shape[0], count/labelvector_test.shape[0])
    return Accuracy, MatrixProjectionV


def init_parameters(param):
    size_ydata = param['D_size']
    param['GlobalMethod'] = 'Steepes_gradient'
    param['channel_initial_dic'] = 'From_data'
    param['inco'] = 'Dinctionary is incoherent'
    K = size_ydata[1]
    bb = np.sqrt(size_ydata[0] * size_ydata[1])
    k_lamda = np.array([-3, -2, -1, 0, 1, 2, 3])
    if param['channel_initial_dic'] == 'DCT':
        pass
    elif param['channel_initial_dic'] == 'Random':
        pass
    else:
        param['lamda1'] = param['lamda1']
        param['lamda2'] = param['lamda2']
    Q, unused = np.linalg.qr(np.random.randn(
        param['D_size'][1], param['Reduced_dims']))
    P = np.dot(Q, Q.T)
    param['P'] = P
    param['initial_ortho_projection'] = Q
    param['N'] = param['D'].shape[0]
    param['max_iter'] = 20
    param['verbose'] = 1
    param['targetMSE'] = 1e-6
    param['lambda3'] = 1e-2
    struct = dict()
    struct['mode'] = 2
    struct['lambda'] = param['lamda1']
    struct['lambda2'] = param['lamda2']
    struct['L'] = np.floor(0.9 * param['N'] * 10)
    param['paramLasso'] = struct

    return param


def DicInit(out, batch_y, outtest, batch_ytest):
    print('out.shape', out.shape)
    D_1st, TrainingVector, TestingVector, labelvector_train, labelvector_test, param = initD(out, batch_y, outtest,
                                                                                             batch_ytest)

    print("Dinit:", labelvector_train.shape,
          labelvector_test.shape, batch_y.shape, batch_ytest)
    number_perclass_dict = param['number_perclass_dict']
    number_class = len(set(batch_y.detach().numpy()))
    number_perclass_trainDinit = param['perclass_trainDinit']
    number_perclass_test = 0
    TrainingVector = TrainingVector.astype(float)
    TestingVector = TestingVector.astype(float)

    param['classes'] = number_class
    param['Reduced_dims'] = param['classes'] - 1
    NumOfTraining = number_perclass_dict
    NumOfClass = number_class
    DistMeasure = 'mindist'

    dim_dl = np.size(TrainingVector, 0)
    print('Numofclass', NumOfClass)
    Accuracy_directLDA, MatrixProjectionV = directldaFreeTest(TrainingVector, TestingVector, labelvector_train,
                                                              labelvector_test, NumOfTraining, NumOfClass, dim_dl,
                                                              DistMeasure)
    print('Accuracy_directLDA_orifea', Accuracy_directLDA,
          MatrixProjectionV.shape)  # (512,14)

    # clf = LinearDiscriminantAnalysis()
    # clf.fit(train_prepare.T, labelvector_train.T)
    # ma = clf.transform(TestingVector.T)
    # print('ma.shape', ma.shape)
    # score = clf.score(TestingVector.T, labelvector_test.T)
    # print('score', score)

    # knn
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(TrainingVector.T, labelvector_train)  # 15000, 999
    svmscore = clf.score(TestingVector.T, labelvector_test)

    clf = LinearDiscriminantAnalysis()
    clf.fit(TrainingVector.T, labelvector_train)
    ldascore = clf.score(TestingVector.T, labelvector_test)

    neigh = KNeighborsClassifier()
    neigh.fit(TrainingVector.T, labelvector_train)
    knnscore = neigh.score(TestingVector.T, labelvector_test)
    print('orifea_KNN score', svmscore, 'orifea_KNN score',
          knnscore, 'orifea_LDA score', ldascore)

    from sklearn.cluster import KMeans
    TrainingVectorplus = KMeans(
        n_clusters=args.number_class, random_state=0).fit(TrainingVector.T)
    TestingVectorplus = KMeans(
        n_clusters=args.number_class, random_state=0).fit(TestingVector.T)

    D_1st = D_1st.astype(float)
    param['D_size'] = D_1st.shape
    param['D'] = D_1st
    param['labels_dic'] = np.zeros([1, D_1st.shape[1]])
    param = init_parameters(param)
    for i in range(number_class):
        param['labels_dic'][0, (i * number_perclass_dict + 1)                            :((i + 1) * number_perclass_dict)] = i  #

    param['lamda1'] = param['lamda1']
    param['lamda2'] = param['lamda2']
    struct = dict()
    struct['mode'] = 2
    struct['lambda'] = param['lamda1']
    struct['lambda2'] = param['lamda2']
    struct['L'] = np.floor(0.9 * param['N'] * 10)
    param['paramLasso'] = struct
    param3 = {'D': D_1st.astype(float),
              'lambda1': param['paramLasso']['lambda'],
              'lambda2': param['paramLasso']['lambda2'],
              'mode': param['paramLasso']['mode'],
              'L': int(param['paramLasso']['L']),
              }
    X = np.asfortranarray(TrainingVector.astype(float))
    train_DL = spams.lasso(X, **param3)

    X = np.asfortranarray(TestingVector.astype(float))
    test_DL = spams.lasso(X, **param3)

    train_DL = np.array(train_DL.todense())
    test_DL = np.array(test_DL.todense())

    # net_array = {
    #     'train_DL': train_DL.T,
    #     'label_train_DL': labelvector_train,
    #     'test_DL': test_DL.T,
    #     'label_test_DL': labelvector_test
    # }
    # D_init_Phi_array_name = 'D_init_Phi_array-' + str(args.model) + '-' + str(args.dataset) + '-' + str(
    #     args.lamda1) + '-' + str(
    #     args.number_perclass_dict) + '-' + str(args.perclass_trainDinit) + '-' + str(
    #     args.number_perclass_trainUDP) + '.npy'
    # np.save(D_init_Phi_array_name, net_array)
    # from visual_MDS import v_MDS
    # from visual_PCA import v_PCA
    # from visual_TSNE import v_3D_TSNE, v_2D_TSNE
    # v_2D_TSNE(train_DL.T, out, labelvector_train, '2D-DL-PIE')

    # spams.lasso ==> Phi
    print('D.shape', D_1st.shape, 'train_Phi.shape',
          train_DL.shape, 'test_Phi.shape', test_DL.shape)
    sparsity = np.zeros([2, 1])
    sparsity[0] = np.argwhere(train_DL != 0).shape[0]
    print(param['paramLasso']['lambda'], 'sparsity', np.argwhere(
        train_DL != 0).shape[0] / train_DL.size)  # shape[1])

    Accuracy_directLDA, MatrixProjectionV = directldaFreeTest(train_DL, test_DL, labelvector_train,
                                                              labelvector_test, NumOfTraining, NumOfClass,
                                                              train_DL.shape[0],
                                                              DistMeasure)

    print('Accuracy_Dinit_LDA_initD', Accuracy_directLDA,
          MatrixProjectionV.shape)  # (512,14)

    train_extra = np.dot(MatrixProjectionV.T, train_DL)
    test_extra = np.dot(MatrixProjectionV.T, test_DL)

    Accuracy_directLDA, _ = directldaFreeTest(train_extra, test_extra, labelvector_train,
                                              labelvector_test, NumOfTraining, NumOfClass,
                                              train_extra.shape[0],
                                              DistMeasure)
    print('Accuracy_Dinit_LDA__extra_initD', Accuracy_directLDA,
          MatrixProjectionV.shape)  # (512,14)

    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(train_extra.T, labelvector_train)
    svmscore = clf.score(test_extra.T, labelvector_test)

    clf = LinearDiscriminantAnalysis()
    clf.fit(train_extra.T, labelvector_train)
    ldascore = clf.score(test_extra.T, labelvector_test)

    neigh = KNeighborsClassifier()
    neigh.fit(train_extra.T, labelvector_train)
    knnscore = neigh.score(test_extra.T, labelvector_test)
    print('Dinit_SVM score', svmscore, 'Dinit_KNN score',
          knnscore, 'Dinit_LDA score', ldascore)

    y_pred = clf.predict(test_extra.T)
    print(classification_report(labelvector_test, y_pred, digits=4))
    print(confusion_matrix(labelvector_test,
          y_pred, labels=range(args.number_class)))
    print('---')

    param['regularizations'] = 'There many regularizations'
    param['inco'] = 'Dinctionary is incoherent'
    param['null_space'] = 'Don not skip the null space'
    param['changes_dic'] = 'Near to original dictionary'
    param['fisher_style'] = 'Standard_LDA'
    param['lambda_changes_dic'] = 5 * (1e-2)
    Dsize = param['D'].shape
    param['D_thresh'] = (Dsize[1] - Dsize[0]) / (Dsize[0] * (Dsize[1] - 1))
    param['IncoDic'] = 2.5 * (1e-4)
    param['t'] = 0
    param['proj'] = MatrixProjectionV
    param['P'] = np.dot(MatrixProjectionV, MatrixProjectionV.T)
    param['initial_ortho_projection'] = MatrixProjectionV
    param['Reduced_dims'] = param['initial_ortho_projection'].shape[1]
    param['NumOfTraining'] = number_perclass_trainDinit
    param['NumOfClass'] = number_class
    param['NumPerClass'] = param['NumOfTraining'] + \
        number_perclass_test  # number_perclass_trainDinit+0
    DistMeasure = 'mindist'
    param['DistMeasure'] = DistMeasure
    param['max_iter'] = 10
    param['mu_W'] = 0.0
    param['lambda_changes_dic'] = 0
    param['IncoDic'] = 0

    return param


def block_centerring_matrix(number_block, number_each_block):
    block_matrix = np.eye(number_each_block) - 1 / number_each_block * np.ones([number_each_block, 1]) @ np.ones(
        [number_each_block, 1]).T
    W = np.zeros([1, 1])
    for i in range(number_block):
        W = block_diag(W, block_matrix)
    print(W.shape)

    return W[1:, 1:]


def block_betweenin_matrix(number_blocks, number_each_block):
    block_matrix_sides = 1 / \
        np.sqrt(number_each_block) * np.ones([number_each_block, 1])
    W_sides = np.zeros([1, 1])
    for i in range(number_blocks):
        W_sides = block_diag(W_sides, block_matrix_sides)
    block_matrix_mid = np.eye(number_blocks) - 1 / number_blocks * np.ones([number_blocks, 1]) @ np.ones(
        [number_blocks, 1]).T
    W_sides = W_sides[1:, 1:]
    W = W_sides @ block_matrix_mid @ W_sides.T
    return W


def FuncValue(param):
    print('funcValue-f-', param['Phi'].shape, param['B'].shape,
          param['Phi'].T.shape, param['P'].shape, param['W'].shape)
    f = -np.trace(param['Phi'] @ param['B'] @ param['Phi'].T @ param['P']) / np.trace(
        param['Phi'] @ param['W'] @ param['Phi'].T @ param['P'])
    # print(f)
    if param['inco'] == 'Dinctionary is incoherent':  # True
        Dsize = np.shape(param['D'])
        Gram = param['D'].T @ param['D']
        Gram_sign = np.sign(Gram)
        delta_index_size = Dsize[1]
        mu = param['D_thresh']
        I = np.eye(delta_index_size)
        for jj in range(delta_index_size):
            for kk in range(delta_index_size):
                if jj != kk:
                    if np.abs(Gram[jj, kk]) <= mu:
                        I[jj, kk] = Gram[jj, kk]
                    else:
                        I[jj, kk] = Gram_sign[jj, kk] * mu
        temp_incoDic = 0.5 * param['IncoDic'] * \
            np.trace((Gram - I) @ (Gram - I).T)
        f = f + temp_incoDic

    if param['null_space'] == 'Skip the null space':
        pass

    if param['changes_dic'] == 'Near to original dictionary':
        temp_nearDic = param['lambda_changes_dic'] * np.trace(
            (param['D'] - param['D00']) @ (param['D'] - param['D00']).T)
        f = f + temp_nearDic

    return f


def scattermat(data, Y):
    l = np.shape(data)[1]
    clases = np.unique(Y)
    tot_clases = np.shape(clases)[0]
    B = np.zeros([l, l])
    W = np.zeros([l, l])
    T = np.zeros([l, l])
    overallmean = np.mean(data, axis=0)
    for i in range(tot_clases):
        clasei = np.argwhere(Y == clases[i])[:, 0]
        xi = data[clasei, :]
        mci = np.mean(xi, axis=0)
        xi = xi - np.tile(mci, (np.shape(clasei)[0], 1))
        W = W + xi.T @ xi
        B = B + np.shape(clasei)[0] * \
            (mci - overallmean).T @ (mci - overallmean)

    x = data - np.tile(overallmean, (np.shape(Y)[0], 1))
    T = T + x.T @ x

    return B, W, T


def P_diff_param(Phi, B, W, P, param):
    denominator = np.trace(Phi @ W @ Phi.T @ P)
    nominator = np.trace(Phi @ B @ Phi.T @ P)
    param['Penalty_sum'] = 1
    diff_result = param['Penalty_sum'] * 1 / denominator * (
        Phi @ B.T @ Phi.T - nominator / denominator * Phi @ W.T @ Phi.T)

    return diff_result


def uqr(A):
    q, r = np.linalg.qr(A)
    d = np.diag(r)
    Id = np.argsort(d)
    d = np.sort(d)
    d = np.sum(d < 0)
    e = np.eye(np.shape(A)[0])

    for i in range(d):
        e[Id[i], Id[i]] = -1

    q = q @ e
    r = e @ r
    return q, r


def exp_mapping_Grassm(X, U, t, param):
    OMG = param['P_skew_egrad']
    UPT, r = uqr(np.eye(np.shape(X)[0]) + t * OMG)
    Y = UPT @ X @ UPT.T
    proj = UPT @ param['proj']

    return Y, proj


def Update_P_SGodl(X, Phi, B, W, D, P, P_grad, param, f0):
    Phi_SIZE = np.shape(Phi)
    val = 0
    alpha = 1e-2
    iter_D = 0
    max_iter_D = 10
    dx_P = -P_grad
    val = val - np.sum(P_grad ** 2)
    f_c = f0
    if param['it'] == 1 and param['t'] == 0:
        t = 1 / np.linalg.norm(dx_P)
    else:
        t = param['t_p']
    P0 = P
    D0 = D
    beta = np.array([0.9, 0.8, 0.5, 0.5, 0.5])
    Norm_dx_P = np.sqrt(np.sum(dx_P ** 2, axis=0))
    while (iter_D == 0) or (f0 >= f_c + alpha * t * val) and (iter_D < max_iter_D):
        display1 = f_c + alpha * t * val
        temp = np.abs(f0 - display1)
        if (temp / np.abs(display1)) > 1000:
            t = t * beta[2]
        elif iter_D == 0:
            t = t
        elif (temp / np.abs(display1)) > 10:
            t = t * beta[1]
        elif (temp / np.abs(display1)) > 0.1:
            t = t * beta[0]
        else:
            t = t * beta[0]
        P, proj = exp_mapping_Grassm(P0, dx_P, t, param)
        param['P'] = P
        if param['regularizations'] == 'There many regularizations':
            f0 = FuncValue(param)
        else:
            f0 = -np.trace(Phi @ B @ Phi.T @ P) / np.trace(Phi @ W @ Phi.T @ P)
        iter_D = iter_D + 1
        if param['verbose']:
            sparsity = np.shape(np.argwhere(Phi != 0)[:, 0])[
                0] / np.shape(Phi)[1]

    param['proj'] = proj
    t = t / (beta[0] ** 2)
    param['t_p'] = t

    return P, Phi, param, f0


def InversK(D_delta, lamda2):
    Dsize = np.shape(D_delta)
    identity_matrix = np.eye(Dsize[1])
    temp = D_delta.T @ D_delta + lamda2 * identity_matrix
    InversValue = np.linalg.inv(temp)

    return InversValue


def D_diff(X, Phi, B, W, P, param):
    D = param['D']
    Dsize = np.shape(D)
    D_egrad = np.zeros(Dsize)
    Phi_size = np.shape(Phi)
    denominator = np.trace(Phi @ W @ Phi.T @ P)
    nominator = np.trace(Phi @ B @ Phi.T @ P)
    U = P @ Phi @ B
    V = P @ Phi @ W
    mu = param['D_thresh']
    if param['null_space'] == 'Skip the null space':
        pass
    for i in range(Phi_size[1] - 1):
        delta_index = np.argwhere(Phi[:, i] != 0)[:, 0]
        delta_index_size = np.shape(delta_index)[0]
        K_inverse = InversK(D[:, delta_index], param['lamda2'])
        S = np.zeros((1, delta_index_size))
        for j in range(delta_index_size):
            if Phi[delta_index[j], i] > 0:
                S[0, j] = 1
            else:
                S[0, j] = -1
        same_bracket = np.dot(D[:, delta_index].T, X[:, i]
                              [:, np.newaxis]) - param['lamda1'] * S.T
        T_U = np.dot(same_bracket, U[delta_index, i][:, np.newaxis].T) + np.dot(U[delta_index, i][:, np.newaxis],
                                                                                same_bracket.T)
        Firstpart = X[:, i][:, np.newaxis] @ U[delta_index, i][:, np.newaxis].T @ K_inverse - D[:,
                                                                                                delta_index] @ K_inverse @ T_U @ K_inverse
        T_V = same_bracket @ V[delta_index, i][:, np.newaxis].T + \
            V[delta_index, i][:, np.newaxis] @ same_bracket.T

        Secondpart = np.dot(X[:, i][:, np.newaxis], V[delta_index, i][:, np.newaxis].T) @ K_inverse - D[:,
                                                                                                        delta_index] @ K_inverse @ T_V @ K_inverse
        D_egrad[:, delta_index] = D_egrad[:, delta_index] + 2 * (
            1 / denominator * Firstpart - nominator / denominator ** 2 * Secondpart)
        if param['null_space'] == 'Skip the null space':
            pass

    if param['changes_dic'] == 'Near to original dictionary':
        D_egrad = D_egrad - \
            param['lambda_changes_dic'] * 2 * (D - param['D00'])
    if param['inco'] == 'Dinctionary is incoherent':
        Gram = D.T @ D
        Gram_sign = np.sign(Gram)
        I = np.eye(Dsize[1])
        for jj in range(Dsize[1]):
            for kk in range(Dsize[1]):
                if jj != kk:
                    if np.abs(Gram[jj, kk]) <= mu:
                        I[jj, kk] = Gram[jj, kk]
                    else:
                        I[jj, kk] = Gram_sign[jj, kk] * mu
        D_egrad = D_egrad - param['IncoDic'] * 4 * D @ (Gram - I)

    return D_egrad


def exp_mapping_sphere(D, dX, t, Norm_dx, sel):
    D[:, sel] = np.multiply(D[:, sel], np.cos(t * Norm_dx[np.newaxis, :][:, sel])) + np.multiply(dX[:, sel], (
        np.sin(t * Norm_dx[np.newaxis, :][:, sel]) / Norm_dx[np.newaxis, :][:, sel]))
    D = np.multiply(D, 1 / np.sqrt(np.sum(D ** 2, axis=0)))

    return D


def Update_D_SGodl(X, Phi, B, W, D, D_grad, P, param, f0):
    alpha = 1e-2
    iter_D = 0
    max_iter_D = 10
    dx = -D_grad
    val = -np.sum(D_grad ** 2)
    f_c = f0
    if param['it'] == 1 and param['t'] == 0:
        t = 1 / np.linalg.norm(dx)
    else:
        t = param['t_D']
    D0 = D
    beta = [0.9, 0.8, 0.5, 0.5, 0.5]
    Norm_dx = np.sqrt(np.sum(dx ** 2, axis=0))
    sel = Norm_dx > 0
    while (iter_D == 0) or (f0 >= f_c + alpha * t * val) and (iter_D < max_iter_D):
        display1 = f_c + alpha * t * val
        temp = np.abs(f0 - display1)
        if ((temp / np.abs(display1)) > 1000):
            t = t * beta[2]
        elif iter_D == 0:
            t = t
        elif ((temp / np.abs(display1)) > 10):
            t = t * beta[1]
        elif ((temp / np.abs(display1)) > 0.1):
            t = t * beta[0]
        else:
            t = t * beta[0]

        D = exp_mapping_sphere(D0, dx, t, Norm_dx, sel)
        param['D'] = D
        if param['SR_update'] == 'Closed form':
            pass
        else:
            param5 = {'D': D.astype(float),
                      'lambda1': param['paramLasso']['lambda'],
                      'lambda2': param['paramLasso']['lambda2'],
                      'mode': param['paramLasso']['mode'],
                      'L': int(param['paramLasso']['L']),
                      }
            Phi = spams.lasso(X, **param5)
            Phi = np.array(Phi.todense())
            param['Phi'] = Phi

        if param['regularizations'] == 'There many regularizations':
            f0 = FuncValue(param)
        else:
            f0 = -np.trace(Phi @ B @ Phi.T @ P) / np.trace(Phi @ W @ Phi.T @ P)

        iter_D = iter_D + 1
        if param['verbose']:
            sparsity = np.shape(np.argwhere(Phi != 0)[:, 0])[
                0] / np.shape(Phi)[1]
    t = t / (beta[0] ** 2)
    param['t_D'] = t

    return D, Phi, param, f0


def func_loss_my(out, batch_y, param):
    batch_y = batch_y.cpu().detach().numpy().astype(float)
    X = out.cpu().detach().numpy().astype(float).T
    mu_W = param['mu_W']
    D = np.multiply(param['D'], 1 / np.sqrt(np.sum(param['D'] ** 2, axis=0)))
    param['D00'] = D
    param['D'] = D
    P = param['P']
    sparsity = np.zeros([param['max_iter'] + 2, 1])
    Error_Rec_DL = np.ones([param['max_iter'] + 2, 1])
    number_class = param['classes']
    number_perclass_trainUDP = args.number_perclass_trainUDP
    param['t_p'] = 0
    param['t_D'] = 0
    Total_trainingNumber = batch_y.shape[0]
    W = block_centerring_matrix(number_class, number_perclass_trainUDP)
    B = block_betweenin_matrix(number_class, number_perclass_trainUDP)
    W_D = block_centerring_matrix(number_class, param['number_perclass_dict'])
    if mu_W != 0:
        W_D = W_D + mu_W * np.eye(np.shape(W_D))
        W = W + mu_W * np.eye(np.shape(W))
    B_D = block_betweenin_matrix(number_class, param['number_perclass_dict']
    if param['fisher_style'] == 'Direct_LDA':
        W = block_centerring_matrix(1, number_perclass_trainUDP * number_class)
        W_D = block_centerring_matrix(
            1, param['number_perclass_dict'] * number_class)
    param['B'] = B
    param['W'] = W
    param['B_D'] = B_D
    param['W_D'] = W_D
    Func_value = np.zeros([1, param['max_iter'] + 1])
    param['numerator'] = np.zeros([1, param['max_iter'] + 1])
    param['denumerator'] = np.zeros([1, param['max_iter'] + 1])
    param['Gramvalue_dic'] = np.zeros([1, param['max_iter'] + 1])
    param['max_iter'] = 1

    for k in range(param['max_iter']):
        print('---  iteration %d, learning with data set' % (k))
        if k == 0:
            param5 = {'D': param['D'].astype(float),
                      'lambda1': param['paramLasso']['lambda'],
                      'lambda2': param['paramLasso']['lambda2'],
                      'mode': param['paramLasso']['mode'],
                      'L': int(param['paramLasso']['L']),
                      }
            Phi = spams.lasso(X, **param5)
            Phi = np.array(Phi.todense())

            param['Phi'] = Phi
            temp_sum = (X - D @ Phi) ** 2
            Error_Rec_DL[0, 0] = np.sum(temp_sum)
            sparsity[0, 0] = np.argwhere(Phi != 0).shape[0]
            if param['regularizations'] == 'There many regularizations':
                f0 = FuncValue(param)
                param['f0'] = f0
                print(f0)
            else:
                f0 = -np.trace(Phi @ B @ Phi.T @ P) / \
                    np.trace(Phi @ W @ Phi.T @ P)
                param['f0'] = f0

            Func_value[0] = f0
            Gram = param['D'].T @ param['D']
            I_D = np.eye(np.shape(Gram)[1])
            Gram = np.abs(Gram - I_D)
            param['Gramvalue_dic'][0, 0] = np.sum(Gram)

        param['SR_update'] = 'Learning via lasso'
        param['it'] = k
        param['Penalty_sum'] = 1
        P_egrad = -P_diff_param(Phi, B, W, P, param)
        P_skew = P @ P_egrad - P_egrad @ P
        P_grad = P @ P_skew - P_skew @ P
        param['it'] = k
        param['P_eGrad'] = P_egrad
        param['P_skew_egrad'] = P_skew
        param['SR_update'] = 'Learning via lasso'

        P_c, Phi, param, f0 = Update_P_SGodl(
            X, Phi, B, W, D, P, P_grad, param, f0)
        param['f0'] = f0

        param['P'] = P_c
        P = param['P']

        D_egrad = -D_diff(X, Phi, B, W, P, param)
        D_grad = D_egrad - D @ np.diag(np.diag(D.T @ D_egrad))

        D_c, Phi, param, f0 = Update_D_SGodl(
            X, Phi, B, W, D, D_grad, P, param, f0)
        param['f0'] = f0
        print('f0')

        param['D'] = D_c
        param['Phi'] = Phi
        P = P_c
        D = D_c
        param['D'] = D
        param['P'] = P

    denominator = np.trace(param['Phi'] @ param['B']
                           @ param['Phi'].T @ param['P'])
    nominator = np.trace(param['Phi'] @ param['W'] @
                         param['Phi'].T @ param['P'])
    X_egrad = np.zeros(X.shape)  # ()

    Phi_size = np.shape(param['Phi'])
    D = param['D']
    for i in range(Phi_size[1]):
        delta_index = np.argwhere(param['Phi'][:, i] != 0)[:, 0]
        UW = param['P'] @ param['Phi'] @ param['W']
        UB = param['P'] @ param['Phi'] @ param['B']
        D_delta = param['D'][:, delta_index]
        Dsize = np.shape(D_delta)
        identity_matrix = np.eye(Dsize[1])
        temp = D_delta.T @ D_delta + param['lamda2'] * identity_matrix
        K_Inverse = np.linalg.inv(temp)
        DW = D[:, delta_index] @ K_Inverse @ UW[delta_index, :]
        DB = D[:, delta_index] @ K_Inverse @ UB[delta_index, :]
        f = ((DW / denominator) - (nominator * DB) / (denominator ** 2)) * 2
        X_egrad = X_egrad + f
    return torch.tensor(X_egrad.T, dtype=torch.float), param


def exp_mapping_GrassmX(X, X_egrad):
    UPT, r = uqr(np.eye(np.shape(X)[0]) + X_egrad)
    Y = UPT @ X @ UPT.T
    return Y
# sddssss


def testacc(featuretrain, featuretest, batch_ytrain, batch_ytest, param):
    MatrixProjectionV0 = param['proj']
    batch_ytrain = batch_ytrain.cpu().detach().numpy().astype(float)
    batch_ytest = batch_ytest.cpu().detach().numpy().astype(float)

    featuretrain = featuretrain.cpu().detach().numpy().astype(float).T
    featuretest = featuretest.cpu().detach().numpy().astype(float).T

    param6 = {'D': param['D'].astype(float),
              'lambda1': param['paramLasso']['lambda'],
              'lambda2': param['paramLasso']['lambda2'],
              'mode': param['paramLasso']['mode'],
              'L': int(param['paramLasso']['L']),
              }

    Phitrain = spams.lasso(featuretrain, **param6)
    Phitrain = np.array(Phitrain.todense())
    Phitest = spams.lasso(featuretest, **param6)
    Phitest = np.array(Phitest.todense())
    sparsity = np.argwhere(Phitrain != 0).shape[0]
    printsparsity = sparsity / Phitrain.size
    print('sparsity', printsparsity)
    TrainingVector_ODL = Phitrain.T
    TestingVector_ODL = Phitest.T
    dim_odl = np.shape(Phitrain)[0]

    Accuracy_directLDA, MatrixProjectionV = directldaFreeTest(Phitrain, Phitest, batch_ytrain,
                                                              batch_ytest, param['NumOfTraining'], param['NumOfClass'],
                                                              dim_odl, param['DistMeasure'])
    print('testacc_Accuracy_directLDA_trainD',
          Accuracy_directLDA, MatrixProjectionV.shape)

    train_extra04 = np.dot(MatrixProjectionV0.T, Phitrain)
    test_extra04 = np.dot(MatrixProjectionV0.T, Phitest)
    Accuracy_directLDA, MatrixProjectionV0 = directldaFreeTest(train_extra04, test_extra04, batch_ytrain,
                                                               batch_ytest, param['NumOfTraining'], param['NumOfClass'],
                                                               train_extra04.shape[0], param['DistMeasure'])
    print('testacc_Accuracy_extra_trainD',
          Accuracy_directLDA, MatrixProjectionV.shape)

    train_extra04 = np.dot(MatrixProjectionV.T, Phitrain)
    test_extra04 = np.dot(MatrixProjectionV.T, Phitest)
    # lda
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_extra04.T, batch_ytrain)
    ldascore = clf.score(test_extra04.T, batch_ytest)
    print('.lda')
    # knn
    neigh = KNeighborsClassifier()
    neigh.fit(train_extra04.T, batch_ytrain)
    knnscore = neigh.score(test_extra04.T, batch_ytest)
    print('.knn')

    # svm
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [
        0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight=None), param_grid)
    clf = clf.fit(train_extra04.T, batch_ytrain)
    y_pred = clf.predict(test_extra04.T)
    print(classification_report(batch_ytest.T, y_pred, digits=4))
    print(confusion_matrix(batch_ytest.T, y_pred, labels=range(args.number_class)))
    svmscore = precision_score(batch_ytest.T, y_pred, average='macro')
    print('DL-svmscore', svmscore)
    print('DL-sklearn_ldascore', ldascore)
    print('DL-knnscore', knnscore)
    print('DL-LDA', Accuracy_directLDA, MatrixProjectionV.shape)

    return svmscore, Accuracy_directLDA, ldascore, knnscore, printsparsity


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    remain_time = step_time * (total - current)
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    # L.append('  Step: %s' % format_time(step_time))
    L.append('  Remain: %s' % format_time(remain_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
