import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.optimize import linear_sum_assignment


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def view_quality_function(views_features, true_labels, n_clusters):
    view_acc = []
    for features in views_features:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(features)
        p_labels = kmeans.labels_
        acc = cluster_acc(true_labels, p_labels)
        view_acc.append(acc)
    print('test----',view_acc)
    e_acc = np.exp(view_acc - np.max(view_acc))  # 减去最大值以提高数值稳定性
    print('test----',e_acc / e_acc.sum(axis=0))
    
    return e_acc / e_acc.sum(axis=0)


class Hdigit():
    def __init__(self, path,class_num):
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][0][1].T.astype(np.float32)
        self.data = [self.V1, self.V2]
        self.view_acc = view_quality_function(self.data,self.Y,class_num)
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Cifar100():
    def __init__(self, path, class_num):
        data = scipy.io.loadmat(path + 'cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
        self.data = [self.V1, self.V2,self.V3]
        #self.view_acc = view_quality_function(self.data,self.Y,class_num)            
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MNIST_USPS():
    def __init__(self, path,class_num):
        data = scipy.io.loadmat(path + 'MNIST_USPS.mat')
        self.Y = data['Y'].astype(np.int32).reshape(5000,)
        V1 = data['X1'].astype(np.float32)
        V2 = data['X2'].astype(np.float32)
        self.V1 = V1.reshape(5000, 28*28)
        self.V2 = V2.reshape(5000, 28*28)  
        self.data = [self.V1, self.V2]
        self.view_acc = view_quality_function(self.data,self.Y,class_num)        
    def __len__(self):
        return 5000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Fashion():
    def __init__(self, path, class_num):
        data = scipy.io.loadmat(path + 'Fashion.mat')
        self.Y = data['Y'].astype(np.int32).reshape(10000,)
        V1 = data['X1'].astype(np.float32)
        V2 = data['X2'].astype(np.float32)
        V3 = data['X3'].astype(np.float32)
        self.V1 = V1.reshape(10000, 28*28)
        self.V2 = V2.reshape(10000, 28*28) 
        self.V3 = V3.reshape(10000, 28*28)
        self.data = [self.V1, self.V2,self.V3]
        self.view_acc = view_quality_function(self.data,self.Y,class_num)       
        print(type(data))
        for key in data:
             print(f"Key: {key}, Type: {type(data[key])}")  
        print(data["Y"].shape)
        print(V1.shape)
        print(V2.shape)    
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    

class coil20():
    def __init__(self, path,class_num):
        data = np.load(path + 'coil20_train.npz') 
        self.Y = data['labels'].astype(np.int32).reshape(480,)
        V1 = data['view_0'].astype(np.float32)
        V2 = data['view_1'].astype(np.float32)
        V3 = data['view_2'].astype(np.float32)
        self.V1 = V1.reshape(480, 64*64)
        self.V2 = V2.reshape(480, 64*64) 
        self.V3 = V3.reshape(480, 64*64) 
        self.data = [self.V1, self.V2,self.V3]
        self.view_acc = view_quality_function(self.data,self.Y,class_num)      
    def __len__(self):
        return 480
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()




class Sources3():
    def __init__(self, path,class_num):
        data = scipy.io.loadmat(path + '3Sources.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(169,)
        self.V1 = data["data"][0][0].T.astype(np.float32)
        self.V2 = data["data"][0][1].T.astype(np.float32)
        self.V3 = data["data"][0][2].T.astype(np.float32)
        self.data = [self.V1, self.V2,self.V3]
        self.view_acc = view_quality_function(self.data,self.Y,class_num)  
    def __len__(self):
        return 169
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class NUSWIDE(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        # scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)][:, :-1].astype(np.float32))
            # self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)][:, :-1].shape)
            self.dims.append(data['X' + str(i + 1)][:, :-1].shape[1])
        self.data_size = self.multi_view[0].shape[0]   

        self.data = self.multi_view
        self.view_acc = view_quality_function(self.data,self.labels,self.class_num) 


    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class DHA(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)].astype(np.float32))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]
        self.data = self.multi_view
        self.view_acc = view_quality_function(self.data,self.labels,self.class_num)  

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()




def load_data(dataset):
    if dataset == "Hdigit":
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
        dataset = Hdigit('./data/', class_num)
        accs = dataset.view_acc
    if dataset == "Cifar100":
        dims = [512, 2048,1024]
        view = 3
        data_size = 50000
        class_num = 100
        dataset = Cifar100('./data/',class_num)
        #accs = dataset.view_acc
    if dataset == "MNIST-USPS":
        dims = [784, 784]
        view = 2
        data_size = 5000
        class_num = 10
        dataset = MNIST_USPS('./data/',class_num)
        accs = dataset.view_acc
    if dataset == "Fashion":
        dims = [784,784,784]
        view = 3
        data_size = 10000
        class_num = 10
        dataset = Fashion('./data/', class_num)
        accs = dataset.view_acc
    if dataset == "coil20":
        dims = [4096,4096,4096]
        view = 3
        data_size = 480
        class_num = 20
        dataset = coil20('./data/',class_num)
        accs = dataset.view_acc
    if dataset == "Sources3":
        dims = [3560,3631,3068]
        view = 3
        data_size = 169
        class_num = 6
        dataset = Sources3('./data/',class_num)
        accs = dataset.view_acc
    if dataset == "NUSWIDE":
        dataset = NUSWIDE('data/NUSWIDE.mat', view=5)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
        accs = dataset.view_acc
        print('-----',accs)
    if dataset == "DHA":
        dataset = DHA('data/DHA.mat', view=2)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
        accs = dataset.view_acc
    return dataset, dims, view, data_size, class_num

