from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn

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

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


def inference(loader, model, device, view, data_size):
    model.eval()
    commonZ_fused_high = []
    labels_vector = []
    # zsl = []
    # hsl = []
    ##新加##
    """
    Xs
    Zs
    Hs
    """
    pred_vectors = []
    Xs = []
    Zs = []
    Hs = []
    Ss = []
    Qs = []
    for v in range(view):
        pred_vectors.append([])
        Xs.append([])
        Zs.append([])
        Hs.append([])
        Ss.append([])
        Qs.append([])
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            xrs, zs, hs, qls = model.forward(xs)
            commonz_fused_high, _, _, _,_,_,ss,_ = model.AdaM(xs)
            commonz_fused_high = commonz_fused_high.detach()
            commonZ_fused_high.extend(commonz_fused_high.cpu().detach().numpy())
        for v in range(view):
            zs[v] = zs[v].detach()
            hs[v] = hs[v].detach()
            Xs[v].extend(xs[v].cpu().detach().numpy())
            Zs[v].extend(zs[v].cpu().detach().numpy())
            Hs[v].extend(hs[v].cpu().detach().numpy())
            Ss[v].extend(ss[v].cpu().detach().numpy())
            Qs[v].extend(qls[v].cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    for v in range(view):
        Xs[v] = np.array(Xs[v])
        Zs[v] = np.array(Zs[v])
        Hs[v] = np.array(Hs[v])
        Ss[v] = np.array(Ss[v])
        Qs[v] = np.array(Qs[v])     
        pred_vectors[v] = np.array(pred_vectors[v])
    #return  Xs, [], Zs, labels_vector, Hs
    return labels_vector, Hs, commonZ_fused_high


     
def valid(model, device, dataset, view, data_size, class_num):
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    times_for_K=1.0
    labels_vector, h_vectors, commonZ_fused_high = inference(test_loader, model, device, view, data_size)
    h_clusters = []
    print("Clustering results on each view (H^v):")
    acc_avg, nmi_avg, ari_avg, pur_avg = 0, 0, 0, 0
    for v in range(view):
        kmeans = KMeans(n_clusters=int(class_num * times_for_K), n_init=100)
        if len(labels_vector) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=int(class_num * times_for_K), batch_size=5000, n_init=100)
        y_pred = kmeans.fit_predict(h_vectors[v])
        h_clusters.append(y_pred) 
        nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
        print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
                                                                                    v + 1, nmi,
                                                                                    v + 1, ari,
                                                                                    v + 1, pur))
        acc_avg += acc
        nmi_avg += nmi
        ari_avg += ari
        pur_avg += pur

    print('Mean = {:.4f} Mean = {:.4f} Mean = {:.4f} Mean={:.4f}'.format(acc_avg / view,
                                                                            nmi_avg / view,
                                                                            ari_avg / view,
                                                                            pur_avg / view))
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    y_common_clusters = kmeans.fit_predict(commonZ_fused_high)
    if len(labels_vector) > 10000:
        kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
    h = np.concatenate(h_vectors, axis=1)
    pseudo_label = kmeans.fit_predict(h)
    print("Clustering results on all views ([H^1...H^V]): " + str(labels_vector.shape[0]))
    nmi, ari, acc, pur = evaluate(labels_vector, pseudo_label)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
    return acc, nmi, pur
    
