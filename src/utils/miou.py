import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert(np.max(pred) <= self.nclass)
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == 255:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert(matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall/self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy/self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass)/len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass: #and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m
    
    def f1_score(self):  
        f1_scores = []  
        for i in range(self.nclass):  
            # 避免除以零的情况  
            if (self.M[i, i] == 0) or (np.sum(self.M[:, i]) == 0) or (np.sum(self.M[i, :]) == 0):  
                f1_scores.append(0.0)  
                continue  
              
            precision = self.M[i, i] / np.sum(self.M[i, :])  
            recall = self.M[i, i] / np.sum(self.M[:, i])  
            f1 = 2 * (precision * recall) / (precision + recall)  
            f1_scores.append(f1)  
  
        # 返回平均F1分数和每个类别的F1分数  
        return np.mean(f1_scores), f1_scores  

def get_iou(data_list, class_num, save_path=None):
    if(len(data_list)==0):
        return 
    from multiprocessing import Pool
	
    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    recall = ConfM.recall()
    acc = ConfM.accuracy()
    mean_f1, f1 = ConfM.f1_score()

    classes =np.array(('fake','real'))

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.4f} F1 {:.4f}'.format(i, classes[i], j_list[i],f1[i]))
    print('meanIOU: {:.4f} recall: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(aveJ,recall,acc,mean_f1))

    return aveJ


def get_Acc(data_real,data_pre):
    test_image_labels = data_real
    test_p = data_pre
    f1 = f1_score(test_image_labels, test_p)
    acc = accuracy_score(test_image_labels, test_p)
    precision = precision_score(test_image_labels, test_p)
    recall = recall_score(test_image_labels, test_p)
    print("Image F1 score: {:.4f} Accuracy: {:.4f} Precision: {:.4f} Recall: {:.4f}". format(f1,acc,precision, recall))
    return acc