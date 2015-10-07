# coding: utf-8
import numpy as np

#accuracyの計算
def calc_accuracy(conf_array):
    #行列のすべての要素の和
    sum_all = np.sum(conf_array)

    #行列の対角成分のみを取り出した配列diagの作成
    diag = np.diag(conf_array)

    #diagのサイズ(つまりラベル数)をsizeと定義
    size = np.size(diag)

    #True Positiveの配列=diag
    TP = diag

    #False Negative, False Positive, True Negativeの配列の初期化
    FN = np.zeros(size)
    FP = np.zeros(size)
    TN = np.zeros(size)

    #sizeの大きさで、すべての成分がsum_allの配列の作成
    sum_array=np.ones(size)*sum_all



    for i in range(size):
        for j in range(size):
            #False Negativeの計算(i行目すべての和)
            FN[i] =conf_array[i,j]+FN[i]

            #False Negativeの計算(i列目すべての和)
            FP[i] =conf_array[j,i]+FP[i]

        #False Negativeの計算(i行目全ての和から、TPの値を引く)
        FN[i]=FN[i]-TP[i]

        #False Positiveの計算(i列目すべての和から、TPの値を引く)
        FP[i]=FP[i]-TP[i]

        #True Negativeの計算(すべての要素の和からTP,TN,FPを引く)
        TN[i]=sum_array[i]-TP[i]-FN[i]-FP[i]

    
    return (TP+TN)/sum_all

#precisionの計算
def calc_precision(conf_array):
    sum_all = np.sum(conf_array)
    diag = np.diag(conf_array)
    size = np.size(diag)
    TP = diag
    FN = np.zeros(size)
    FP = np.zeros(size)
    TN = np.zeros(size)
    sum_array=np.ones(size)*sum_all

    for i in range(0, size):
        for j in range(0, size):
            FN[i] =conf_array[i,j]+FN[i]
            FP[i] =conf_array[j,i]+FP[i]

        FN[i]=FN[i]-TP[i]
        FP[i]=FP[i]-TP[i]
        TN[i]=sum_array[i]-TP[i]-FN[i]-FP[i]
    

    return TP/(TP+FP)

#recallの計算
def calc_recall(conf_array):
    sum_all = np.sum(conf_array)
    diag = np.diag(conf_array)
    size = np.size(diag)
    TP = diag
    FN = np.zeros(size)
    FP = np.zeros(size)
    TN = np.zeros(size)
    sum_array=np.ones(size)*sum_all

    for i in range(0, size):
        for j in range(0, size):
            FN[i] =conf_array[i,j]+FN[i]
            FP[i] =conf_array[j,i]+FP[i]

        FN[i]=FN[i]-TP[i]
        FP[i]=FP[i]-TP[i]
        TN[i]=sum_array[i]-TP[i]-FN[i]-FP[i]
    return TP/(TP+FN)

#F_measureの計算
def calc_f_measure(precisions, recalls):
    return 2*precisions*recalls/(precisions+recalls)




#precisionのリストから適合率の高い上位n個のラベル(添え字)のリストを返す
def calc_top_n_precision(precisions,n):
     return np.argsort(precisions)[::-1][0:n]




#accuracyのリストから適合率の高い上位n個のラベル(添え字)のリストを返す 
def calc_top_n_accuracy(accuracies,n):
    
    return np.argsort(accuracies)[::-1][0:n]

#recallのリストから適合率の高い上位n個のラベル(添え字)のリストを返す
def calc_top_n_recall(recalls,n):
    
    return np.argsort(recalls)[::-1][0:n]



if __name__ == "__main__":
    test_array = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [1, 0, 1, 0],
                           [0, 0, 0, 1]])

    print calc_precision(test_array)

    print calc_accuracy(test_array)
    
    print calc_recall(test_array)


    p = calc_precision(test_array)
    a = calc_accuracy(test_array)
    r = calc_recall(test_array)
    print calc_f_measure(p,r)

    print calc_top_n_precision(p,2)

    print calc_top_n_accuracy(a,2)

    print calc_top_n_recall(r,2)



