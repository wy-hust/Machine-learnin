#Naive_Bayes处理enron数据集进行垃圾邮件识别
import numpy as np

import os


class NaiveBayesBase(object):

    def __init__(self):
        pass

    def fit(self, trainMatrix, trainCategory):
        '''
        朴素贝叶斯分类器训练函数，求：p(Ci),基于词汇表的p(w|Ci)
        Args:
            trainMatrix : 训练矩阵，即向量化表示后的文档（词条集合）
            trainCategory : 文档中每个词条的列表标注
        Return:
            p0Vect : 属于0类别的概率向量(p(w1|C0),p(w2|C0),...,p(wn|C0))
            p1Vect : 属于1类别的概率向量(p(w1|C1),p(w2|C1),...,p(wn|C1))
            pAbusive : 属于1类别文档的概率
        '''
        numTrainDocs = len(trainMatrix)
        # 长度为词汇表长度
        numWords = len(trainMatrix[0])
        # p(ci)
        self.pAbusive = sum(trainCategory) / float(numTrainDocs)
        # 由于后期要计算p(w|Ci)=p(w1|Ci)*p(w2|Ci)*...*p(wn|Ci)，若wj未出现，则p(wj|Ci)=0,因此p(w|Ci)=0，这样显然是不对的
        # 故在初始化时，将所有词的出现数初始化为1，分母即出现词条总数初始化为2
        p0Num = np.ones(numWords)
        p1Num = np.ones(numWords)
        p0Denom = 2.0
        p1Denom = 2.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                p1Num += trainMatrix[i]
                p1Denom += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])
        # p(wi | c1)
        # 为了避免下溢出（当所有的p都很小时，再相乘会得到0.0，使用log则会避免得到0.0）

        self.p1Vect = np.log(p1Num / p1Denom)
        # p(wi | c2)
        self.p0Vect = np.log(p0Num / p0Denom)
        return self

    
    def predict(self, testX):
        '''
        朴素贝叶斯分类器
        Args:
            testX : 待分类的文档向量（已转换成array）
            p0Vect : p(w|C0)
            p1Vect : p(w|C1)
            pAbusive : p(C1)
        Return:
            1 : 为垃圾邮件 (基于当前文档的p(w|C1)*p(C1)=log(基于当前文档的p(w|C1))+log(p(C1)))
            0 : 为正常邮件 (基于当前文档的p(w|C0)*p(C0)=log(基于当前文档的p(w|C0))+log(p(C0)))
        '''

        p1 = np.sum(testX * self.p1Vect) + np.log(self.pAbusive)
        p0 = np.sum(testX * self.p0Vect) + np.log(1 - self.pAbusive)
        if p1 > p0:
            return 1
        else:
            return 0


def createVocabList(dataSet):
    '''
    创建所有文档中出现的不重复词汇列表
    Args:
        dataSet: 所有文档
    Return:
        包含所有文档的不重复词列表，即词汇表
    '''
    vocabSet = set([])
    # 创建两个集合的并集
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def read_file(filename):
    f = open(filename, "rb")
    str = f.read()
    list = str.split()
    return list

# 读取训练数据


def load_train_DataSet():
    path_train_ham = "E:\\AIπ2020\\第一轮测试_final\\Naive_Bayes\\enron\\train\\ham\\"
    path_train_spam = "E:\\AIπ2020\\第一轮测试_final\\Naive_Bayes\\enron\\train\\spam\\"
    fileList_ham = os.listdir(path_train_ham)
    fileList_spam = os.listdir(path_train_spam)
    files_ham_num = len(fileList_ham)
    files_spam_num = len(fileList_spam)
    train_ham_List = [None]*files_ham_num
    train_spam_List = [None]*files_spam_num
    count = 0
    for file in fileList_ham:
        train_ham_List[count] = read_file(path_train_ham+file)
        count += 1
        #print("正在读取 第%d条"%(count))

    count = 0
    for file in fileList_spam:
        train_spam_List[count] = read_file(path_train_spam+file)
        count += 1
        #print("正在读取 第%d条 " %(count+13236))
    train_List = train_ham_List+train_spam_List
    classVec = [0]*13236+[1]*13726  # 正常为0，垃圾为1
    return train_List, classVec


def bagOfWords2Vec(vocabList, inputSet):
    '''
    依据词汇表，将输入文本转化成词袋模型词向量
    Args:
        vocabList: 词汇表
        inputSet: 当前输入文档
    Return:
        returnVec: 转换成词向量的文档
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 读取测试数据


def load_test_Dataset():
    path_test_ham = "E:\\AIπ2020\\第一轮测试_final\\Naive_Bayes\\enron\\test\\ham\\"
    path_test_spam = "E:\\AIπ2020\\第一轮测试_final\\Naive_Bayes\\enron\\test\\spam\\"

    fileList_ham = os.listdir(path_test_ham)
    fileList_spam = os.listdir(path_test_spam)
    files_ham_num = len(fileList_ham)
    files_spam_num = len(fileList_spam)
    test_List = [None]*(files_ham_num+files_spam_num)
    count = 0
    for file in fileList_ham:
        test_List[count] = read_file(path_test_ham+file)
        count += 1

    for file in fileList_spam:
        test_List[count] = read_file(path_test_spam+file)
        count += 1

    classVec = [0]*3309+[1]*3431  # 正常为0，垃圾为1
    return test_List, classVec


def runNaiveBayes():
    '''测试'''
    listPosts, listClasses = load_train_DataSet()
    testPosts, testClasses = load_test_Dataset()
    
    # 训练贝叶斯模型
    
    offset_train_ham = int(input("输入训练正常邮件偏移量:"))
    amount_train_ham = int(input("输入训练的正确邮件数量"))
    offset_train_spam = int(input("输入训练错误邮件偏移量:"))
    amount_train_spam = int(input("输入训练的错误邮件数量"))
    
    listTrain = [None]*(amount_train_ham+amount_train_spam)
    ClassTrain = [None]*(amount_train_ham+amount_train_spam)
    count = 0
    for i in range(amount_train_ham):
        listTrain[count]=listPosts[i+offset_train_ham]
        count += 1
    for i in range(amount_train_spam):
        listTrain[count]=listPosts[i+offset_train_spam+13236]
        count += 1

    count = 0
    for i in range(amount_train_ham):
        ClassTrain[count]=listClasses[i+offset_train_ham]
        count += 1
    for i in range(amount_train_spam):
        ClassTrain[count]=listClasses[i+offset_train_spam+13236]
        count += 1
    
    myVocabList = createVocabList(listTrain)
    print("开始训练模型，需要一定时间")
    trainMat = []
    for i in range(amount_train_ham):
        postDoc = listPosts[i+offset_train_ham]
        trainMat.append(bagOfWords2Vec(myVocabList, postDoc))
    i = 0
    for i in range(amount_train_spam):
        postDoc = listPosts[i+offset_train_spam+13236]
        trainMat.append(bagOfWords2Vec(myVocabList, postDoc))
    i = 0
    nb = NaiveBayesBase()
    nb.fit(np.array(trainMat), np.array(ClassTrain))
    print("训练结束\n")
    
    offset_ham = int(input("输入正常测试邮件偏移量:"))
    amount_ham = int(input("输入测试的正确邮件数量"))
    offset_spam = int(input("输入错误测试邮件偏移量:"))
    amount_spam = int(input("输入测试的错误邮件数量"))
    error_count = 0
    # 开始测试
    '''
    测试正确邮件
    '''
    for i in range(amount_ham):
        testarray = np.array(bagOfWords2Vec(
            myVocabList, testPosts[i+offset_ham]))
        test_class = nb.predict(testarray)
        if(test_class == 0):
            test_class_describe = "正确邮件"
        elif(test_class == 1):
            test_class_describe = "垃圾邮件"
        print("第 %d 篇文档, 分类器结果: %d %s, 实际值: %d" 
        % (i, test_class,test_class_describe, testClasses[i+offset_ham]), end=' ')
        if(test_class != testClasses[i+offset_ham]):
            error_count = error_count+1.0
            print("分类错误", end='')
        print('当前错误率：%2.2f%%' % (100.0*error_count/(i+0.0001)))  # 防止i的值为0
    i = 0
    '''
    测试垃圾邮件
    '''
    for i in range(amount_spam):
        testarray = np.array(bagOfWords2Vec(
            myVocabList, testPosts[i+offset_spam+3309]))
        #print(testPosts[i+offset_spam+3309])
        test_class = nb.predict(testarray)
        if(test_class == 0):
            test_class_describe = "正确邮件"
        elif(test_class == 1):
            test_class_describe = "垃圾邮件"
        print("第 %d 篇文档, 分类器结果: %d %s, 实际值: %d" % (i, test_class,test_class_describe
                                                   , testClasses[i+offset_spam+3309]), end=' ')
        if(test_class != testClasses[i+offset_spam+3309]):
            error_count = error_count+1.0
            print("分类错误", end='')
        print('当前错误率：%2.2f%%' % (100.0*error_count/(i+amount_ham+0.0001)))
    print("\n总错误数: %d" % error_count)
    
    print("总错误率: %2.2f%%" % (100.0*error_count/(amount_ham+amount_spam)))


if __name__ == "__main__":
    runNaiveBayes()
