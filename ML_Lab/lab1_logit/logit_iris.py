import csv
import numpy as np 
from sklearn import preprocessing


def data_process():
	file = []
	with open('iris.csv') as cf:
		c_r = csv.reader(cf)
		next(c_r)
		for row in c_r:
			file.append(row)

	# 数据集乱序处理
	file_n = np.array(file)
	np.random.shuffle(file_n)
    
	data = []
	species = []                     # 种类(字符串)
	label = []                       # 种类(数值)
	for line in file_n:
		lineArr = []
		for i in range(4):
			lineArr.append(np.float32(line[i]))
		data.append(lineArr)
		species.append(line[-1])

	# 鸢尾花共有setosa,versiclor,virginica三类，将其类别转换为数字标签
	le = preprocessing.LabelEncoder()
	le.fit(["setosa", "versicolor", "virginica"])
	label = le.transform(species)
	data = np.array(data)

	# 数据集归一化处理
	d_m = data.mean(axis = 0)
	d_s = data.std(axis = 0)
	data = (data - d_m) / d_s


	# 构建训练集、cv集、测试集
	# 训练集
	ytrain = []
	ycv = []
	ytest = []
	Xtrain = data[:90,:]
	ytrain = label[:90]
	# cv集
	Xcv = data[90:120,:]
	ycv = label[90:120]
	# 测试集
	Xtest = data[120:,:]
	ytest = label[120:]

	return Xtrain,ytrain,Xcv,ycv,Xtest,ytest


def h_theta(X,w):
	temp_res = X * w
	temp_res = np.mat(temp_res)
	res = 1 / (1 + np.exp(-temp_res))
	return res


def train():
	X1,y1,X2,y2,X3,y3 = data_process()  #读取相关矩阵
	sp1 = X1.shape                      #sp1为训练集形状
	m1 = sp1[0]                         #m为训练集的数目
	n1 = sp1[1]                         #n为训练集的维度
	y_l = len(y1)                       #y_l为训练集元组的数目
	wt = np.random.rand(3,n1+1)         #初始化权重矩阵, wt -> 3 * 5
	X1 = np.insert(X1,0,1,axis = 1)     #X1为增广矩阵

	# 设置训练参数
	iteration = 100000                  # 迭代次数
	K = 3                               # 分类种类数目
	alpha = 0.01                         # 学习率
	lamb = 1                            # 正则化参数

	# 转化为二分类
	y1 = np.mat(y1)
	y1 = np.transpose(y1)
	Yk = np.zeros((y_l,3))
	Yk[:,[0]] = y1 == 0
	Yk[:,[1]] = y1 == 1
	Yk[:,[2]] = y1 == 2

	for i in range(K):
		for j in range(iteration):
			temp = np.mat(wt[[i],:])
			w_temp = np.transpose(temp)
			res = h_theta(X1,w_temp)
			temp_Yk = np.mat(Yk[:,[i]])
			delta = res - temp_Yk
			wt[[i],[0]] = wt[[i],[0]] - alpha * 1 / m1 * np.sum(delta)
			Xt = np.transpose(X1[:,1:])
			damit = np.transpose(Xt * delta)
			wt[[i],1:] = wt[[i],1:] - alpha * 1 / m1 * (damit + lamb * wt[[i],1:])
    
	# 处理测试集数据
	X3 = np.insert(X3,0,1,axis = 1)     #X3为增广矩阵
	y3 = np.mat(y3)
	y3 = np.transpose(y3)

	return wt,X3,y3


def predict():
	wt = np.mat(np.zeros((3,5)))
	Xtest = np.mat(np.zeros((30,5)))
	ytest = np.mat(np.zeros((30,1)))
	wt,Xtest,ytest = train()
	
	# 计算乘积
	mul = np.mat(np.zeros((30,3)))
	wt_t = np.transpose(wt)
	mul = np.dot(Xtest,wt_t)   # mat形式的矩阵乘法需要使用dot方法
	
    # 取较大的索引为种类
	res = mul.argmax(axis = 1)

	count = 0
	for i in range(len(ytest)):
		if res[i] != ytest[i]:
			count = count + 1
	print("weight=", wt)
	print('测试集共有30个样本，准确率为：', 1-float(count / 30))


if __name__ == '__main__':
    predict()
