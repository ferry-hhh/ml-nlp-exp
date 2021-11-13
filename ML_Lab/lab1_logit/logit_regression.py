import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 加载数据集
def load(filepath):
    df = pd.read_excel(filepath)
    # 修改：是=>1, 否=>0
    out_good = df["好瓜"]=="是"
    df.loc[out_good, "好瓜"] = 1
    out_bad = df["好瓜"]=="否"
    df.loc[out_bad, "好瓜"] = 0
    # 获得属性值列表
    density = list(df["密度"])
    sugar_content = list(df["含糖率"])
    out_true = list(df["好瓜"])
    return density, sugar_content, out_true


# 对数几率函数
def sigmoid(z):
    y = 1.0 / (1.0 + math.exp(-z))
    return y


# 分类
def classify(y):
    if y > 0.5:
        return 1
    else:
        return 0


# 批量梯度下降法
def BGD(a):
    # 初始化
    w1, w2, b = 0.0, 0.0, 0.0 # 三个参数
    iter_count = 3000 # 迭代次数
    n = len(density) # 数据量大小
    # 循环迭代更新参数
    while iter_count:
        true_T,true_F = 0, 0 # 好瓜预测为好瓜， (真实)好瓜_坏瓜(预测)
        false_T, false_F = 0, 0 # 坏瓜_好瓜， 坏瓜_坏瓜
        w1_temp, w2_temp, b_temp = 0.0, 0.0, 0.0
        for i in range(n):
            z = w1 * density[i] + w2 * sugar_content[i] + b
            out_pred = sigmoid(z)
            # 求混淆矩阵
            if classify(out_pred) == 1:
                if out_true[i] == 1:
                    true_T += 1
                else:
                    false_T += 1
            else:
                if out_true[i] == 0:
                    false_F += 1
                else:
                    true_F += 1

            # 累加
            w1_temp += (out_pred - out_true[i]) * density[i]
            w2_temp += (out_pred - out_true[i]) * sugar_content[i]
            b_temp += out_pred - out_true[i]
        # 更新参数
        w1 -= a * (w1_temp / n)
        w2 -= a * (w2_temp / n)
        b -= a * (b_temp / n)
        iter_count -= 1

    accuracy = (true_T+false_F) / n # 准确率
    error_rate = (true_F + false_T) / n # 错误率
    precision = true_T / (true_T + false_T) # 精确率
    recall = true_T / (true_T + true_F) # 召回率
    print("TP=", true_T, "FP=", false_T, "FN=", true_F, "TN=", false_F)
    print("准确率=", accuracy, "错误率=", error_rate, "精确率=", precision, "召回率=", recall)
    print("模型为：", w1,"* x1 +", w2, "* x2 +", b)
    return w1, w2, b


# 绘制分类图形
def show(w1, w2, b):
    x1, y1 = [], []
    x2, y2 = [], []
    x3, y3 = [], []
    x4, y4 = [], []
    for i in range(len(density)):
        out_pred = sigmoid(w1 * density[i] + w2 * sugar_content[i] + b)
        if classify(out_pred) == 1:
            if out_true[i] == 1:
                x1.append(density[i])
                y1.append(sugar_content[i])
            else:
                x2.append(density[i])
                y2.append(sugar_content[i])
        else:
            if out_true[i] == 0:
                x3.append(density[i])
                y3.append(sugar_content[i])
            else:
                x4.append(density[i])
                y4.append(sugar_content[i])


    plt.scatter(x1, y1, c="green", marker="*", label='TP') # 实际：好瓜 预测：好瓜 (*预测正确)
    plt.scatter(x4, y4, c="green", marker="x", label='FN') # 实际：好瓜 预测：坏瓜 (x预测错误)
    plt.scatter(x3, y3, c="blue", marker="*", label='TN') # 实际：坏瓜 预测：坏瓜
    plt.scatter(x2, y2, c="blue", marker="x", label='FP') # 实际：坏瓜 预测：好瓜
    plt.legend()
    x = np.linspace(0, 1, 100)
    y = -(w1 * x + b) / w2
    plt.plot(x, y, color = "red")
    plt.xlabel('density')
    plt.ylabel('sugar_content')
    plt.show()

if __name__ == '__main__':
    density, sugar_content, out_true = load("./Watermelon_data.xls")
    algha = 0.01
    w1, w2, b = BGD(algha)
    show(w1, w2, b)
