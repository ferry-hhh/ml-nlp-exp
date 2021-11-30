
import re
import jieba

# 待替代符号
punc = r"""\n ！？｡＂＃《》＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏#$%&'()*+-/;<=>?@[\]^_`{|}~“”？！【】（）。’‘……￥"""

# 对语料的预处理
def Pretreatment(file_path):
    intab = "０１２３４５６７８９"
    outtab = "0123456789"
    trantab = str.maketrans(intab,outtab) # 将所有的in字符与对应的out字符建立映射
    f1 = open("../homework1/filter.txt", "w", encoding='utf-8') # 构造过滤后的文件
    for line in open(file_path):
        new_line = re.sub(r" |/t|/n|/m|/v|/u|/a|/w|/q|r|t|/k|/f|/p|n|/c|s|/|d|i|b|l|j|e|v|g|N|V|R|T|y|o|A|D|h|z|x|A|B|M|a|Y|\d{8}-\d{2}-\d{3}-\d{3}", "", line)
        new_line = new_line.translate(trantab) # 将所有的in字符用对应的out字符替代
        f1.write(new_line)
    f1.close()


# 将一句话添加头和尾
def Modify(s):
    #将结尾标点符号截掉
    if s[-1] in (r"[%s]+"%punc):
        s = s[:-1]  # 截取字符串从开头到倒数一个字符的子串
    if s[0] in (r"[%s]+"%punc): # 如果该行开头有待替代符号
        s = s[1:] 
    #添加起始符BOS和终止符EOS  
    s_modify1 = re.sub(r"[%s]+"%punc, "EOS BOS", s)   ## r'\w+'为正则表达式，匹配多个英文单词或者数字  
    s_modify2="BOS"+s_modify1+"EOS"
    return s_modify2


# 构建训练语料库
def BuildTrainCorpus(file_path):
    f = open(file_path, "r", encoding="utf-8")
    text = f.readlines()
    train_list = [] # 存储训练预料分词的列表
    train_dicts = {} # 存储训练预料词汇的词频
    jieba.load_userdict("homework1/dict.txt") # 加载自定义字典

    for line in text:
        line = line.strip('\n') # 去除句末隐藏的换行符
        if line != "": # 避免读取空行
            s_temp = Modify(line) # 为一行语句添加头和尾
            word_list = jieba.lcut(s_temp, cut_all=False)  # 分词
            while ' ' in word_list: # 删除分词后分到的空格
                word_list.remove(' ')
            train_list.extend(word_list)
    f.close()
    # 统计词频
    for word in train_list:
        if word not in train_dicts:
            train_dicts[word] = 1
        else:
            train_dicts[word] += 1
    return train_list, train_dicts


# 构建测试语料
def BuildTestCorpus(sen):
    jieba.load_userdict("homework1/dict.txt") # 加载自定义字典
    sen = sen.strip('\n')
    s_temp = Modify(sen)
    print(s_temp)
    test_list = jieba.lcut(s_temp, cut_all=False) # 分词
    while ' ' in test_list:
        test_list.remove(' ')
    return test_list


# 计算语句概率
def Probability(train_list, train_dicts, test_list):
    count_list=[0]*(len(test_list)-1) # 申请空间
    # 遍历测试的字符串
    for i in range(0, len(test_list)-1):
        # 遍历语料字符串，因为是二元语法，不用比较语料字符串的最后一个字符
        for j in range(0,len(train_list)-1):
            #如果测试的第一个词和语料的第一个词相等则比较第二个词（二元语法）
            if test_list[i]==train_list[j]:
                if test_list[i+1]==train_list[j+1]:
                    count_list[i]+=1
    p = 1
    for i in range(len(count_list)):
        # 使用加法平滑进行数据平滑，防止出现概率为0的情况
        c_ii = float(count_list[i]) + 1
        c_i = float(train_dicts.get(test_list[i], 1)) + len(test_list)
        p *= c_ii / c_i # 概率累乘
    return p


if __name__ == '__main__':
    # 第二次运行可注释
    # Pretreatment("F:\code\NLP_Lab\homework1\训练语料.txt")
    train_list, train_dicts = BuildTrainCorpus("homework1/filter.txt")
    print("BuildTrainCorpus 完成")

    sen1 = "1997年，是中国发展历史上非常重要的很不平凡的一年。"
    sen2 = "相信随着手机付款的普及，这种用糖果代替找零的行为，即便没有相关部门的追究，也会逐渐退出历史舞台了。"
    sen3 = "让党旗在基层一线高高飘扬，必须进一步发挥党员先锋模范作用。"
    sen4 = "明确而言，国家法律对这种擅自拆除承重墙装修的当事人应受到何种行政处罚和民事责任均有规定。"
    test_list1 = BuildTestCorpus(sen1)
    test_list2 = BuildTestCorpus(sen2)
    test_list3 = BuildTestCorpus(sen3)
    test_list4 = BuildTestCorpus(sen4)
    print("BuildTestCorpus 完成")

    p1 = Probability(train_list, train_dicts, test_list1)
    p2 = Probability(train_list, train_dicts, test_list2)
    p3 = Probability(train_list, train_dicts, test_list3)
    p4 = Probability(train_list, train_dicts, test_list4)
    print("sen1 的概率为：", p1)
    print("sen2 的概率为：", p2)
    print("sen3 的概率为：", p3)
    print("sen4 的概率为：", p4)
