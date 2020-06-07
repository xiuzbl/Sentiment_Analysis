# 逐行读取文件数据进行jieba分词
import os
import jieba
import jieba.analyse
import codecs, sys, string, re
import codecs, sys
import pycantonese as pc

import warnings
warnings.filterwarnings('ignore')

# 文本分词
def prepareData(sourceFile, targetFile):
    f = codecs.open(sourceFile, 'r', encoding='utf-8')
    target = codecs.open(targetFile, 'w', encoding='utf-8')
    print('open source file: ' + sourceFile)
    print('open target file: ' + targetFile)

    lineNum = 1
    line = f.readline()
    while line:
        print('---processing ', lineNum, ' article---')
        line = clearTxt(line)
        seg_line = sent2word(line)
        target.writelines(seg_line + '\n')
        lineNum = lineNum + 1
        line = f.readline()
    print('well done.')
    f.close()
    target.close()

# 清洗文本
def clearTxt(line):
    if line != '':
        line = line.strip()
        intab = ""
        outtab = ""
        trantab = str.maketrans(intab, outtab)
        pun_num = string.punctuation + string.digits
        # # line = line.encode('utf-8')
        # line = line.translate(trantab,pun_num)
        # line = line.decode("utf8")
        # 去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]", "", line)
        # 去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line)
    return line

def save(file_path, init_words_path, tagged_words):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(init_words_path, 'r') as t:
        lines = t.readlines()
        with open(file_path, 'w') as f:
            for word in tagged_words:
                word_freq = freq[word[0]] if word[0] in freq else None
                word_tag = word[1].lower()
                word_tag_matched = bool(re.match('^[a-z]+$', word_tag))
                word_line = word[0]
                if word_freq is not None:
                    word_line = word_line + ' ' + str(word_freq)
                if word_tag_matched is True:
                    word_line = word_line + ' ' + str(word_tag)
                f.write(word_line + '\n')

            for line in lines:
                f.write(line)

# 文本切割
def sent2word(line):
    jieba.load_userdict("cantonese-corpus/data/dict.txt")
    segList = jieba.cut(line, cut_all=False)
    segSentence = ''
    for word in segList:
        if word != '\t':
            segSentence += word + " "
    return segSentence.strip()


def stopWord(sourceFile, targetFile, stopkey):
    sourcef = codecs.open(sourceFile, 'r', encoding='utf-8')
    targetf = codecs.open(targetFile, 'w', encoding='utf-8')
    print('open source file: ' + sourceFile)
    print('open target file: ' + targetFile)
    lineNum = 1
    line = sourcef.readline()
    while line:
        print('---processing ', lineNum, ' article---')
        sentence = delstopword(line, stopkey)
        # print sentence
        targetf.writelines(sentence + '\n')
        lineNum = lineNum + 1
        line = sourcef.readline()
    print('well done.')
    sourcef.close()
    targetf.close()


# 删除停用词
def delstopword(line, stopkey):
    wordList = line.split(' ')
    sentence = ''
    for word in wordList:
        word = word.strip()
        if word not in stopkey:
            if word != '\t':
                sentence += word + " "
    return sentence.strip()



if __name__ == '__main__':
    corpus = pc.hkcancor()
    freq = corpus.word_frequency()
    save('cantonese-corpus/data/dict.txt', 'cantonese-corpus/data/init_dict.txt', corpus.tagged_words())
    sourceFile = 'test_text.txt'
    targetFile = 'text_cut.txt'
    prepareData(sourceFile, targetFile)
    # sourceFile = 'neg_train.txt'
    # targetFile = 'neg_cut.txt'
    # prepareData(sourceFile, targetFile)
    #
    #
    # sourceFile = 'pos_train.txt'
    # targetFile = 'pos_cut.txt'
    # prepareData(sourceFile, targetFile)

    stop_words = pc.stop_words()
    # stopkey = [w.strip() for w in
    #            codecs.open('/Users/gm/Xiu/5014B /酒店评论/data/stopWord.txt', 'r', encoding='utf-8').readlines()]
    sourceFile = 'text_cut.txt'
    targetFile = 'text_cut_stw.txt'
    stopWord(sourceFile, targetFile, stop_words)
    # sourceFile = 'neg_cut.txt'
    # targetFile = 'neg_cut_stw.txt'
    # stopWord(sourceFile, targetFile, stop_words)
    #
    # sourceFile = 'pos_cut.txt'
    # targetFile = 'pos_cut_stw.txt'
    # stopWord(sourceFile, targetFile, stop_words)

