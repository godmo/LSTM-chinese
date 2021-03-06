'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
#############2words################

from __future__ import print_function
from collections import defaultdict
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Bidirectional
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os

starwords = []
names = []

path = "ch/youtuber.txt"
text = open(path,encoding="utf8").read().lower()
print('corpus length:', len(text))
text= text[::-1] #將文章反向
title = open("ch/百家姓前20.txt","r",encoding="utf8")
count=defaultdict(int)

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars)) 
indices_char = dict((i, c) for i, c in enumerate(chars))

for name in title:
        name = name.strip('\n')   
        names.append(name)
        names = list(set(names)) 
for i in text:#計算字頻
    count[i]+=1
sorted_by_value = sorted(count.items(), key=lambda kv: kv[1])

for (cc,num) in sorted_by_value:#過濾開頭字
    if (num>1 and num<28) or (num>28 and cc in names):
        starwords.append((cc))

    
maxlen = 1 
step = 1
sentences = []
next_chars = []
output = []
output2 = []
avgjj = 0

for i in range(0, len(text) - maxlen, step):# range（start,end,scan): scan為跳躍間距
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')#Word2Vec
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: Bi-LSTM
print('Build model...')
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(maxlen, len(chars))))
model.add(Bidirectional(LSTM(128)))
#model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01) #lr : Learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64') #asarray將List轉為array
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    #probas = np.random.multinomial(1, preds, 1)#從有可能的結果中骰一個出來 OUTPUT是矩陣 例如[0,0,0,1,0,0]
    #return np.argmax(probas)#argmax 矩陣中最大值的索引 也就是該字的編號 
    

txt_file=open("2words.txt","w",encoding='utf8')
txt_file2=open("2words反.txt","w",encoding='utf8')

# train the model, output generated text after each iteration
for iteration in range(1, 50):#疊代次數
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              shuffle=False,
              batch_size=128,#每批次處理的數量
              epochs=1#資料讀1輪
             ) 

for z in starwords:
        #sentence = indices_char[z] 
        sentence = z
              
        for diversity in [1.0]:
                print()
                print('----- diversity:', diversity)                 
                print('----- Generating with seed: "' + sentence + '"')
                         
                for i in range(1):#預測的字數
                         x_pred = np.zeros((1, maxlen, len(chars)))
                         for t, char in enumerate(sentence):
                            x_pred[0, t, char_indices[char]] = 1.
                         preds = model.predict(x_pred, verbose=0)[0]
                         
                         next_index = sample(preds, diversity)#jump到副程式
                         #p = np.amax(preds)
                         p1 = preds[next_index] #該次預測的整個機率矩陣
                         p2 = np.where(preds > 0.1) #機率矩陣中大於門檻值的編號
                         next_index = p2[0] # 大於門檻值的元素放到next_index中
                         
                         
                         for j in next_index:
                             next_char = indices_char[j]#找字典對應字詞
                             sys.stdout.write(sentence)
                             sys.stdout.write(next_char)
                             print('Probability:%.3f'%(preds[j]))#顯示預測機率值
                             jjstr=str(preds[j])
                             jjint=preds[j]
                             avgjj = avgjj+jjint #計算平均門檻值
                    
                             if next_char != " " and next_char != "":                                                       
                                 output.append(sentence+next_char+"\n")#正向結果
                                 output2.append(next_char+sentence+"\n")#反向結果需再反向
                                 #output.append(sentence+next_char+" "+jjstr+"\n")
                                 #txt_file.write(sentence+next_char+"\n") 
                                 print("writed ! ")  
                         sentence = next_char#所有可能的next char跑完才換開頭        
               # print('total:',generated)#整串字        
                output = list(set(output))
                output2 = list(set(output2))   
avgjjj = avgjj/len(output)
for w in output:
    txt_file.write(w)
    
for r in output2:
    txt_file2.write(r)


txt_file.close()
txt_file2.close()





