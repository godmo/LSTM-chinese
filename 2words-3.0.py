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

#path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
path = "ch/Data/youtuber.txt"
text = open(path,encoding="utf8").read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars)) #char > index (我,1)
indices_char = dict((i, c) for i, c in enumerate(chars))#index > char (1,我)

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 1 
step = 2
sentences = []
next_chars = []
output = []
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

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(maxlen, len(chars))))
model.add(Bidirectional(LSTM(128)))
#model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01) #lr : Learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
#model.save('2words_model.h5')
# Deletes the existing model
#del model  
# Returns a compiled model identical to the previous one
#model = load_model('2words_model.h5')

###############################################################################

def sample(preds, temperature=1.0):#加入隨機性運算 預設1.0
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64') #asarray將List轉為array
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)#從有可能的結果中骰一個出來 OUTPUT是矩陣 例如[0,0,0,1,0,0]
    return np.argmax(probas)#argmax 矩陣中最大值的索引 也就是該字的編號 



def yeepre(sentence):
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.
    preds = model.predict(x_pred, verbose=0)[0]
    p2 = np.where(preds > 0.2) #機率矩陣中大於門檻值的編號
    next_index = p2[0]
    return next_index,preds

###############################################################################
    

starwords = random.sample(list(indices_char) , 200)#隨機於字典挑選50個開頭字
txt_file=open("2words.txt","w",encoding='utf8')
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

    #start_index = random.randint(0, len(text) - maxlen - 1)
for z in starwords:
    sentence = indices_char[z] #數字轉字
              
        #for diversity in [1.0]:
   
                #print('----- diversity:', diversity)
                #generated = sentence+''
                #sentence = text[start_index: start_index + maxlen] # slicing list  ex:A[n:m] n讀取到m-1 
                #sentence = indices_char[724] 
                #generated += sentence 
    print()
    print('----- Generating with seed: "' + sentence + '"')   
        
                         #x_pred = np.zeros((1, maxlen, len(chars)))
                         #for t, char in enumerate(sentence):
                            #x_pred[0, t, char_indices[char]] = 1.
                         #preds = model.predict(x_pred, verbose=0)[0]                     
                         #next_index = sample(preds, diversity)#jump到副程式
                         #p = np.amax(preds)
                         #p1 = preds[next_index]
                         #p2 = np.where(preds > 0.3) #機率矩陣中大於門檻值的編號
                         #next_index = p2[0] # 大於門檻值的元素放到next_index中          
    next_index , preds = yeepre(sentence)#傳進副程式一個字他會給我大於門檻值的下個字
    for k in next_index:
        next_char = indices_char[k]#找字典對應字詞
        
        if next_char != " " and next_char != "":
            sys.stdout.write(sentence)
            sys.stdout.write(next_char)#sys.stdout.write不會換行           
            print('Probability:%.6f'%(preds[k]))
            output.append(sentence+next_char+"\n")                  
            sentence2 = next_char
            next_index2 , preds2 = yeepre(sentence2) 
            
            for j in next_index2:
                next_char = indices_char[j]
                
                if next_char != " " and next_char != "":                 
                    sys.stdout.write(sentence2)
                    sys.stdout.write(next_char)
                    print('Probability:%.6f'%(preds2[j]))
                    output.append(sentence2+next_char+"\n")
                    
                             
               # print('total:',generated)#整串字        
output = list(set(output))
for w in output:
    txt_file.write(w)
print()
print(model.summary())
txt_file.close()
#cmd = '3words_lstm.py'
#os.system(cmd)
