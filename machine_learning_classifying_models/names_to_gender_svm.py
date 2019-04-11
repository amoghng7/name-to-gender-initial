# For Preprocessing
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
# For Training
from sklearn.svm import SVC

# For Validating
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

# For Visualization
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pylab as pl
from pandas.plotting import scatter_matrix

import re
import copy

# For feature extraction

def sylco(word) :
        word = word.lower()

        # exception_add are words that need extra syllables
        # exception_del are words that need less syllables

        exception_add = ['serious','crucial']
        exception_del = ['fortunately','unfortunately']

        co_one = ['cool','coach','coat','coal','count','coin','coarse','coup','coif','cook','coign','coiffe','coof','court']
        co_two = ['coapt','coed','coinci']

        pre_one = ['preach']

        syls = 0 #added syllable number
        disc = 0 #discarded syllable number

        #1) if letters < 3 : return 1
        if len(word) <= 3 :
            syls = 1
            return syls

        #2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end.
        # if it has only 1 vowel or 1 set of consecutive vowels, discard. (like "speed", "fled" etc.)

        if word[-2:] == "es" or word[-2:] == "ed" :
            doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]',word))
            if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]',word)) > 1 :
                if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[-3:] == "ies" :
                    pass
                else :
                    disc+=1

        #3) discard trailing "e", except where ending is "le"  

        le_except = ['whole','mobile','pole','male','female','hale','pale','tale','sale','aisle','whale','while']

        if word[-1:] == "e" :
            if word[-2:] == "le" and word not in le_except :
                pass

            else :
                disc+=1

        #4) check if consecutive vowels exists, triplets or pairs, count them as one.

        doubleAndtripple = len(re.findall(r'[eaoui][eaoui]',word))
        tripple = len(re.findall(r'[eaoui][eaoui][eaoui]',word))
        disc+=doubleAndtripple + tripple

        #5) count remaining vowels in word.
        numVowels = len(re.findall(r'[eaoui]',word))

        #6) add one if starts with "mc"
        if word[:2] == "mc" :
            syls+=1

        #7) add one if ends with "y" but is not surrouned by vowel
        if word[-1:] == "y" and word[-2] not in "aeoui" :
            syls +=1

        #8) add one if "y" is surrounded by non-vowels and is not in the last word.

        for i,j in enumerate(word) :
            if j == "y" :
                if (i != 0) and (i != len(word)-1) :
                    if word[i-1] not in "aeoui" and word[i+1] not in "aeoui" :
                        syls+=1

        #9) if starts with "tri-" or "bi-" and is followed by a vowel, add one.

        if word[:3] == "tri" and word[3] in "aeoui" :
            syls+=1

        if word[:2] == "bi" and word[2] in "aeoui" :
            syls+=1

        #10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"

        if word[-3:] == "ian" : 
        #and (word[-4:] != "cian" or word[-4:] != "tian") :
            if word[-4:] == "cian" or word[-4:] == "tian" :
                pass
            else :
                syls+=1

        #11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

        if word[:2] == "co" and word[2] in 'eaoui' :

            if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two :
                syls+=1
            elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one :
                pass
            else :
                syls+=1

        #12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

        if word[:3] == "pre" and word[3] in 'eaoui' :
            if word[:6] in pre_one :
                pass
            else :
                syls+=1

        #13) check for "-n't" and cross match with dictionary to add syllable.

        negative = ["doesn't", "isn't", "shouldn't", "couldn't","wouldn't"]

        if word[-3:] == "n't" :
            if word in negative :
                syls+=1
            else :
                pass   

        #14) Handling the exceptional words.

        if word in exception_del :
            disc+=1

        if word in exception_add :
            syls+=1     

        # calculate the output
        return numVowels - disc + syls

def calc_ratio_of_syllables(name):

    open_syllable = 0
    # print("First Letter: "+name[-1:].lower()+" Second Letter: "+name[0].lower())
    try:
        if name[-1:].lower() in 'aeiou':
            open_syllable += 1
        if name[0].lower() in 'aeiou':
            open_syllable += 1
        total_syllables = sylco(name)

        return open_syllable/total_syllables
    except Exception as e:
        return 0
        
# Scatter Plot

def scatter_plot(X_data, y_data, title="Classification Plot", legend_1='legend-1', legend_2='legend-2', color_1='#76aad3', color_2='#5439a4'):
    X = X_data
    y = y_data
    
    
    pca = PCA(n_components=2).fit(X)
    pca_2d = pca.transform(X)

    for i in range(0, pca_2d.shape[0]):
        if y[i] == 1:
            c1 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c=color_1, marker='o')
        elif y[i] == 0:
            c2 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c=color_2, marker='o')
    
    pl.legend([c1, c2], [legend_1, legend_2])
    pl.title(title)
    pl.figure(figsize=(40,20))
    pl.show()

data = pd.read_csv("indian_name_dataset.csv")
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder(categories='auto')

categorical = ['1-gram', '2-gram', '3-gram', 'vowel', 'sonorants', 'gender']
one_hot_enc = ['vowel','last_letter_vowel','sonorants']

categ = []

for header in categorical:
    categ.append(copy.deepcopy(le.fit(data[header].astype(str))))
    data[header] = le.transform(data[header].astype(str))

data = data.drop(columns=['name', 'last_letter_vowel'])
    
# print(data.tail(20))

X = data.iloc[:,:-1].values
y = data.iloc[:,-1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10}

plt.rc('font', **font)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

data.hist()
plt.figure(figsize = (40,20), dpi=80,)
plt.show()

data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()

data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

scatter_matrix(data)
plt.show()

correlations = data.corr()
names = data.columns
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,7,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

# Fitting Naive Bayes to the Training set
classifier = SVC(random_state = 2)
classifier.fit(X_train, y_train.ravel())

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Accuracy: {0:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print("Precision Score: {0:.2f}%".format(precision_score(y_test,y_pred)*100))
print("Recall Score: {0:.2f}%".format(recall_score(y_test,y_pred)*100))
print("F1 Score: {0:f}".format(f1_score(y_test,y_pred)))

df_cm = pd.DataFrame(cm, index=['Male', 'Female'], columns=['Male', 'Female'])
# plt.figure(figsize = (10,7))
print("\nconfusion Matrix: ")
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}, cmap='Blues', fmt='g')
plt.show()

#scatter_plot(X_train, y_train, 'Gender Classification Training Data', 'male', 'female')
#scatter_plot(X_test, y_test, 'Gender Classification Testing Data', 'male', 'female')
#scatter_plot(X_test, y_pred, 'Gender Classification Predicted Data', 'male', 'female')

# Features for predicting

def get_features(name):
    
    vowels = ['a', 'e', 'i', 'o', 'u']
    sonorants = ['ha', 'mha', 'vha', 'lha', 'rha']
    cat = ['1-gram', '2-gram', '3-gram', 'vowel', 'sonorants']
    
    if name[-3:].lower() in sonorants:
        sono = name[-3:]
    elif  name[-2:].lower()  in  sonorants:
        sono = name[-2:]
    else:
        sono = "na"
        
    ratio = calc_ratio_of_syllables(name)
    
    df = pd.DataFrame(columns=data.columns)
    df = df.drop(columns='gender')
    
    print(df)
    
    if name[-1:] in vowels:
        df.loc[0] = [name[-1:], name[-2:], name[-3:], name[-1:], sono, ratio]
    else:
        df.loc[0] = [name[-1:], name[-2:], name[-3:], 'n', sono, ratio]
        
    count = 0
    
    print(categ)
    
    for header in cat:
        df[header] = categ[count].transform(df[header].astype(str))
        count += 1

    return sc.transform(df.values)

while(True):
    name = input("Enter Name to predict: ")
    
    if name == '0':
        sys.exit()
    new = get_features(name)
    
    predicted = classifier.predict(new)
    
    #print(predicted)
    
    if predicted[0] == 0:
        gender = 'Female'
    else:
        gender = 'Male'
    
    print("%s is %s" %(name, gender))