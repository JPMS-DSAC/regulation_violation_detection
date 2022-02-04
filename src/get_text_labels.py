import json
import pickle

# Load the data 
regs = json.load(open("../data/regViolated.json"))
data = json.load(open("../data/procFiles.json"))
files = json.load(open("../data/files.json"))

# Define the holding lists 
cleanRegs = {} # Cleaned up regulations 
facts = [] # Get the facts of the case 
file_data = {} # Link these to file names 
label_violations = [] # Get the list of violations  with file names 
regs_violated = [] # Get the regulations violated
labelViolationsSet = set() # Get the most common violations

for i in regs:
    cleanReg = []
    for j in regs[i]:
        if j == 'disp':
            continue
        for k in regs[i][j]:
            cleanReg.append(j+'_'+''.join([l for l in k if l != " "]))
    cleanRegs[i] = cleanReg

for i in range(len(data)):
    x = []
    for j in data[i][-1]:
        if j[1] in [0,2,5,6]:
            if len(j[0]) < 300:
                x.append(j[0])
    facts.append(' '.join(x))


for i in range(len(files)):
    file_data[files[i]] = facts[i]


for i in regs:
    if i in file_data:
        label_violations.append([file_data[i],cleanRegs[i]])
    else:
        print(i)

for i in cleanRegs:
    regs_violated += cleanRegs[i]
from collections import Counter
c = Counter(regs_violated)


for i in label_violations:
    for j in i[1]:
        labelViolationsSet.add(j)


(labelViolationsSet) = sorted(list(labelViolationsSet))
most_common_violations = c.most_common()
most_common_violations = [i[0] for i in most_common_violations]

# Create a one hot vector from the list of most common regulations 
def getOneHotVector(reg):
    indicesPresent = []
    for i in reg:
        indexInMostCommonReg = most_common_violations.index(i)
        if indexInMostCommonReg <= 48:
            indicesPresent.append(indexInMostCommonReg)
        else:
            indicesPresent.append(49)
    one_hot_vec = [0 for i in range(50)]
    for i in indicesPresent:
        one_hot_vec[i] = 1
    return one_hot_vec

sentence_labels = [[i[0],getOneHotVector(i[1])] for i in label_violations]

# dump the data into a file 

with open('data.pkl','wb') as f:
    pickle.dump(sentence_labels,f)