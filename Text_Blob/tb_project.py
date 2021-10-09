import re 
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Turns the textfile into an iterable array, data[]
def read_data(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            # removes any spaces or specified characters at the start and end of the string
            line = line.strip()
            # seperates the label from the text file as a string, seperated by the character ]
            label = ' '.join(line[1:line.find("]")].strip().split())
            # seperates the text from the file string as a string separated by the character
            text = line[line.find("]")+1:].strip()
            # creates the array data which is returned from the function
            data.append([label, text])
    return data

file = 'training.txt'
data = read_data(file)


def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

def create_feature(text, nrange=(1, 1)):
    text_features = [] 
    #changes the text into lowercase
    text = text.lower() 
    #
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)    
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)

def convert_label(item, name): 
    #splits into individual items based into many lists
    items = list(map(float, item.split()))
    #resets list
    label = ""
    #the length of the items is 7
    for idx in range(len(items)): 
        if items[idx] == 1: 
            label += name[idx] + " "
    
    return label.strip()

emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

X_all = []
y_all = []

for label, text in data:
    #ends up associating an emotion with a line
    y_all.append(convert_label(label, emotions))

    X_all.append(create_feature(text, nrange=(1, 4)))

from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer(sparse = True)
X_train = vectorizer.fit_transform(X_all)

rforest = RandomForestClassifier(random_state=123)
lsvc = LinearSVC(random_state=123)

lsvc.fit(X_train, y_all)
l = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
# sorts the list ascending by default (alphabetical order)
l.sort()
label_freq = {}
for label, _ in data: 
    label_freq[label] = label_freq.get(label, 0) + 1

emoji_dict = {"joy":"😂", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"😒", "shame":"😳", "guilt":"😳"}
t1 = "This looks so impressive"
t2 = "I have a fear of dogs"
t3 = "My dog died yesterday"
t4 = "I don't love you anymore..!"

texts = [t1, t2, t3, t4]
for text in texts: 
    features = create_feature(text, nrange=(1, 4))
    features = vectorizer.transform(features)
    prediction = lsvc.predict(features)[0]
    print( text,emoji_dict[prediction])
