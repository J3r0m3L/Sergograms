import re
from collections import Counter
from sklearn.svm import LinearSVC

#train model
def parse_data(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data   

def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

def create_feature(text, nrange=(1, 1)):
    text_features = [] 
    text = text.lower() 
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)    
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)

def convert_label(item, name): 
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)): 
        if items[idx] == 1: 
            label += name[idx] + " "
    return label.strip()

def train_model():
    file = 'training.txt'
    data = parse_data(file)

    emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
    x_all = []
    y_all = []

    for label, text in data:
        y_all.append(convert_label(label, emotions))
        x_all.append(create_feature(text, nrange=(1, 4)))

    from sklearn.feature_extraction import DictVectorizer
    vectorizer = DictVectorizer(sparse = True)
    x_all = vectorizer.fit_transform(x_all)

    lsvc = LinearSVC(random_state=123)

    lsvc.fit(x_all, y_all)
    return [vectorizer, lsvc]

# utilize model
def main():
    trained = train_model()

    emoji_dict = {"joy":"😀", "fear":"😱", "anger":"😣", "sadness":"😭", "disgust":"🤮", "shame":"😨", "guilt":"🥺"}
    t1 = "This looks so impressive"
    t2 = "I have a fear of dogs"
    t3 = "My dog died yesterday"
    t4 = "I don't love you anymore..!"
    t5 = "I am very angry at you!"

    texts = [t1, t2, t3, t4, t5]
    for text in texts: 
        features = create_feature(text, nrange=(1, 4))
        features = trained[0].transform(features)
        prediction = trained[1].predict(features)[0]
        print( text,emoji_dict[prediction])

main()