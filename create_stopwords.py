import pickle
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

with open('stopwords.pkl', 'wb') as f:
    pickle.dump(stop_words, f)

print("stopwords.pkl created")