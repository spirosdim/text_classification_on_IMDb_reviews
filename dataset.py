
import chazutsu
import pandas as pd 
import numpy as np
import pickle

data = chazutsu.datasets.IMDB().download()

# concatenate the two sets
texts = np.concatenate([data.train_data().review, data.test_data().review])
labels = np.concatenate([data.train_data().polarity, data.test_data().polarity])

# We use a random permutation np array to shuffle the text reviews.
np.random.seed(37)
perm_idx = np.random.permutation(len(texts))

texts = texts[perm_idx]
labels = labels[perm_idx]

col_names = ['label','text']

df = pd.DataFrame({'text':texts, 'label':labels}, columns=col_names)


# storing as Pickle Files
with open("df.pickle", 'wb') as f:
    pickle.dump(df,f)