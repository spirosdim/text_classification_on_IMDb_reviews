# Text Classification

A sentiment analysis task of classifying the polarity (positive or negative) of IMDb reviews. Using 'traditional' machine learning.

## Data
**Large Movie Review Dataset.**
This is a dataset for binary sentiment classification. A set of 25,000 highly polar movie reviews for training, and 25,000 for testing. But I will concatenate them to 50,000 examples and then use 40,000 (80%) as training set and 10,000 (20%) as test set. 

The data is provided by [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/) 


To load the data I used the package [chazutsu](https://github.com/chakki-works/chazutsu
). The package manager [pip](https://pip.pypa.io/en/stable/) was used to install it.

```bash
!pip install <name of the library>
```

* Load the dataset using chazutsu package
* Shuffle the dataset using permutation of its index
* Split it to 80% train and 20% test sets using train_test_split from sklearn 

## Take a glimpse of the raw text

* **The dataset is balanced** with 25,000 positive and 25,000 negative reviews

![bl_data](https://user-images.githubusercontent.com/31864574/50576013-b119c600-0e11-11e9-9e79-f86c29094a27.png)


* **Most common words** First using 1-gram and then 2-grams.

![1gram](https://user-images.githubusercontent.com/31864574/50576027-f1794400-0e11-11e9-94af-4119f5ba598d.png)

![2grams](https://user-images.githubusercontent.com/31864574/50576029-fb02ac00-0e11-11e9-9b14-4d3b9d38da9e.png)

* **Distribution of the length of positive & negative reviews** 

![lennpos](https://user-images.githubusercontent.com/31864574/50576037-08b83180-0e12-11e9-9cb0-f7bec367a81c.png)

![lenneg](https://user-images.githubusercontent.com/31864574/50576041-11a90300-0e12-11e9-9918-4c088b42faea.png)


## Text pre-processing
* **Expanding contractions.** using the package [contractions](https://github.com/kootenpv/contractions)

In the English language, contractions are basically shortened versions of words or syllables. These shortened versions of existing words or phrases are created by removing specific letters and sounds. Converting each contraction to its expanded, original form often helps with text standardization. e.g. don’t -> do not, I’d -> I would, he’s -> he has.
```python
text = [contractions.fix(w) for w in df['text']]
```
Example:

'**It's** not Roger Corman that I hate, **it's** this god-awful movie.'

'**it is** not Roger Corman that I hate, **it is** this god-awful movie.'

* **Remove** links, special characters (i.e. non-alphanumeric characters), numbers and extra letters (i.e. fuuuuun->fuun, goooooood->good), make all letters lower case with the help of [regular expression](https://docs.python.org/3/library/re.html)
```python
def reg_ex(corpus):
  # Use regular expressions for the preprocessing
  docum = []
  for doc in corpus:
    doc = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", doc) #remove all links that appear in the beginning
    doc = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", doc)  #between 
    doc = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", doc)  # in the end of the sentence
    
    doc = re.sub('<.*?>', ' ', doc) # #substitute all <br\> with a space  
    
    doc = doc.lower()  # make all letters lower case
    
    doc = re.sub(r'(.)\1{3,}', r'\1\1', doc)  # replace fuuuuun->fuun, goooooood->good

    doc = re.sub(r"\W"," ",doc) #substitute all punctuations with a space  
    doc = re.sub(r"\d"," ",doc) #substitute all digits with a space
    doc = re.sub(r"\s+[a-z]\s+"," ",doc) #substitute all single letters in between with a space  
    doc = re.sub(r"\s+[a-z]$"," ",doc) # at the end
    doc = re.sub(r"^[a-z]\s+"," ",doc) # at the beggining

    doc = re.sub(r"\s+"," ",doc) # substitute all spaces with a single space
    doc = doc.strip()  #  characters(whitespaces) to be removed from beginning or end of the string
    docum.append(doc)
  return docum
```
Example:

'take a look at this one. **<br /><br />**I just loved the scene where hundred soldiers get shooting at the jungle.'

'take look at this one just loved the scene where hundred soldiers get shooting at the jungle'


* **Removing stopwords:** the most common words in a language  which have little or no significance.  Words like a, this, the etc. There is no universal stopword list but I used a standard English language stopwords list from [nltk](https://pythonspot.com/nltk-stop-words/).
Note: the word 'not' contains usefull information for n-grams so we will remove it from the stopwords thus we will keep it. (this is my thought on that, maybe not that helpful)
```python
stop_words = nltk.corpus.stopwords.words('english')
# we can see that the word 'not' is in stop_words
'not' in stop_words #output True
# maybe 'not' will be usefull for the n-grams so we will keep it
for i, w in enumerate(stop_words):
  if w == 'not':
    del stop_words[i]    

def remove_stopwords(doc):
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc
  
text = [remove_stopwords(w) for w in text]
```
Example:

'this was not the worst movie have ever seen but that is about as much as can be said about it'

'not worst movie ever seen much said'

* **Lemmatization:** using [WordNetLemmatizer](https://www.nltk.org/_modules/nltk/stem/wordnet.html) from nltk. Lemmatization is quite similar to stemming, it converts a word into its base form. However the root word also called lemma, is present in dictionary. It is considerably slower than stemming becasue an additonal step is perfomed to check if the lemma formed is present in dictionary. **Note:** We also have to specify the parts of speech of the word in order to get the correct lemma. Words can be in the form of Noun(n), Adjective(a), Verb(v), Adverb(r). Therefore, first we have to get the POS of a word before we can lemmatize it.

```python
# Lemmatize with POS Tag
import nltk
from nltk.corpus import wordnet  # To get words in dictionary with their parts of speech
nltk.download('averaged_perceptron_tagger')
# create a dataframe for the processsed data
prepDF = pd.DataFrame(text, columns=['text'])
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

w_tokenizer = nltk.tokenize.word_tokenize
wnl = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [wnl.lemmatize(w, get_wordnet_pos(w)) for w in w_tokenizer(text)]

prepDF['text_lemmatized'] = prepDF['text'].apply(lemmatize_text)

# each line is a review
prepDF['text_lemmatized_sent'] = np.asarray([' '.join(prepDF['text_lemmatized'][i]) for i in range(len(prepDF['text_lemmatized']))])
```

* **Save the dataframes:** using [pickle](https://docs.python.org/3/library/pickle.html) package

```python
prepDF['label'] = df['label']
pre_testpDF['label'] = test_df['label']
# storing as Pickle Files
with open("prepDF.pickle", 'wb') as f:
    pickle.dump(prepDF,f)
# Unpickling the dataset
with open('prepDF.pickle', 'rb') as f:
    prepDF = pickle.load(f)
```

## WordCloud Visualizations

**Positive WordCloud**

![wcpos](https://user-images.githubusercontent.com/31864574/50576046-238aa600-0e12-11e9-83e0-85d4e0a078a2.png)


**Negative WordCloud**

![wcneg](https://user-images.githubusercontent.com/31864574/50576048-2daca480-0e12-11e9-9d33-ad098c00c08a.png)


**Outcome:** We can see that in the Positive WordCloud the most frequent words are neutral words like: film, movie, one, see, show, character etc. Words with positive meaning are quite frequent e.g. good, love, well, great. On the other hand, in the Negative WordCloud we notice a similar words as the Positive one which is not that helpful. Thus the frequency of the words does not give enough information to classify the reviews.    



# Building Basic Models

## Evaluation Metrics

As our data are balanced the [accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score) and [f1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) would have similar values. Moreover, we will track the best model by [log-loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss).

```python
# import the evaluation metrics
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
```


## Bag Of Words
use **word counts** as features.  using [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) from scikit-learn.

* max_features=2000: use the 2000 most frequent words as features
* min_df=3: ignore all the words that appear in less than 3 documents (really weird-useless words)
* max_df=0.6: ignore all the words that appear in more than 60% of the documents i.e. 'the', 'is' etc
* ngram_range=(1, 2): use 1-gram and 2-grams features
* analyzer ='word': the feature should be made of word n-grams. 




```python
from sklearn.feature_extraction.text import CountVectorizer

ctv = CountVectorizer(max_features=2000,min_df=3, max_df=0.6, analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2))

# Fitting Count Vectorizer 
xtrain_ctv = ctv.fit_transform(train_x)
xtest_ctv  = ctv.transform(test_x)
```

apply logistic regression 
**NOTE:**  The C value is the inverse regularization strength (1/C regularization strength)

```python
from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(C=0.05)

start = time.time()
clf_lr.fit(xtrain_ctv, train_y)
print ("training time:", time.time() - start, "s") 

start = time.time()
predictions = clf_lr.predict(xtest_ctv)
print ("predict time:", time.time() - start, "s") 

lg = log_loss(test_y, predictions, eps=1e-15)*0.1
f1 = f1_score(test_y, predictions, pos_label=1, average='binary')

print ("logloss: %0.3f " % lg)
print ("f1_score: %0.3f " % f1)
```
We get f1_score: 0.884 and logloss: 0.403, thus we notice that this basic model performs really well in our data.

### Let's see the 20 most important features for positive and negative polarity separately; using the  LogisticRegression model with features from CountVectorizer 

```python
feature_to_coef = {word: coef for word, 
                   coef in zip(ctv.get_feature_names(), clf_lr.coef_[0])}

print("BEST POSITIVE WORDS")   
sorted(feature_to_coef.items(),key=lambda x: x[1],reverse=True)[:20]

print("BEST NEGATIVE WORDS")    
sorted(feature_to_coef.items(),key=lambda x: x[1])[:20]
```
![cvec20](https://user-images.githubusercontent.com/31864574/50576061-3f8e4780-0e12-11e9-8225-8d336cf93ee5.png)


**We notice that the best features have a strong positive or negative meaning for positive and negative polarity respectively. Furthermore, we took the decision to keep the stop_word 'not' which seems to be a nice move, really useful.**

## Tf-Idf
Term Frequency - Inverse Document Frequency [(tf-idf)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) from sklearn

* strip_accents='unicode': Remove accents and perform other character normalization during the preprocessing step.‘unicode’ works on any characters.
* min_df=3: ignore all the words that appear in less than 3 documents (really weird-useless words)
* max_df=0.9: ignore all the words that appear in more than 90% of the documents 
* ngram_range=(1, 2): use 1-gram and 2-grams features
* analyzer ='word': the feature should be made of word n-grams.
* use_idf='True': enable inverse-document-frequency reweighting.
* smooth_idf=1: smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.
* sublinear_tf=1: apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(strip_accents='unicode', min_df=3, max_df=0.9, analyzer='word', 
                      token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf='True',
                      smooth_idf=1,sublinear_tf=1)
start = time.time()
# Fitting TF-IDF to both training and test sets (semi-supervised learning)
train_x_tfv = tfv.fit_transform(train_x)
test_x_tfv  = tfv.transform(test_x)
print ("transform time:", time.time() - start, "s") 

```
Apply Logistic Regression

```python
clf_lr = LogisticRegression(C=10, penalty='l2')

start = time.time()
clf_lr.fit(train_x_tfv, train_y)
print ("training time:", time.time() - start, "s") 

start = time.time()
predictions = clf_lr.predict(test_x_tfv)
print ("predict time:", time.time() - start, "s") 

lg = log_loss(test_y, predictions, eps=1e-15)*0.1
f1 = f1_score(test_y, predictions, pos_label=1, average='binary')

print ("logloss: %0.3f " % lg)
print ("f1_score: %0.3f " % f1)  
print ("accuracy_score: %0.3f " % accuracy_score(test_y, predictions))
```
We get f1_score: 0.914 and logloss: 0.298. 

**Outcome:** The Tf-Idf model performs better than just CountVecorizer model; using the same classification algorithm. That was expected as we have seen it numerous time in different datasets.

### Let's see the 20 most important features for positive and negative polarity separately; using the LogisticRegression model with features from Tf-Idf

![tfi20](https://user-images.githubusercontent.com/31864574/50576065-546adb00-0e12-11e9-9501-09745c65f411.png)


**Outcome:** As previous in CountVecorizer features, the most important features have strong positive or negative meaning for positive and negative polarity respectively. 

The remarkable is that in the features of TfidfVectorizer have higher weights than CountVecorizer features. For example, the word 'excellent' in CountVecorizer has weight 0.877 and in TfidfVectorizer 14.082 and the word 'awful' in CountVecorizer has weight -1.144 and in TfidfVectorizer -14.960. Thus TfidfVectorizer has a stronger understanding of the polarity than the CountVecorizer.

## Next Steps 
* use embeddings and neural networks
* use pre-trained neural networks

## References
1. chazutsu package: [https://github.com/chakki-works/chazutsu](https://github.com/chakki-works/chazutsu)
2. Large Movie Review Dataset: [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/) 

