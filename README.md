# text_classification_on_IMDb_reviews

A sentiment analysis task of classifying the polarity (positive or negative) of IMDb reviews.


### Models


| Model | Accuracy |
| ------ | ------ |
| Log_Reg-cvec | 88.4% |
| Log_Reg-tfidf | 91.52% |
| NB-tfidf | 88.7%  |
| XGB-tfidf | 86.8%  |
| NBSVM | 91.4%  |

where: cvec: CountVectorizer, tfidf: TfidfVectorizer, Log_Reg: LogisticRegression, NB: BernoulliNB, XGB: XGBoost, NBSVM: Naive Bayes with Support Vector Machines 

Note: The pre=processing on the raw text for the first four models was too computetional expensive.

On the model NBSVM, both the pre-processing and the training was really quick.
