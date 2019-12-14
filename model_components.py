import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def vectorizer(remove_stop_words, use_stemming, stop_words_lang):
    if remove_stop_words:
        count_vect = CountVectorizer(stop_words=stop_words_lang)
    elif use_stemming:
        count_vect = StemmedCountVectorizer(stop_words=stop_words_lang)
    else:
        count_vect = CountVectorizer(ngram_range=(1, 2))
    return count_vect


def load_train_data(column_to_predict):
    df = pd.read_csv('data/train.csv', sep=chr(1), error_bad_lines=False)
    # print(df.shape)
    # print(df.describe())
    # print(df.head())
    # Checking whether we have null values for features
    print(df.isna().sum())

    # df['gender'].value_counts().plot.bar()
    # df['category'].value_counts().plot.bar()
    # df['sub_cat'].value_counts().plot.bar()
    # df['sub_sub_cat'].value_counts().plot.bar()

    # plt.show()

    df = df[df[column_to_predict].notna() & (df['title']+ df['description']).notna()]  # selecting only non null gender
    labelData = df[column_to_predict]  # Column to predict
    data = df['title'] + df['description']  # Text columns
    data = data.apply(lambda x: np.str_(x))
    return data, labelData


def build_model(train_data, train_labels, classifier, fit_prior, count_vect):
    # Fitting the training data into a data processing pipeline and eventually into the model itself
    if classifier == "NB":
        print("Training NB classifier")
        # Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
        # The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
        # We will be using the 'text_clf' for prediction
        params = {'clf__alpha': 0.01, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 2),
                  'clf__fit_prior': fit_prior}  # found in GridSearchCV
        text_clf = Pipeline([
            ('vect', count_vect),
            ('tfidf', TfidfTransformer()),
            # Down sampling given very low accuracy so using over sampling
            ('sampling', RandomOverSampler(random_state=42)),
            # ('scale', StandardScaler()),
            ('clf', MultinomialNB())
        ])
        text_clf.set_params(**params)
        return text_clf.fit(train_data, train_labels)

    elif classifier == "SVM":
        print("Training SVM classifier")
        # Training Support Vector Machines - SVM
        text_clf = Pipeline([(
            'vect', count_vect),
            ('tfidf', TfidfTransformer()),
            ('sampling', RandomOverSampler(random_state=42)),
            # ('scale', StandardScaler()),
            ('clf', SGDClassifier(
                loss='hinge', penalty='l2', alpha=1e-3,
                random_state=42
            )
             )])
        return text_clf.fit(train_data, train_labels)


def grid_search(classifier, train_data, train_labels, text_clf):
    # Grid Search
    # Here, we are creating a list of parameters for which we would like to do performance tuning.
    # All the parameters name start with the classifier name (remember the arbitrary name we gave).
    # E.g. vect__ngram_range; here we are telling to use unigram and bigrams and choose the one which is optimal.
    if classifier == "NB":
        # NB parameters
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3)
        }
    else:
        # SVM parameters
        parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000, 10000, 50000),
            'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__alpha': (0.00001, 0.000001),
            'clf__penalty': ('l2', 'elasticnet'),
            'clf__n_iter_no_change': (10, 50, 80),
        }

    # Next, we create an instance of the grid search by passing the classifier, parameters
    # and n_jobs=-1 which tells to use multiple cores from user machine.
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_data, train_labels)

    # To see the best mean score and the params, run the following code
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)


def model_eval(text_clf, test_data, test_labels, use_grid_search, gs_clf):
    print("Evaluating model")
    # Score and evaluate model on test data using model without hyperparameter tuning
    predicted = text_clf.predict(test_data)
    prediction_acc = np.mean(predicted == test_labels)
    print("Confusion matrix without GridSearch:")
    print(confusion_matrix(test_labels, predicted))
    print("Mean without GridSearch: " + str(prediction_acc))

    # Score and evaluate model on test data using model WITH hyperparameter tuning
    if use_grid_search:
        predicted = gs_clf.predict(test_data)
        prediction_acc = np.mean(predicted == test_labels)
        print("Confusion matrix with GridSearch:")
        print(confusion_matrix(test_labels, predicted))
        print("Mean with GridSearch: " + str(prediction_acc))


def save_model(use_grid_search, gs_clf, text_clf, column_to_predict):
    # Save trained models to /output folder
    if use_grid_search:
        pickle.dump(
            gs_clf,
            open('outputs' + column_to_predict + ".model", 'wb')
        )
    else:
        pickle.dump(
            text_clf,
            open('outputs' + column_to_predict + ".model", 'wb')
        )
