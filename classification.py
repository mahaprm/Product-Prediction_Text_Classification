from model_components import vectorizer, build_model, grid_search, load_train_data, save_model
from test_prediction import test_prediction

classifier = "NB"  # Supported algorithms # "SVM" # "NB"
use_grid_search = False  # grid search is used to find hyperparameters. Searching for hyperparameters is time consuming
remove_stop_words = True  # removes stop words from processed text
stop_words_lang = 'english'  # used with 'remove_stop_words' and defines language of stop words collection
use_stemming = False  # word stemming using nltk
fit_prior = True  # if use_stemming == True then it should be set to False ?? double check
column_to_predict = "category"  # target variable.

if __name__ == '__main__':
    # creating only train data, train labels Since we have test csv to check confusion matrix
    train_data, train_labels = load_train_data(column_to_predict)

    # # Split dataset into training and testing data
    # train_data, test_data, train_labels, test_labels = train_test_split(
    #     data, labelData, test_size=0.2
    # )  # split data to train/test sets with 80:20 ratio

    count_vect = vectorizer(remove_stop_words, use_stemming, stop_words_lang)

    text_clf = build_model(train_data, train_labels, classifier, fit_prior, count_vect)
    gs_clf = ""
    if use_grid_search:
        gs_clf = grid_search(classifier, train_data, train_labels, text_clf)

    save_model(use_grid_search, gs_clf, text_clf, column_to_predict)

    test_prediction(column_to_predict)
