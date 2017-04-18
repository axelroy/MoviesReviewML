from pathlib import Path
from shutil import copyfile
from shutil import rmtree
from os import makedirs

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np

from tpot import TPOTClassifier

import numpy as np
import random
import sklearn.datasets

neg_path = Path('./tagged/neg')
pos_path = Path('./tagged/pos')

trainig_dir = './training/'
validation_dir = './validation/'

positive_directory_training = trainig_dir + 'pos/'
negative_directory_training = trainig_dir + 'neg/'

positive_directory_validation = validation_dir + 'pos/'
negative_directory_validation = validation_dir + 'neg/'

allowed_categories = ["VER", "ADJ", "NOM", "ADV"]

def create_training_set_random(count):
    """Crée un set de test aléatoire afin d'entrainer le modèle, avec exemples du dataset complet.
       X étant passé en paramètre"""

    neg_path = Path('./tagged/neg')
    list_negative_reviews = ([x for x in neg_path.iterdir()])

    pos_path = Path('./tagged/pos')
    list_positive_reviews = ([x for x in pos_path.iterdir()])

    training_set = []
    validation_set = []

    for _ in range(0,count):
        # Liste négative
        selected_item = random.choice(list_negative_reviews)
        training_set.append(selected_item)
        list_negative_reviews.remove(selected_item)

        # Liste positive
        selected_item = random.choice(list_positive_reviews)
        training_set.append(selected_item)
        list_positive_reviews.remove(selected_item)

    validation_set[0:0] = list_negative_reviews
    validation_set[0:0] = list_positive_reviews

    return training_set, validation_set

def create_training_set_fixed():
    """Create a fixed training dataset and validation set. The files names might
       not be consequentives, but they are always the same. Made to test the
       performances of the models."""
    training_set = []
    validation_set = []

    global neg_path
    list_negative_reviews = [x for x in neg_path.iterdir()]

    global pos_path
    list_positive_reviews = [x for x in pos_path.iterdir()]

    # Insertion dans le validation set de 80% fixe du dataset
    validation_set[0:0] = list_positive_reviews[0:200]
    validation_set[0:0] = list_negative_reviews[0:200]

    # Insertion dans le training set de 20% du dataset
    training_set[0:0] = list_positive_reviews[200:400]
    training_set[0:0] = list_negative_reviews[200:400]

    validation_set[0:0] = list_positive_reviews[400:1000]
    validation_set[0:0] = list_negative_reviews[400:1000]

    return training_set, validation_set

def copy_files(list_training_files, list_validation_files):
    """Copies the chosen files from the tagged directory to the training and
       validation directory"""
    global trainig_dir
    global validation_dir

    global positive_directory_training
    global positive_directory_validation
    global negative_directory_training
    global negative_directory_validation

    # Deleting and creating the directories tree
    create_directories()

    # copying the files
    for file_path in list_training_files:
        print(file_path.as_posix())
        if 'pos' in file_path.as_posix():
            copyfile(file_path.as_posix(), positive_directory_training + file_path.name)
        else:
            copyfile(file_path.as_posix(), negative_directory_training + file_path.name)

    for file_path in list_validation_files:
        if 'pos' in file_path.as_posix():
            copyfile(file_path.as_posix(), positive_directory_validation + file_path.name)
        else:
            copyfile(file_path.as_posix(), negative_directory_validation + file_path.name)

def create_directories():
    """ Erases without question the training and validation directories"""
    global positive_directory_training
    global positive_directory_validation
    global negative_directory_training
    global negative_directory_validation

    # Temporaire
    try:
        rmtree(negative_directory_training)
        rmtree(positive_directory_training)
        rmtree(negative_directory_validation)
        rmtree(positive_directory_validation)
    except:
        print("Directories will be created")

    makedirs(positive_directory_training)
    makedirs(negative_directory_training)
    makedirs(positive_directory_validation)
    makedirs(negative_directory_validation)

def preprocessing():
    """ Retains only the pertinent keywords to classify our reviews"""
    global trainig_dir
    global validation_dir

    files_to_clean = []

    # Cleaning the training data
    path = Path(trainig_dir)
    files_to_clean = collect_files_path(path)
    clean(files_to_clean)

    # Cleaning the training data
    path = Path(validation_dir)
    files_to_clean = collect_files_path(path)
    clean(files_to_clean)

def collect_files_path(path):
    """ Tries to parse the given path directory and parse it. The given directory
        must contain a directory for each classes, and each category contains a
        list of txt data to process. """
    files_to_clean = []

    dir_classes_collection = ([x for x in path.iterdir() if x.is_dir()])

    for directory in dir_classes_collection:
        for file in directory.iterdir():
            if not file.is_dir():
                files_to_clean.append(Path(file))

    return files_to_clean


def clean(files_to_clean):
    """ Deleting all the categories that are not in the allowed_categories in all the
        files listes in the list of files. The list must be a Path() list"""
    global allowed_categories

    for file_path in files_to_clean:
        lines = open(file_path.as_posix()).readlines()
        list_allowed = []

        for i, line in enumerate(lines[:]):
            try:
                line = line.rstrip('\r\n')
                _, category, canon = tuple(line.split("\t", maxsplit = 2))

                if ":" in category:
                    category = category.split(":")[0]

                if category in allowed_categories:
                    list_allowed.append(canon + "\r\n")
            except (ValueError):
                print('ERROR : BAD VALUE FORMAT : file : ', file_path, 'line ', line)

        open(file_path.as_posix(), 'w').writelines(list_allowed)

def load(dataset):
    global trainig_dir
    global validation_dir

    from sklearn.datasets import load_files

    if dataset == "training":
        files = load_files(trainig_dir)
        print(len(files.filenames))
        print(files.target_names[files.target[0]])
    else:
        files = load_files(validation_dir)

    return files

def classify(training_data):
    # Creates a occurency matrix for the reviews, without preprocessing
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(training_data.data)

    # Converts into a inverse frequency matrix
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # X_train_tfidf.shape

    # Training a classifier
    clf = MultinomialNB().fit(X_train_tfidf, training_data.target)

    # Load the validation set
    validation_set = load("validation")

    # Building pipeline
    text_clf_NB = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
    ])

    text_clf_SGDC = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
    ])

    text_clf_NB = text_clf_NB.fit(training_data.data, training_data.target)
    _ = text_clf_SGDC.fit(training_data.data, training_data.target)

    # Predictions with both models
    predicted_NB = text_clf_NB.predict(validation_set.data)
    predicted_SGDC = text_clf_SGDC.predict(validation_set.data)

    print("\nPrediction accuracies :")
    print("\tNaïve Bayes prediction \t : {0}".format(np.mean(predicted_NB == validation_set.target)))
    print("\tSVM prediction \t\t : {0}".format(np.mean(predicted_SGDC == validation_set.target)))

    # Possible parameters
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
    }

    gs_clf = GridSearchCV(text_clf_SGDC, parameters, n_jobs=-1) # Use all cores

    # Try fit on a subset of data
    gs_clf = gs_clf.fit(training_data.data, training_data.target)

    demoString = 'moche'

    print("\nThe demo prediction for \"{0}\" is : {1}".format(demoString, training_data.target_names[gs_clf.predict([demoString])[0]]))

    print("\nThe best score with SVM is {0}".format(gs_clf.best_score_))
    print("\nFound with the following parameters :\n")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    return text_clf_NB

# def parameter_optimisation():

# def evaluate(classifier_pipeline):
    # print(classifier_pipeline)

def TPOT_training():
    training_data = load("training")
    validation_data = load("validation")

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(training_data.data)
    tfidf_transformer = TfidfTransformer()

    X_validation_counts = count_vect.fit_transform(training_data.data)
    X_validation = tfidf_transformer.fit_transform(X_validation_counts)

    X_train = tfidf_transformer.fit_transform(X_train_counts)
    X_test = X_validation
    y_train = training_data.target
    y_test = validation_data.target

    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
    tpot.fit(X_train, y_train)
    
    # print(tpot.score(X_test, y_test))
    tpot.export('tpot_pipeline.py')


def main():
    """
        NAME
            Review Classification : Classifies the reviews with scikit learn.
        SYNOPSIS
            python main.py
        PARAMETERS
            None
    """
    # list_training_files, list_validation_files = create_training_set_random(800)
    list_training_files, list_validation_files = create_training_set_fixed()
    copy_files(list_training_files, list_validation_files)
    preprocessing()
    data = load("training")
    classifier_pipeline = classify(data)

    TPOT_training()


if __name__ == '__main__':
    main()
