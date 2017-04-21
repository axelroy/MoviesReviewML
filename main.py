"""
    Auteurs                      : Axel Roy
    Date dernière modification   : 21 Avril 2017
    But                          : Implémentation en machine learning d'un classificateur
                                   de revues de film


    ***********************************************************************************
                                        Utilisation
    ***********************************************************************************
    Ce script a été développé sous Linux puis adapté sous Windows.
    !! Sous Windows, l'installation de scipy peut être très problématique,
    se reporter à la page suivante : https://www.scipy.org/install.html

    le script se lance via python main.py.

    Le fichier requierements.txt contient les dépendances, que l'on peut installer
    via la commande pip install -r requierements.txt.

    ***********************************************************************************
                                        Implémentation
    ***********************************************************************************
    Le script s'occupe de supprimer les dossiers training et validation, puis de sélectionner
    aléatoirement x% du dataset dans chacun de ces répertoires, x étant configuré de base à
    80% pour le training et 20% pour la validation.

    Puis on build deux pipeline appropriés au traitement de texte :
    * Vectorisation via un bag of word puis une transformation de la matrice de compte
      en matrice de fréquence inversée.
    * Entrainement des modèles Naive Bayesian et SVM
    * Prédiction de la validité de la classification
    * Recherche d'optimisation des paramétres via GridSearchCV

    Et en supplément, une tentative d'optimisation automatique du pipeline
    (feature selection, model selection, hyperparameters selection et parameters selection).

    Cette section est laissée en commentaires afin de ne pas faire durer trop l'exécution,
    mais est utilisée dans le cadre du travail de Bachelor.

    Il ne fournit pas de classification d'un meilleur ordre car les algorithmes liés
    au traitement de texte ne sont pas encore implémentés.
"""

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

def create_training_set_random(training_percent):
    """Creates random training and validation subset in order to train and validate
       the model parameters.
       The parameter is the percentage of the files that will
       go into the training set.
       Output : A list of training file, a list of validation file"""
       print("Debut de la selection aleatoire des fichiers")

    neg_path = Path('./tagged/neg')
    list_negative_reviews = ([x for x in neg_path.iterdir()])

    pos_path = Path('./tagged/pos')
    list_positive_reviews = ([x for x in pos_path.iterdir()])

    training_set = []
    validation_set = []

    count = len(list_negative_reviews)
    count = int(count * training_percent / 100)

    # We add both to the positive and negative list the files into the training set,
    # and every file that hasn't been added to the training set will go into the
    # validation set
    for _ in range(0,count):
        selected_item = random.choice(list_negative_reviews)
        training_set.append(selected_item)
        list_negative_reviews.remove(selected_item)

        selected_item = random.choice(list_positive_reviews)
        training_set.append(selected_item)
        list_positive_reviews.remove(selected_item)

    validation_set[0:0] = list_negative_reviews
    validation_set[0:0] = list_positive_reviews

    print("Fin de la selection aleatoire des fichiers")
    return training_set, validation_set

def create_training_set_fixed():
    """ Create a fixed training dataset and validation set. The files names might
        not be consequentives, but they are always the same. Made to test the
        performances of the models."""

    print("Creation du set de donnees fixes")
    training_set = []
    validation_set = []

    list_negative_reviews = [x for x in neg_path.iterdir()]
    list_positive_reviews = [x for x in pos_path.iterdir()]

    # Insertion dans le validation set de 80% fixe du dataset
    training_set[0:0] = list_positive_reviews[0:200]
    training_set[0:0] = list_negative_reviews[0:200]

    # Insertion dans le training set de 20% du dataset
    validation_set[0:0] = list_positive_reviews[200:400]
    validation_set[0:0] = list_negative_reviews[200:400]

    training_set[0:0] = list_positive_reviews[400:1000]
    training_set[0:0] = list_negative_reviews[400:1000]

    print("Fin de la creation du set fixe")

    return training_set, validation_set

def copy_files(list_training_files, list_validation_files):
    """Copies the chosen files from the tagged directory to the training and
       validation directory"""
    print("Debut de la copie des fichiers")

    # Deleting and creating the directories tree
    create_directories()

    # copying the files
    for file_path in list_training_files:
        # print(file_path.as_posix())
        if 'pos' in file_path.as_posix():
            copyfile(file_path.as_posix(), positive_directory_training + file_path.name)
        else:
            copyfile(file_path.as_posix(), negative_directory_training + file_path.name)

    for file_path in list_validation_files:
        if 'pos' in file_path.as_posix():
            copyfile(file_path.as_posix(), positive_directory_validation + file_path.name)
        else:
            copyfile(file_path.as_posix(), negative_directory_validation + file_path.name)

    print("Fin de la copie des fichiers")

def create_directories():
    """ Erases without question the training and validation directories"""
    global positive_directory_training
    global positive_directory_validation
    global negative_directory_training
    global negative_directory_validation

    print("Suppression de repertoires :")
    # Temporaire
    try:
        rmtree(negative_directory_training)
        rmtree(positive_directory_training)
        rmtree(negative_directory_validation)
        rmtree(positive_directory_validation)
    except:
        print("Les repertoires n'existaient pas.")

    makedirs(positive_directory_training)
    makedirs(negative_directory_training)
    makedirs(positive_directory_validation)
    makedirs(negative_directory_validation)

    print("Repertoires de trainings supprimés et recrees :")
    print(positive_directory_validation)
    print(negative_directory_validation)
    print(positive_directory_training)
    print(negative_directory_training)

def preprocessing():
    """ Retains only the pertinent keywords to classify our reviews"""
    print("Debut du pretraitement des donnees afin de ne conserver que les mots interessants")
    global trainig_dir
    global validation_dir

    files_to_clean = []

    # Cleaning the training data
    path = Path(trainig_dir)
    files_to_clean = collect_files_path(path)
    clean(files_to_clean)

    # Cleaning the validation data
    path = Path(validation_dir)
    files_to_clean = collect_files_path(path)
    clean(files_to_clean)

    print("Fin du pretraitement des donnees")

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
        try:
            lines = open(file_path.as_posix(), encoding='utf8').readlines()
            list_allowed = []

            for i, line in enumerate(lines[:]):
                try:
                    line = line.rstrip('\r\n')
                    _, category, canon = tuple(line.split("\t", maxsplit = 2))

                    if ":" in category:
                        category = category.split(":")[0]

                    if category in allowed_categories:
                        list_canon = canon.split("|")
                        for word in list_canon:
                            list_allowed.append(word.lower() + " ")

                except (ValueError):
                    print('ERROR : BAD VALUE FORMAT : file : ', file_path, 'line ', line)

            open(file_path.as_posix(), 'w', encoding='utf8').writelines(list_allowed)
        except:
            print("Erreur lors de l'ouverture du fichier", file_path.as_posix())

def load(dataset):
    global trainig_dir
    global validation_dir

    from sklearn.datasets import load_files

    if dataset == "training":
        files = load_files(trainig_dir)
    else:
        files = load_files(validation_dir)

    return files

def classify(training_data):
    """ Tries to vectorize the document, normalize then and then predict
        trains the model, optimize parameters and then validate the model"""
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
    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-5),
    }

    gs_clf = GridSearchCV(text_clf_SGDC, parameters, n_jobs=-1) # Use all cores

    # Try fit on a subset of data
    gs_clf = gs_clf.fit(training_data.data, training_data.target)

    print("\nThe best score with SVM is {0}".format(gs_clf.best_score_))
    print("\nFound with the following parameters :\n")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    return text_clf_NB

def main():
    """
        NAME
            Review Classification : Classifies the reviews with scikit learn.
        SYNOPSIS
            python main.py
        PARAMETERS
            None
    """
    list_training_files, list_validation_files = create_training_set_random(80)
    # list_training_files, list_validation_files = create_training_set_fixed()
    copy_files(list_training_files, list_validation_files)
    preprocessing()
    data = load("training")
    classifier_pipeline = classify(data)

    # TPOT_training()

# Optimisation du pipeline via un algorithme génétique. Intéressant, mais fonctionne
# sous Linux, donc mis en commentaires pour le rendu
# def TPOT_training():
#     """Trying pipeline optimisation with the full automatic pipeline
#        optimisation TPOT utilities."""
#
#     # Loading both datasets
#     training_data = load("training")
#     validation_data = load("validation")
#
#     # Vectorize the text datas
#     count_vect = CountVectorizer()
#     X_train_counts = count_vect.fit_transform(training_data.data)
#
#     # Conversion to term-frequency times inverse document-frequency
#     tfidf_transformer = TfidfTransformer()
#     X_validation_counts = count_vect.fit_transform(training_data.data)
#     X_validation = tfidf_transformer.fit_transform(X_validation_counts)
#
#     X_train = tfidf_transformer.fit_transform(X_train_counts)
#     X_test = X_validation
#     y_train = training_data.target
#     y_test = validation_data.target
#
#     tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
#     tpot.fit(X_train, y_train)
#
#     print(tpot.score(X_test, y_test))
#     # tpot.export('tpot_pipeline.py')

if __name__ == '__main__':
    main()
