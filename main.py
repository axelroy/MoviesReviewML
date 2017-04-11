from pathlib import Path
from shutil import copyfile
from shutil import rmtree

import random
from os import makedirs

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

    # Cleaning the training datas
    path = Path(trainig_dir)
    files_to_clean = collect_files_path(path)
    clean(files_to_clean)

    # Cleaning the training datas
    path = Path(validation_dir)
    files_to_clean = collect_files_path(path)
    clean(files_to_clean)

def collect_files_path(path):
    """ Tries to parse the given path directory and parse it. The given directory
        must contain a directory for each classes, and each category contains a
        list of txt datas to process. """
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

def load():
    global trainig_dir
    sklearn.datasets.load_files(trainig_dir)

def main():
    """
        NAME
            Review Classification : Classifies the reviews with scikit learn.
        SYNOPSIS
            python main.py
        PARAMETERS
            None
    """
    list_training_files, list_validation_files = create_training_set_fixed()
    copy_files(list_training_files, list_validation_files)
    preprocessing()
    # load()


if __name__ == '__main__':
    main()
