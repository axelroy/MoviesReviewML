from pathlib import Path
import random

def create_training_set_random(count):
    """Crée un set de test aléatoire afin d'entrainer le modèle, avec exemples du dataset complet.
       X étant passé en paramètre"""

    neg_path = Path('./tagged/neg')
    list_negative_reviews = ([x for x in neg_path.iterdir()])

    pos_path = Path('./tagged/pos')
    list_positive_reviews = ([x for x in pos_path.iterdir()])

    training_set = []
    available_indexes = []

    for _ in range(0,count):
        # Liste négative
        selected_item = random.choice(list_negative_reviews)
        training_set.append(selected_item)
        list_negative_reviews.remove(selected_item)

        # Liste positive
        selected_item = random.choice(list_positive_reviews)
        training_set.append(selected_item)
        list_positive_reviews.remove(selected_item)

    return training_set

def create_training_set_fixed():
    """Crée un set de test fixe afin d'entrainer le modèle, avec exemples du dataset complet"""
    training_set = []

    neg_path = Path('./tagged/neg')
    list_negative_reviews = [x for x in neg_path.iterdir()]

    pos_path = Path('./tagged/pos')
    list_positive_reviews = [x for x in pos_path.iterdir()]

    positive_review_selection = list_positive_reviews[200:400]
    negative_review_selection = list_negative_reviews[200:400]

    # Insertion des sous-listes
    training_set[0:0] = positive_review_selection
    training_set[0:0] = negative_review_selection

    return training_set

def preprocessing(list_selected_files):
    for file_path in list_selected_files:
        file = open(file_path.as_posix(), 'r')

        for line in file.readline():
            _, category, canon = tuple(line.split("\t"))
            print(category, "(", canon , ")")

def main():
    """
        NAME
            TSP : Solve the travelling salesman problem using genetic algorithm
        SYNOPSIS
            python DroxlerRoy.py [--nogui] [--maxtime s] [filename]
        PARAMETERS
            [--nogui] : disable the gui, default to true
            [--maxtime s] : diffine the maximum time of exectution in seconds , default at 1000 s
            [filename] : Format expected :
                                        City_Name X_Position Y_Position
                                        i.e :
                                        v0 54 391
                                        v1 77 315
                                        It uses the /data/pb010.txt path
    """
    list_selected_files = create_training_set_random(200)
    # list_selected_files = create_training_set_fixed()
    # print('[%s]' % '\n '.join(map(str, list_selected_files)))
    preprocessing(list_selected_files)


if __name__ == '__main__':
    main()
