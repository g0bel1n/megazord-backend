from tqdm import tqdm
import os
from random import random
import cv2


directory = "/Users/Lucas/swiss_knife/data"


def listdir_nohidden(path):
    return [el for el in os.listdir(path) if not el.startswith(".")]


def data_equilibrator(directory):

    classes = listdir_nohidden(directory)
    max =0

    for classe in classes:
        dir_ = directory+"/"+classe
        labels = listdir_nohidden((dir_))
        for label in labels :
            file_nb = len(listdir_nohidden(dir_+"/"+label))
            if file_nb > max : max =file_nb
    for classe in classes:
        dir_ = directory+"/"+classe
        labels = listdir_nohidden((dir_))
        for label in labels :
            files_list = listdir_nohidden(dir_ + "/" + label)
            delta = max - len(files_list)
            if delta>0 :
                for _ in range(delta):
                    im_name= random.choice(files_list)
                    im = cv2.imread(dir_+"/"+label+"/"+im_name)
                    cv2.imwrite(dir_+"/"+label+"/"+im_name+"_1.jpg", im)
        assert len(listdir_nohidden(dir_+"/"+label))==max; "didnt work"

    print("well done handsome")

if __name__ == "__main__":
    directory = "/Users/lucas/swiss_knife/data"
    data_equilibrator(directory)



