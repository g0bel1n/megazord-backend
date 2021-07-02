import os
import random
from PIL import Image


directory = "/Users/Lucas/swiss_knife/data"


def listdir_nohidden(path, jpg_only = False):
    if jpg_only :return [el for el in os.listdir(path) if not el.startswith(".") and  el.endswith(".jpg")]
    else : return [el for el in os.listdir(path) if not el.startswith(".")]


def data_equilibrator(directory):

    classes = listdir_nohidden(directory)
    max =0
    for classe in classes:
        dir_ = directory+"/"+classe
        labels = listdir_nohidden((dir_))
        for label in labels :
            file_nb = len(listdir_nohidden(dir_+"/"+label, jpg_only=True))
            if file_nb > max : max =file_nb
    print("Target Number of images per folder : ", max)
    for classe in classes:
        dir_ = directory+"/"+classe
        labels = listdir_nohidden((dir_))
        for label in labels :
            files_list = listdir_nohidden(dir_ + "/" + label, jpg_only=True)
            delta = max - len(files_list)
            print("Regularizing " + label + " ...")
            if delta > 0.3*len(files_list):
                print("The gap between the target and the amount of data is quite important : ", delta)
                go = input("Continue ? (y/n)   ==>")
                if go=="n" : continue

            while delta!=0 :
                im_name= random.choice(files_list)
                im = Image.open(dir_+"/"+label+"/"+im_name)
                try : im.save(dir_+"/"+label+"/"+im_name[:-4]+"_1.jpg")
                except : print("no save")
                files_list = listdir_nohidden(dir_ + "/" + label, jpg_only=True)
                delta = max - len(files_list)
            assert len(listdir_nohidden(dir_+"/"+label, jpg_only=True))==max, "didnt work"
            print("\t done")

    print(
        "\n\t\t\t\t\t\t#############################################################\n\t\t\t\t\t\t###########  All folders have been regularized  #############\n\t\t\t\t\t\t#############################################################\n ")


if __name__ == "__main__":
    directory = "/Users/lucas/swiss_knife/data"
    data_equilibrator(directory)



