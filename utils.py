import os
import numpy as np
import matplotlib.image as mpimg

def listdir_nohidden(path, jpg_only=False):
    if jpg_only:
        return sorted(
            [el for el in os.listdir(path) if not el.startswith(".") and (el.endswith(".jpg") or el.endswith(".JPG"))])
    else:
        return sorted([el for el in os.listdir(path) if not el.startswith(".")])


def data_repartition(zord, directory_):
    folders = []
    if zord == "main_zord":
        classes = listdir_nohidden(directory_)
        for classe in classes:
            dir_ = directory_ + "/" + classe
            labels = listdir_nohidden(dir_)
            tot = 0
            for label in labels:
                tot += len(listdir_nohidden(diver(dir_ + "/" + label), jpg_only=True))
            folders.append(tot)

    else:
        directory_ = os.path.join(directory_,zord)
        for label in listdir_nohidden(directory_):
            file_nb = len(listdir_nohidden(diver(directory_ + "/" + label), jpg_only=True))
            folders.append(file_nb)

    return folders

def weighter(folders):
    m = max(folders)
    class_weight = {}
    for i in range(len(folders)):
        class_weight[i] = float(m) / float(folders[i])

    return class_weight

def diver(path):
    while len(listdir_nohidden(path))==1 :
        path+="/" + listdir_nohidden(path)[0]
    return path

class image_from_directory:

    def __init__(self, path, zord_kind):

        int_to_label={}
        int_label =0
        im_compt=0
        err_compt = 0

        if zord_kind == "main_zord" :
            n = sum(data_repartition("main_zord", path))
            x, y = np.empty((n, 256, 256, 3)), np.empty((n, 1), dtype="int32")
            int_label = int_reader(label="classe")
            for classe in self.classes_names :
                path_classe = os.path.join(path,classe)
                for label in listdir_nohidden(path_classe):
                    path_label = os.path.join(path_classe, label)
                    path_label = (diver(path_label))
                    for im in listdir_nohidden(path_label, jpg_only=True):
                        try:
                            img = os.path.join(path_label, im)
                            x[im_compt] = mpimg.imread(img)
                            y[im_compt] = int_label[classe]
                            im_compt += 1
                        except Exception as e :
                            print(e.__class__)
                            err_compt += 1

        elif zord_kind == "one4all" or zord_kind=="megazord_lsa":

            n = sum(data_repartition("main_zord", path))
            x, y = np.empty((n, 256, 256, 3)), []
            int_label = int_reader(label ="label")
            for classe in self.classes_names :
                path_classe = os.path.join(path,classe)
                for label in listdir_nohidden(path_classe):
                    path_label = os.path.join(path_classe,label)
                    path_label=(diver(path_label))
                    for im in listdir_nohidden(path_label, jpg_only=True):
                        try :
                            img=mpimg.imread(os.path.join(path_label,im))
                            x[im_compt] =img
                            y.append(int_label[label])
                            im_compt+=1
                        except Exception as e :
                            print(e.__class__)
                            err_compt+=1

        else :
            n = sum(data_repartition(zord_kind, path))
            x,y = np.empty((n,256,256,3)), []
            self.classes_names = listdir_nohidden(path)
            path_classe = os.path.join(path, zord_kind)
            for label in listdir_nohidden(path_classe):
                path_label = os.path.join(path_classe, label)
                path_label = (diver(path_label))

                for im in listdir_nohidden(path_label, jpg_only=True):
                    try:
                        img = mpimg.imread(os.path.join(path_label, im))
                        x[im_compt] = img
                        y.append(int_label[label])
                        im_compt += 1
                    except Exception as e:
                        print(e.__class__)
                        err_compt += 1

        assert im_compt+err_compt == n, "Some files have been missed"
        print("\n{} files have been imported".format(im_compt))
        print("{} errors occured".format(err_compt))
        assert len(x)==len(y)
        self.x = x
        self.y = y
        self.label_map = int_to_label

def zord_from_pb_file(path):
    path = path[:-3]
    i = 1
    while path[-i] != "/":
        i += 1
    return path[-i + 1:]

def labeller(path):
    f = open("labels.txt", "a")
    path += "/data/megazord_init"
    err_compt = 0
    n = sum(data_repartition("main_zord", path))
    classes_names = listdir_nohidden(path)
    for classe in classes_names:
        path_classe = os.path.join(path, classe)
        for label in listdir_nohidden(path_classe):
            path_label = os.path.join(path_classe, label)
            path_label = (diver(path_label))
            for im in listdir_nohidden(path_label, jpg_only=True):
                try:
                    input = "path " + str(
                        os.path.join(path_label, im)) + " " + "classe " + classe + " " + "label " + label + "\n"
                    f.write(input)
                except:
                    err_compt += 1
    f.close()

def label_reader(label=None):

    f=open("labels.txt")
    lines = f.readlines()
    rep={}
    for line in lines :
        i=0
        while line[i+1:i+4]!="jpg" :
            i+=1
        path = line[5:i]
        i+=3
        if label == "classe" :
            while line[i + 1:i + 8] != "classe ": i += 1
            j=0
            while line[i+8+j+1]!=" ": j+=1
            value = line[i+8: i+8+j+1]
        else :

            while line[i + 1:i + 7] != "label ":
                i += 1
            j=0
            while line[i+7+j+1:i+7+j+2]!="\n": j+=1
            value = line[i+7: i+7+j+1]

        rep[path] = value
    return rep

def int_labeller(label):
    dic = label_reader(label)
    integer_labels = np.unique(list(dic.values()))
    f = open("int_{}.txt".format(label), "a")
    f.write(str(integer_labels))
    f.close()

def int_reader(label):
    f = open("int_{}.txt".format(label))
    txt = f.read()
    dic={}
    nb_found=0
    i=1
    while i<len(txt)-2:
       if txt[i]=="'":
           i+=1
           j=1
           while txt[i+j]!="'":
               j+=1
           if j>2:
               dic[txt[i:i+j]] = nb_found
               nb_found+=1
           i+=j
       else : i+=1
    return dic

if __name__ == "__main__":

    os.remove("labels.txt")
    os.remove("int_label.txt")
    os.remove("int_classe.txt")

    labeller("/Users/lucas/swiss_knife")
    int_labeller("classe")
    int_labeller("label")
