import os
from tqdm import tqdm
from utils import *
import numpy as np
import matplotlib.image as mpimg
path ="/Users/lucas/swiss_knife"

f = open(path+"/labels.txt", "a")
path+="/data/megazord_init"
err_compt=0
n = sum(data_repartition("main_zord", path))
x, y = np.empty((n, 256, 256, 3)), np.empty((n, 1))
classes_names = listdir_nohidden(path)
for classe in tqdm(classes_names):
    path_classe = os.path.join(path, classe)
    for label in listdir_nohidden(path_classe):
        path_label = os.path.join(path_classe, label)
        path_label = (diver(path_label))
        for im in listdir_nohidden(path_label, jpg_only=True):
            try:
                input = "path " + str(os.path.join(path_label, im))+" "+"classe "+classe+" "+"label "+label+"\n"
                f.write(input)
            except:
                err_compt += 1
f.close()
print(err_compt)