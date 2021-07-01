from tensorflow import nn, keras
from tensorflow.keras import layers
import os
import numpy as np

class one4all:

    def __init__(self, dir):

        self.data_augmentation = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomContrast((0, 1))])
        try :
            self.model = keras.models.load_model(dir+"/zords/"+"one4all.pb")
            print("successful importation")

        except :
            dir+="/data/one4all"
            print("one4all need to be trained...")
            temp = one4all_labeller(dir)
            tab = np.array(temp)
            self.labels = tab[:,0]
            print(self.labels)

            folders = tab[:, 1].astype("int32")
            class_weight = weighter(folders)
            train_ds = keras.preprocessing.image_dataset_from_directory(
                dir,
                labels="inferred",
                label_mode="int", shuffle=True, batch_size=32)

            print(train_ds.class_names)
            print("As data is imbalanced, the following weights will be applied ", list(class_weight.values()))

            print("Augmenting the train_ds")
            augmented_train_ds = train_ds.map(lambda x, y: (self.data_augmentation(x), y))
            augmented_train_ds = augmented_train_ds.prefetch(buffer_size=32)  # facilitates training
            augmented_train_ds.shuffle(1000)
            print("Augmentation is done. Importation of InceptionV3 beginning...")

            base_model = keras.applications.InceptionV3(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                classifier_activation="softmax")

            # InceptionV3 layer are frozen so that we only need to train the last layer

            base_model.trainable = False

            print("Compilation of the CNN")
            input_shape = (256, 256, 3)
            inputs = keras.Input(shape=input_shape)
            x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.)(inputs)
            x = base_model(x, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)

            outputs = keras.layers.Dense(len(folders), activation=nn.softmax)(x)

            self.model = keras.Model(inputs, outputs)

            self.model.compile(optimizer='adam',
                                         loss='sparse_categorical_crossentropy',
                                         metrics=['accuracy'])
            epochs = 1

            self.model.fit(augmented_train_ds, epochs=epochs, class_weight=class_weight)

            self.model.save(dir + "/zords/" + "one4all" + ".pb")

            print("model saved")


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
        print(path)
        path+="/" + listdir_nohidden(path)[0]
    return path

def one4all_labeller(path):
    obj_map=[]
    folders = listdir_nohidden(path)
    print(folders)
    for folder in folders :
        path_ = path+"/"+folder
        file_nb = len(listdir_nohidden(diver(path_), jpg_only=True))
        obj_map.append([folder,file_nb])
    return obj_map


if __name__ == "__main__":

    directory = "/Users/lucas/swiss_knife"

    one4all_init = one4all(directory)

    model =  one4all_init.model


