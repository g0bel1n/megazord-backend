'''
The SwissKnife is the center class of our project. It centralizes many of the functions we need.

TO DO LIST :
- Change train_zords to get rid of the dict thing
- Define the attribute .label in the __init__ function
- Get rid of the wildcard import (*) for utils
'''

import coremltools as ct
from tensorflow.keras import layers
from tensorflow import nn, stack, keras
from utils import *

class SwissKnife:
    '''
    Innovative and self explanatory class.
    '''

    # zords = ["zord1", "zord2", "zord3", ...]
    def __init__(self, directory):
        assert isinstance(directory, str), "directory parameter must be a str object"
        zords = listdir_nohidden(directory + "/data")
        self.zords = {}
        self.directory = directory
        self.train_queue = []
        for zord in zords:

            print("Trying to import " + zord + " model ...")
            try:
                if len(listdir_nohidden(self.directory + "/data/" + zord)) == 1:
                    self.zords[zord] = [None, [zord]]
                    print("\tSingle Label Class, no need to import.")
                else:
                    labels = listdir_nohidden(self.directory + "/data/" + zord)
                    self.zords[zord] = [keras.models.load_model(self.directory
                                                                + "/zords/"
                                                                + zord + ".pb"),
                                        labels]
                    print("\tSuccessul importation")
            except OSError:
                self.train_queue.append(zord)
                print("\t" + zord + " model has not been trained yet. It has been added"
                                    "to the training queue.")

        print("Trying to import main zord...")
        try:
            main_zord = keras.models.load_model(self.directory + "/zords/" + "main_zord" + ".pb")
            labels = listdir_nohidden(self.directory + "/data")
            self.zords["main_zord"] = [main_zord, labels]
            print("\tSuccess")
        except OSError:
            self.train_queue.append("main_zord")
            print("\tmain_zord" + " model has not been trained yet. It has been added"
                  "to the training queue.")
        print("\n#############################\n ")
        print("Zord models have been imported, please use .train_zords method to start"
              "the training queue if need "
              "be.\nTraining Queue : ", self.train_queue)

        self.data_augmentation = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomContrast((0, 1))])

    def train_zords(self, epochs=2):
        '''
        Trains the differents zords (CNN) in the self.train_queue.
        '''

        training_zords = {}

        for zord in self.train_queue:
            if zord == "main_zord":
                directory_ = self.directory + "/data"
            else:
                directory_ = self.directory + "/data/" + zord
            print(directory_)
            print("_____________ Training {} __________".format(zord))
            print(" Importing train_ds...")

            train_ds = keras.preprocessing.image_dataset_from_directory(
                directory_,
                labels="inferred",
                label_mode="int", shuffle=True, batch_size=32)

            folders = data_repartition(zord, directory_)

            class_weight = weighter(folders)

            print("As data is imbalanced, the following weights will be applied ",
                  list(class_weight.values()))

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

            # InceptionV3 layers are frozen so that we only need to train the last layer

            base_model.trainable = False

            print("Compilation of the CNN")
            input_shape = (256, 256, 3)
            inputs = keras.Input(shape=input_shape)
            x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.)(inputs)
            x = base_model(x, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)

            outputs = keras.layers.Dense(len(train_ds.class_names), activation=nn.softmax)(x)

            training_zords[zord] = keras.Model(inputs, outputs)

            training_zords[zord]._name = zord  # Changing each model name is capital as
            #it might cause errors when calling same labelled models

            training_zords[zord].compile(optimizer='adam',
                                         loss='sparse_categorical_crossentropy',
                                         metrics=['accuracy'])  # the default learning rate is 1e-3

            print("Fit sequence launched...")
            training_zords[zord].fit(augmented_train_ds, epochs=epochs, class_weight=class_weight)

            self.zords[zord] = [training_zords[zord], train_ds.class_names[::-1]]
            print("{} zord has been fitted and added to pre_megazord dictionnary. \n ".format(zord))
            print("Saving the zord ...")

        print(
            "Pre Megazord dictionnary is now complete. \n You can now fine tune (.fine_tune()) or"
            "call Megazord formation (.assemble_Megazord())")

    def fine_tune(self, zord, fine_tune_at=280, epochs=4, learning_rate=0.0001):
        '''
        Fine tunes the zord by his name from the fine_tune_at layer.
        '''

        print("_____________ Fine tuning {} __________".format(zord))
        print(" Importing train_ds...")
        directory_ = self.directory + "/data/" + zord
        train_ds = keras.preprocessing.image_dataset_from_directory(
            directory_,
            labels="inferred",
            label_mode="int", shuffle=True, batch_size=32)

        folders = data_repartition(zord,directory_)

        class_weight = weighter(folders)

        print("As data is imbalanced, the following weights will be applied ",
              list(class_weight.values()))

        print("Augmenting the train_ds")
        augmented_train_ds = train_ds.map(lambda x, y: (self.data_augmentation(x), y))
        augmented_train_ds = augmented_train_ds.prefetch(buffer_size=32)
        augmented_train_ds.shuffle(1000)
        print("Augmentation is done. Now begins fine-tuning")

        model = self.zords[zord][0]
        model.layers[2].trainable = True

        for layer in model.layers[2].layers[1:fine_tune_at]:
            layer.trainable = False

        model.optimizer.learning_rate = learning_rate  #a smaller learning rate is required to fine
                                            #tune the unfrozen layers

        model.fit(augmented_train_ds, epochs=epochs, class_weight=class_weight)

        self.zords[zord] = [model, train_ds.class_names[::-1]]
        print("{} zord has been fine-tuned and added to pre_megazord"
              "dictionnary. \n ".format(zord))

        print("Saving the zord ...")
        model.save(self.directory + "/zords/" + zord + ".pb")

    def assemble_megazord(self):
        '''
        Assembles Megazord.
        '''

        input_shape = (256, 256, 3)
        inputs_mz = keras.Input(shape=input_shape)

        class_pred = self.zords["main_zord"][0](inputs_mz)
        labels = []
        self.labels = labels
        nb_class = len(self.zords["main_zord"][1])
        compt = 0
        pre_stack = []
        for key in self.zords:
            if key != "main_zord":
                zorg_plus = self.zords[key]
                for label in zorg_plus[1]:
                    labels.append(label)
                if len(zorg_plus[1]) == 1:
                    pre_stack.append(class_pred[0, compt])
                else:
                    out = zorg_plus[0](inputs_mz)
                    for i in range(len(zorg_plus[1])):
                        pre_stack.append(class_pred[0, compt] * out[0, i])
                compt += 1
        assert compt == nb_class, "Classes are missing"
        print("Reaching the bottom of the neural network...")

        output_mz = stack([pre_stack], axis=0)

        print(
            "\n\t\t\t\t\t\t#############################################\n\t\t\t\t\t\t"
            "###########  MEGAZORD DEPLOYED  #############\n\t\t\t\t\t\t################"
            "#############################\n ")

        return keras.Model(inputs_mz,output_mz)  #The Megazord is deliberatly not an attribute
                                                 #of the SwissKnife object to reduce its constraint 
                                                 #on the computer's RAM

    def save(self, megazord):
        '''
        Saves megazord
        '''
        print("Saving Megazord")
        megazord.save(self.directory + "/zords/" + "megazord_lsa.pb")
        print("Megazord is saved")

    def megazord_to_coreml(self, megazord):
        '''
        Converts Megazord to CoreML.
        '''

        print("CoreML conversion is beginning...")

        image_input = ct.ImageType(shape=(1, 256, 256, 3,),
                                   bias=[-1, -1, -1])

        classifier_config = ct.ClassifierConfig(self.labels)

        megazord_cml = ct.convert(megazord, inputs=[image_input], 
                                  classifier_config=classifier_config)

        print("Saving the converted megazord_lsa...")

        megazord_cml.save(self.directory + "/zords/" + "megazord_lsa.mlmodel")

        print("Megazord is ready to serve ;)")

    # following methods does not work yet
    # def show_architecture(self):
    # tf.keras.utils.plot_model(self.megazord_lsa, show_shapes=True)



if __name__ == "__main__":

    DIRECTORY = "/Users/lucas/swiss_knife"

    swiss_knife = SwissKnife(DIRECTORY)
    swiss_knife.train_zords(epochs=2)

    #swiss_knife.fine_tune(zord = "handle", epochs=3)

    megazord = swiss_knife.assemble_megazord()

    swiss_knife.save(megazord)

    swiss_knife.megazord_to_coreml(megazord)
    