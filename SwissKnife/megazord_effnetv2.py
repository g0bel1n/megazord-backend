# The SwissKnife is the center class of our project. It centralizes many of the functions we need.

# ToDo LIST :
# Change train_zords to get rid of the dict thing
# Get rid of the wildcard import (*) for utils


class SwissKnife:
    """
    Innovative and self explanatory class.
    """

    # zords = ["zord1", "zord2", "zord3", ...]
    def __init__(self, directory):
        assert isinstance(directory, str), "directory parameter must be a str object"
        zords = listdir_nohidden(directory + "/data")
        self.zords = {}
        self.directory = directory
        self.train_queue = []
        self.labels = []
        self.base_model = effnetv2_model.get_model("efficientnetv2-b0", include_top=False, pretrained=True)

        # EffNetB0 layers are frozen so that we only need to train the last layer

        self.base_model.trainable = False
        for zord in zords:

            print("Trying to import " + zord + " model ...")
            in_file = listdir_nohidden(self.directory + "/data/" + zord)
            try:
                if len(in_file) == 1:
                    self.zords[zord] = [None, [zord]]
                    print("\tSingle Label Class, no need to import.")
                    self.labels.append(zord)
                else:
                    self.zords[zord] = [keras.models.load_model(self.directory
                                                                + "/zords/"
                                                                + zord + "_effnetv2" + ".pb"),
                                        in_file]
                    self.labels.append(in_file)
                    print("\tSuccessful importation")
            except OSError:
                self.train_queue.append(zord)
                self.labels.append(in_file)
                print("\t" + zord + " model has not been trained yet. It has been added"
                                    "to the training queue.")

        print("Trying to import main zord...")
        try:
            main_zord = keras.models.load_model(self.directory + "/zords/" + "main_zord" + "_effnetv2" + ".pb")
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
        self.labels = flatten(self.labels)

    def train_zords(self, epochs=2):
        """
        Trains the different zords (CNN) in the self.train_queue.
        """
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

            folders = data_repartition(zord, self.directory + "/data")

            class_weight = weighter(folders)

            print("As data is imbalanced, the following weights will be applied ",
                  list(class_weight.values()))

            print("Augmenting the train_ds")
            augmented_train_ds = train_ds.map(lambda x, y: (self.data_augmentation(x), y))
            augmented_train_ds = augmented_train_ds.prefetch(buffer_size=32).shuffle(10)  # facilitates training
            print("Augmentation is done. Importation of InceptionV3 beginning...")



            print("Compilation of the CNN")
            input_shape = (256, 256, 3)
            inputs = keras.Input(shape=input_shape)
            x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.)(inputs)
            x = self.base_model(x, training=False)
            x = keras.layers.Reshape((1, -1, 1280 ))(x)
            x = keras.layers.GlobalAveragePooling2D()(x)

            outputs = keras.layers.Dense(len(train_ds.class_names), activation=nn.softmax)(x)

            model = keras.Model(inputs, outputs)

            model._name = zord  # Changing each model name is capital as
            # it might cause errors when calling same labelled models

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])  # the default learning rate is 1e-3

            print("Fit sequence launched...")
            model.fit(augmented_train_ds, epochs=epochs, class_weight=class_weight)

            self.zords[zord] = [model, train_ds.class_names[::-1]]
            print("{} zord has been fitted and added to pre_megazord dictionary. \n ".format(zord))
            print("Saving the zord ...")
            model.save(self.directory + "/zords/" + zord + "_effnetv2" + ".pb")
            del model

        print(
            "Pre Megazord dictionary is now complete. \n You can now fine tune (.fine_tune()) or"
            "call Megazord formation (.assemble_Megazord())")

    def fine_tune(self, zord, fine_tune_at=280, epochs=4, learning_rate=0.0001):
        """
        Fine tunes the zord by his name from the fine_tune_at layer.
        """

        print("_____________ Fine tuning {} __________".format(zord))
        print(" Importing train_ds...")
        directory_ = self.directory + "/data/" + zord
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
        augmented_train_ds = augmented_train_ds.prefetch(buffer_size=32).shuffle(10)
        print("Augmentation is done. Now begins fine-tuning")

        model = self.zords[zord][0]
        model.layers[2].trainable = True

        for layer in model.layers[2].layers[1:fine_tune_at]:
            layer.trainable = False

        model.optimizer.learning_rate = learning_rate  # a smaller learning rate is required to fine
        # tune the unfrozen layers

        model.fit(augmented_train_ds, epochs=epochs, class_weight=class_weight)

        self.zords[zord] = [model, train_ds.class_names[::-1]]
        print("{} zord has been fine-tuned and added to pre_megazord"
              "dictionary. \n ".format(zord))

        print("Saving the zord ...")
        model.save(self.directory + "/zords/" + zord + ".pb")
        del model

    def assemble_megazord(self):
        """
        Assembles Megazord.
        """

        input_shape = (256, 256, 3)
        megazord_input = keras.Input(shape=input_shape)
        print(self.zords)
        megazord_output = fusion_layer.ZordsCombination(self.zords)(megazord_input)

        print(
            "\n\t\t\t\t\t\t#############################################\n\t\t\t\t\t\t"
            "###########  MEGAZORD DEPLOYED  #############\n\t\t\t\t\t\t################"
            "#############################\n ")

        return keras.Model(megazord_input, megazord_output)

    def save(self, model):
        """
        Saves megazordriot
        """
        print("Saving Megazord")
        model.save(self.directory + "/zords/" + "megazord_effnetv2.pb")
        print("Megazord is saved")

    def megazord_to_coreml(self, model):
        """
        Converts Megazord to CoreML.
        """

        print("CoreML conversion is beginning...")

        image_input = ct.ImageType(shape=(1, 256, 256, 3,),
                                   bias=[-1, -1, -1])

        classifier_config = ct.ClassifierConfig(self.labels)

        megazord_cml = ct.convert(model, inputs=[image_input],
                                  classifier_config=classifier_config)

        print("Saving the converted megazord...")

        megazord_cml.save(self.directory + "/zords/" + "megazord.mlmodel")

        print("Megazord is ready to serve ;)")


if __name__ == "__main__":

    from tensorflow.keras import layers
    from tensorflow import nn, keras
    from efficientnet.tfkeras import EfficientNetB0
    import coremltools as ct
    from megazord.utilitaries.utils import listdir_nohidden, data_repartition, weighter, flatten
    from megazord.layers import fusion_layer
    from megazord.efficientnetv2 import effnetv2_model

    DIRECTORY = "/Users/lucas/swiss_knife"

    swiss_knife = SwissKnife(DIRECTORY)
    swiss_knife.train_zords(epochs=3)

    # swiss_knife.fine_tune(zord="handle", epochs=3)

    megazord = swiss_knife.assemble_megazord()

    # swiss_knife.save(megazord)
    print(swiss_knife.labels)

    try:
        swiss_knife.megazord_to_coreml(megazord)
    except Exception as e:
        print(e.__class__)
