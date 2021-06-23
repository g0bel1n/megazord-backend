from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import os
import coremltools as ct

class SwissKnife():

    #zords = ["zord1", "zord2", "zord3", ...]
    def __init__(self,zords,directory):
        assert type(zords)==list, "zords parameter must be a list object."
        assert type(directory)== str, "directory parameter must be a str object"
        self.zords = {}
        self.directory = directory
        self.train_queue = []
        for zord in zords :

            print("Trying to import " + zord + " model ...")
            try :
                if len(listdir_nohidden(self.directory+"/"+zord))==1:
                    self.zords[zord] = [None,[zord]]
                    print("\tSingle Label Class, no need to import.")
                else:
                    labels = listdir_nohidden(self.directory+"/"+zord)
                    self.zords[zord] = [tf.keras.models.load_model(self.directory+"/"+ zord+".pb"),labels]
                    print("\tSuccessul importation")
            except :
                self.train_queue.append(zord)
                print("\t"+ zord + " model has not been trained yet. It has been added to the train queue.")

        print("Trying to import main zord...")
        try :
            main_zord = tf.keras.models.load_model(self.directory+ "/"+"main_zord"+".pb")
            labels = listdir_nohidden(self.directory+"/"+"main_zord")
            self.zords["main_zord"] = [main_zord,labels]
            print("\tSuccess")
        except :
            self.train_queue.append("main_zord")
            print("\tmain_zord" + " model has not been trained yet. It has been added to the train queue.")
        print("\n#############################\n ")
        print("Zord models have been imported, please use .train_zords method to start the training queue if need be.\nTraining Queue : ", self.train_queue)

        self.data_augmentation = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomContrast((0,1))])



    def train_zords(self,epochs=2):
        training_zords = {}

        for zord in self.train_queue:

            print("_____________ Training {} __________".format(zord))
            print(" Importing train_ds...")
            directory_ = self.directory+"/"+zord
            train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory_,
            labels="inferred",
            label_mode="int", shuffle=True, batch_size=32)

            if len(train_ds.class_names) == 1 :
                self.zords[zord] = [None,[zord]]
                print("Single label class doesn't need to be trained.")
                continue



            print("Augmenting the train_ds")
            augmented_train_ds = train_ds.map(lambda x,y : (self.data_augmentation(x),y))
            augmented_train_ds = augmented_train_ds.prefetch(buffer_size=32)
            augmented_train_ds.shuffle(1000)
            print("Augmentation is done. Importation of InceptionV3 beginning...")

            base_model = tf.keras.applications.InceptionV3(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                classifier_activation="softmax")

            base_model.trainable = False

            print("Compilation of the CNN")
            input_shape=(256,256,3)
            inputs = keras.Input(shape=input_shape)
            x = tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255.)(inputs)
            x = base_model(x, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)

            outputs = keras.layers.Dense(len(train_ds.class_names),activation=tf.nn.softmax)(x)

            training_zords[zord] = keras.Model(inputs, outputs)
            training_zords[zord]._name = zord

            training_zords[zord].compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

            print("Fit sequence launched...")
            training_zords[zord].fit(augmented_train_ds, epochs=epochs)

            self.zords[zord] = [training_zords[zord], train_ds.class_names]
            print("{} zord has been fitted and added to pre_megazord dictionnary. \n ".format(zord))
            print("Saving the zord ...")
            training_zords[zord].save(self.directory+"/"+zord+".pb")
            self.train_queue.remove(zord)

        print("Pre Megazord dictionnary is now complete. \n You can now fine tune (.fine_tune()) or call Megazord formation (.assemble_Megazord())")

    def fine_tune(self, zord, fine_tune_at = 280, epochs =4, lr = 0.0001):

        print("_____________ Fine tuning {} __________".format(zord))
        print(" Importing train_ds...")
        directory_ = self.directory + "/" + zord
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory_,
            labels="inferred",
            label_mode="int", shuffle=True, batch_size=32)

        print("Augmenting the train_ds")
        augmented_train_ds = train_ds.map(lambda x, y: (self.data_augmentation(x), y))
        augmented_train_ds = augmented_train_ds.prefetch(buffer_size=32)
        augmented_train_ds.shuffle(1000)
        print("Augmentation is done. Now begins fine-tuning")

        model = self.zords[zord][0]
        model.layers[2].trainable = True

        for layer in model.layers[2].layers[1:fine_tune_at]:
            layer.trainable =  False

        model.optimizer.learning_rate = lr

        model.fit(augmented_train_ds, epochs=epochs)

        self.zords[zord] = [model, train_ds.class_names]
        print("{} zord has been fine-tuned and added to pre_megazord dictionnary. \n ".format(zord))

        print("Saving the zord ...")
        model.save(self.directory+"/" + zord + ".pb")

##TW L'ordre des fichiers de main_zorg doit être le meme que celui du dossier parent

    def assemble_Megazord(self):

        input_shape = (256,256,3)
        inputs_MZ = keras.Input(shape=input_shape)

        class_pred=self.zords["main_zord"][0](inputs_MZ)
        labels_count = 0
        labels = []
        for key in self.zords.keys() :
            if key != "main_zord":
                el = self.zords[key]
                labels_count+= len(el[1])
                for label in el[1] :
                    labels.append(label)

        self.labels = labels
        nb_class = len(self.zords["main_zord"][1])
        compt = 0
        pre_stack = []
        for key in self.zords.keys() :
            if key != "main_zord":
                zorg_plus = self.zords[key]
                if len(zorg_plus[1])==1 :
                    pre_stack.append(class_pred[0,compt])
                else :
                    out = zorg_plus[0](inputs_MZ)
                    for i in range(len(zorg_plus[1])):
                        pre_stack.append(class_pred[0,compt]*out[0,i])
                compt+=1
        assert compt == nb_class, "Classes are missing"
        print("Reaching the bottom of the neural network...")

        output_MZ = tf.stack([pre_stack], axis=0)

        print("\n\t\t\t\t\t\t#############################################\n\t\t\t\t\t\t###########  MEGAZORD DEPLOYED  #############\n\t\t\t\t\t\t#############################################\n ")


        return keras.Model(inputs_MZ, output_MZ)

    def save(self, megazord):
        print("Saving Megazord")
        megazord.save("megazord.pb")
        print("Megazord is saved")

    def megazord_to_coreML(self, megazord):

        image_input = ct.ImageType(shape=(1, 256, 256, 3,),
                       bias=[-1,-1,-1])

        classifier_config = ct.ClassifierConfig(self.labels)

        megazord_CML =  ct.convert(megazord, inputs=[image_input], classifier_config=classifier_config)

        print("Saving the converted megazord...")

        megazord_CML.save("megazord.mlmodel")

        print("Megazord is ready to go ;)")


    def show_architecture(self):
        tf.keras.utils.plot_model(self.megazord, show_shapes=True)


def listdir_nohidden(path):
        return [el for el in os.listdir(path) if not el.startswith(".")]


if __name__ == "__main__" :

    directory = "/Users/lucas/swiss_knife_data"
    zords = ["ball_bearing", "handle", "motor"]

    swiss_knife = SwissKnife(zords, directory)

    swiss_knife.train_zords(epochs = 1)

    swiss_knife.fine_tune(zord = "handle", epochs=1)

    megazord = swiss_knife.assemble_Megazord()

    swiss_knife.save(megazord)

    swiss_knife.megazord_to_coreML(megazord)
