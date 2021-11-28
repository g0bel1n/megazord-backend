<div align="center"> 

# Megazord
![plot](./ressources/megazord_pic.png?raw=true)

[![Unit tests](https://github.com/iSab01/megazord/actions/workflows/python-app.yml/badge.svg)](https://github.com/iSab01/megazord/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>
<div align="left">
Deep learning project realised by Lucas Saban and Augustin Cramer for Soft Next. 

Please mention us if you use this code.  
Feel free to contact us at : lucas.saban@ensae.fr or augustin.cramer@ensae.fr. 

**The goal is to classify the maximum amount of industrial pieces while minimizing misclassifcations.**
      
## Demonstration
      
<div align="center"> 


https://user-images.githubusercontent.com/73651505/143784896-72a36cc8-8747-4a00-a287-573eb60714fb.mov

   
<div align="left"> 
      


## Lexicon :
 **Zords** : Convolutionnal Neural Networks (CNN) trained to classify labels from the same class.  For example,
 **handle** distinguish the labels *handle_lock* and *handle_lockless*. 

**Main Zord** : CNN trained to distinguish classes. (*ex: handles from motors*)

**MegaZord** : Assemblage of the Main Zord and the different Zord

## Concept :

Our model makes predictions following a two-stage process.

Firstly, the image is inputed into the main_zord which determines its class. If it's a single classe label, then 
the job is done and the prediction is made.   
If it's not the case, if the image belongs to a class with several labels, a mask is applied so that the input is 
inputed in the adequate zord and only this one. Then a label prediction is made and outputted.  

**However, it will not be valid for the frontend as the confidence of the output must be superior to 0.7 to be
considered as such**

## OnBoarding
      
### Installation

```
pip install git+https://github.com/iSab01/megazord-backend
```


### Creating the dataset

In order to get enough data to train our models, we used *video_to_frame.py*. It allowed to video-scan our pieces in
different environnements quite quickly and to extract pictures from it.

For example :
```python
from MegaZord.utilitaries.video_to_frame import framer

directory = "my_dir_with_videos"

framer(fps_goal=8,  directory=directory, rescale=True, shape=(256,256))
```
You might want to put your video into labels folders and iterate upon the directories to automatize the process.


### Make the app 
#### Training and assembling a model

The main file is megazord.py in the directory SwissKnife  
``` python
from MegaZord.SwissKnife.megazord import SwissKnife
```

Instantiates the SwissKnife object with your main directory path and the base_model used for zord.
For now, InceptionV3, MobileNetV2, EfficientNetV2 and EfficientNet are supported (the input must be in low cap)
>The main_zord will be trained with inceptionV3 no matter the base_model chosen. 
> Making good main predictions on the main zord is crucial.
``` python
swiss_knife = SwissKnife(directory = "/Users/lucas/swiss_knife", base_model = "mobilenetv2")
```
If some zords are missing, they will be trained with the following command :
```python
swiss_knife.train_zords(epochs= 2)
```
Then, it's time to assemble the main_zord and every zords : 
```python
megazord = swiss_knife.assemble_megazord()
```
To save the megazord : 
```python
swiss_knife.save(megazord)
```
>  **Warnings** : If the base_model is efficientNetV2, you might encounter errors when loading the saved file.


Finally, the user can convert the megazord to .mlmodel with :
```python
swiss_knife.megazord_to_coreml(megazord)
```

#### Configuring the app

Now that the mlmodel has been saved, you need to create a plist file containing info about your labels for the app to 
work.
Please run the following command : 

```python
execfile("Megazord/utilitaries/ProductCatalog_Maker.py")
```

Then you have to open the [xcode project](https://github.com/iSab01/megazord_frontend) and 
update the files *ProductCatalog.plist* and *MegaZord_mobilenetv2.mlmodel*.

IOS >= 13.0.0 is required.

Finally, build the app on the simulator or your iDevice after signing the project with your developper account. 

## Versions

| Module      | Version | 
| :---        |    ----:   |  
| Python      | 3.9        |
| tensorflow   | 2.5.0        | 
| coremltools      | 4.1        |
| opencv      | 4.5.2.54       |
| tqdm      | 4.61.1       |
| numpy      | 1.19.5     |
| matplotlib | 3.4.2  |

## The frontend

The app is based on code sample found on the Apple Developper website. We only modified it to work with our model and to
adapt the detection threshold.  
It's not energy efficient and is quite heavy.  
For now, it's only for visualization but we might spend more time on it later on.

The code can be found here, in the [frontend repo](https://github.com/iSab01/megazord_frontend)

The .plist files needed to show predictions are the output of :
```python
execfile('MegaZord/utilitaries/ProductCatalog_Maker.py')
execfile('MegaZord/utilitaries/UndefinedProductCatalog_Maker.py')
execfile('MegaZord/utilitaries/ref_to_id_maker.py')
```

## Techniques used

In this project we used Data Augmentation, Transfer Learning (Inception V3,EfficentNet, EfficientNetV2, MobileNetV2),
Fine Tuning.

      
## Errors known

The importation processes in-between files is quite messy, you might check that first if you encounter errors. 

## Why *Megazord* and *zords* ? 

Well, it is because of the architecture of our neural network. It combines several small neural networks, trained to 
distinguish object from the same class (ex: motor_s and motor_m) to make a big neural networks that classify each label.
It made us remember the good old days, when we used to watch the *Power Rangers series*. They used to merge their zords 
(each Ranger had its own zord aka fighter robot) into a megazord which could defeat the bad guys. In our case, the 
megazord is the final model and the zords are the secondary models.

## Tips

### Directory organization

Please remember that your main directory must look like this :

      main_directory

            main_directory/class_1

                  main_directory/class_1/label_1

                        main_directory/class_1/label_1/pic_1 ...

                  main_directory/class_1/label_2

            main_directory/class_2

                  main/directory/class_2/label_1

                        main/directory/class_2/label_1/pic_2 ..
    
