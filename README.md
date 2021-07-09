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
## OnBoarding

The main file is megazord.py in the directory SwissKnife  
``` python
from MegaZord.SwissKnife.megazord import SwissKnife
```

Instantiates the SwissKnife object with your your main directory path and the base_model used for zord.

>The main_zord while be trained with inceptionV3 no matter the base_model chosen. 
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


Finally, the user can convert the megazord the .mlmodel with :
```python
swiss_knife.megazord_to_coreml(megazord)
```

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

## The app (interface)

The app is based on code sample found on the Apple Developper website. We only modified it to work with our model and to adapt the detection threshold.  
It's not energy efficient and is quite heavy.  
For now, it's only for visualization but we might spend more time on it later on.

The .plist file needed to show predictions is the output of :
```python
execfile('MegaZord/utilitaries/plis_maker.py')
```

## Techniques used

In this project we used Data Augmentation, Transfer Learning (Inception V3, EfficientNetV2, MobileNetV2), Fine Tuning and multi inputs/outputs neural networks
      
## Errors known

None.

## Why *Megazord* and *zords* ? 






Well, it is because of the architecture of our neural network. It combines several small neural networks, trained to distinguish object from the same class (ex: motor_s and motor_m) to make a big neural networks that classify each label. It made us remember the good old days, when we used to watch the *Power Rangers series*. They used to merge their zords (each Ranger had its own zord aka fighter robot) into a megazord which could defeat the bad guys. In our case, the megazord is the final model and the zords are the secondary models.


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
      
      
 ### video_to frame
 
 The file containing the frames extracted from the video is in the input directory. 
 </div>
 

      
 
    
