# Megazord

[![Unit tests](https://github.com/iSab01/megazord/actions/workflows/python-app.yml/badge.svg)](https://github.com/iSab01/megazord/actions/workflows/python-app.yml)

Deep learning project realised by Lucas Saban and Augustin Cramer for Soft Next. 

Please mention us if you use this code.  
Feel free to contact us at : lucas.saban@ensae.fr or augustin.cramer@ensae.fr. 

**The goal is to classify the maximum amount of industrial pieces while minimizing errors.**

## Versions

IDE : PyCharm

Python 3.9 

tensorflow 2.5.0

coremltools 4.1

opencv 4.5.2.54

tqdm 4.61.1

## The app

The app is based on code sample found on the Apple Developper website. It's not energy efficient and is quite heavy.  
For now, it's only for visualization but we might improve it later on.

## Techniques used

In this project we used Data Augmentation, Transfer Learning (Inception V3), Fine Tuning and multi inputs/outputs neural networks
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
 
 

      
 
    
