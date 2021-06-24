# megazord

Deep learning project realised with Augustin Cramer for Soft Next.

## Why Megazord and zords ? 

Well, it is because of the architecture of our neural network? It combines several small neural networks, trained to distinguish object from the same class (ex: motor_s and motor_m) to make a big neural networks that classify each label. It made us remember the good old days, when we used to watch the Power Rangers series. They used to merge their zords (each Ranger had its own zord aka fighter robot) into a megazord which could defeat the bad guys. In our case, the megazord is the final model and the zords are the secondary models


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
 
 The file containing the output data is in the input directory

      
 
    
