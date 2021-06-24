import cv2
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt



#fps_goal is the number of frame per second the user wants to extract
def framer(list_object, directory, fps_goal = 2, rescale=False, shape=(128,128)):
    for object in list_object :
        j=0
        directory_= directory+"/"+object
        for filename in tqdm(os.listdir(directory_)):
            if filename.endswith(".MOV") or filename.endswith(".mov") :
                #Import the video file
                cap = cv2.VideoCapture(directory_+"/"+filename)
                #Get the fps of the video
                fps = cap.get(cv2.CAP_PROP_FPS)
                #we take a frame every step
                step = int(fps/fps_goal)
                print(filename + " is being treated")
                i = 0
                #the loop ends when we reach the end of the video
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    if i % step == 0:
                        #shape is RGB so frame.shape is (1920,1080,3).
                        # We only need the first dimension
                        shape_frame = frame.shape[:2]
                        w = min(shape_frame)
                        l = max(shape_frame)
                        d = int((l-w)/2)
                        #the following condition enables the function to crop
                        # the frame wether they are in a landscape or portrait format
                        if w == shape_frame[0]: frame = frame[:,d:d+w]
                        else : frame = frame[d:d+w, :]
                        if rescale : frame = cv2.resize(frame, shape)
                        assert frame.shape[0]==frame.shape[1]
                        #creates the directory if it doesn't already exists.
                        os.makedirs(directory_+"/frames_"+object, exist_ok=True)
                        cv2.imwrite(directory_+"/frames_"+object+"/"+object+str(j)+"_"+str(i) + '.jpg', frame)
                    i += 1
                cap.release()
                cv2.destroyAllWindows()
            j+=1
    print("Directory done")

# Contains the name of the object  and must be the name of the folder it  is in.
#list_object = ["handle_lock","ball_bearing", "handle_lockless", "motor_m", "motor_S"]
list_object=["motor_S"]
# main directory
directory = "/Users/lucas/test_framer"

framer(list_object=list_object,fps_goal=4,  directory=directory, rescale=True, shape=(256,256))
