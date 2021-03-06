import cv2
from tqdm import tqdm
import os

def listdir_nohidden(path, mov_only = True):
    if mov_only :return [el for el in os.listdir(path) if not el.startswith(".") and  (el.endswith(".MOV") or el.endswith(".mov"))]
    else : return [el for el in os.listdir(path) if not el.startswith(".")]

#fps_goal is the number of frame per second the user wants to extract
def framer(directory, fps_goal = 2, rescale=False, shape=(128,128)):
    list_object = listdir_nohidden(directory, mov_only=False)
    print(list_object)
    l_count= []
    for object in list_object :
        compt =0
        directory_= directory+"/"+object
        for filename in tqdm(listdir_nohidden(directory_)):
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
                    #os.makedirs(directory_+object, exist_ok=True)
                    cv2.imwrite(directory_+"/"+object+"_"+str(compt) + '.jpg', frame)
                    compt+=1
                i += 1
            cap.release()
            cv2.destroyAllWindows()
        l_count.append((object,compt))

    for object,compt in l_count:
        print("The class " + object + " contains " + str(compt) + " images.")


# main directory
directory = "/Volumes/WD_BLACK/ressources/train_set/gache"

#for classe in listdir_nohidden(directory, mov_only=False):
framer(fps_goal=4,  directory=directory, rescale=True, shape=(256,256))

# [o_o]