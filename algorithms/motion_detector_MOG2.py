import cv2
import pandas
from datetime import datetime

def motion_detector_MOG2():
    motion_list = [None, None]
    time_list = []
    df = df = pandas.DataFrame(columns = ['Start Motion', 'End Motion']) # will turn into csv file later

    previous_frame = None

    #img_brg = cv2.VideoCapture("/Users/weifengcao/Documents/mp4/traffic.mp4")
    img_brg = cv2.VideoCapture(0)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    frame_width = int(img_brg.get(3))
    frame_height = int(img_brg.get(4))

    size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('/Users/weifengcao/Documents/mp4' + motion_detector_MOG2.__name__ + '.avi', fourcc, 10.0, size)

    while True:
        # Load image and convert to RGB
        check, cur_frame = img_brg.read()
        motion = 0;

        fgmask = fgbg.apply(cur_frame)

        # Find contours
        contours, _ = cv2.findContours(image=fgmask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                # skip if too small
                continue
            motion = 1
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img=cur_frame, pt1=(x,y), pt2=(x+w, y+h), color=(0,255,0), thickness=2)

        out.write(cur_frame)
        #Append list of motions for each new motion detected and set start/end time
        motion_list.append(motion)
        motion_list = motion_list[-2:]

        if motion_list[-1] == 1 and motion_list[-2] == 0: time_list.append(datetime.now()) #start motion's time
        if motion_list[-1] == 0 and motion_list[-2] == 1: time_list.append(datetime.now()) #end motion's time

        cv2.imshow('Motion detector', cur_frame)

        if (cv2.waitKey(10) == 27):
            if motion == 1: time_list.append(datetime.now()) #catch the last movement
            break
     # Put list of motions into the dataframe and save it as csv file
    for i in range(0,len(time_list) - 1, 2):
        df = df.append({"Start Motion": time_list[i], "End Motion": time_list[i+1]}, ignore_index = True)

    df.to_csv("Time_of_Motions.csv")
    # Turn off cam and close cv2's windows
    img_brg.release()
    out.release()
    cv2.destroyAllWindows()

