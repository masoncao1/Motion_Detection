import cv2
import numpy as np
import pandas
from datetime import datetime

def motion_detector_FrameDiff():
    motion_list = [None, None]
    time_list = []
    df = df = pandas.DataFrame(columns = ['Start Motion', 'End Motion']) # will turn into csv file later

    previous_frame = None

    img_brg = cv2.VideoCapture("/Users/weifengcao/Documents/mp4/traffic.mp4")
    #img_brg = cv2.VideoCapture(0)
    frame_width = int(img_brg.get(3))
    frame_height = int(img_brg.get(4))

    size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('/Users/weifengcao/Documents/mp4' + motion_detector_FrameDiff.__name__ + '.avi', fourcc, 10.0, size)

    while True:
        # Load image and convert to RGB
        check, cur_frame = img_brg.read()
        motion = 0;

        # Prepare image: grayscale and blur
        prepared_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        prepared_frame = cv2.GaussianBlur(src = prepared_frame, ksize = (5,5), sigmaX = 0)

        # Set previous frame and continue if there is None
        if (previous_frame is None):
            # First frame
            previous_frame = prepared_frame
            continue

        # calculate difference and update previous frame
        diff_frame = cv2.absdiff(src1 = previous_frame, src2 = prepared_frame)
        previous_frame = prepared_frame

        # Dilute image slightly to make differences easier to see; more suitable for contour detection
        kernel = np.ones((5,5))
        diff_frame = cv2.dilate(diff_frame, kernel, 2)

        # Only take different areas that are different enough (>30 / 255)
        thresh_frame = cv2.adaptiveThreshold(diff_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

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
    for i in range(0,len(time_list) - 1,2):
        df = df.append({"Start Motion": time_list[i], "End Motion": time_list[i+1]}, ignore_index = True)

    df.to_csv("Time_of_Motions.csv")
    # Turn off cam and close cv2's windows
    img_brg.release()
    out.release()
    cv2.destroyAllWindows()

