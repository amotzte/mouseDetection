import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from CalcUtils import FpsCounter, angle_between,medfilt1

GENERAL_MOVEMENT_HISTORY = 5
LASER_INTEVAL_MINUTES = 3
NUM_OF_CYCLES = 3
VECTOR_SIZE_NORMALIZE = 5 # "Normalize" vectors size, avoid working with small numbers that can't later be divided for average.
SHORT_MOVEMENT_THRESHOLD = VECTOR_SIZE_NORMALIZE *2
GENERAL_MOVEMENT_THRESHOLD = VECTOR_SIZE_NORMALIZE *2

COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (51, 153, 255)
COLOR_GREEN = (0, 255, 0)


#cap = cv2.VideoCapture('C:\\Users\\amotz\\PycharmProjects\\openCv\\4dms_cocaine_1st_laser.mpg')
#filePath = "C:\\Users\\amotz\\PycharmProjects\\openCv\\vlc-record-2016-05-29-11h07m23s-3DMS NO COCAINE.mpg-.mp4"
filePath = "C:\\Users\\amotz\\PycharmProjects\\openCv\\4DMS_no_cocaine.mpg"
cap = cv2.VideoCapture(filePath)

print cap.isOpened()

# Detection parameters
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 20
params.blobColor = 0
params.maxArea=4000
params.minArea=1000
params.minThreshold = 0
params.maxThreshold = 30
params.minRepeatability=2
params.filterByConvexity = True
params.maxConvexity = 3
params.minConvexity = 0.6


"""
blobColor = {int} 0
filterByArea = {bool} True
filterByCircularity = {bool} False
filterByColor = {bool} True
filterByConvexity = {bool} False
filterByInertia = {bool} False
maxArea = {float} 5000.0
maxCircularity = {float} 3.40282346639e+38
maxConvexity = {float} 3.40282346639e+38
maxInertiaRatio = {float} 3.40282346639e+38
maxThreshold = {float} 30.0
minArea = {float} 300.0
minCircularity = {float} 0.800000011921
minConvexity = {float} 0.949999988079
minDistBetweenBlobs = {float} 10.0
minInertiaRatio = {float} 0.10000000149
minRepeatability = {long} 2
minThreshold = {float} 0.0
thresholdStep = {float} 10.0

"""


class MouseData:
    def __init__(self,history_size):
        self.history = []
        self.movement_sum = 0
        self.angle_sum = 0
        self.avg_angle = 0
        self.data = {'x': [],
                     'y': [],
                     'vSize': []}

        for i in range(history_size):
            self.history.append(np.array([0, 0]))

    def update_history(self,point):
        self.history.pop()
        self.history.insert(0, point)

    def get_super_avg(self):
        return np.mean(self.history, axis=0, dtype=np.int32)


print params
detector = cv2.SimpleBlobDetector_create(params)
prev_frame = cap.read()
last_center_point = None
print cap.get(cv2.CAP_PROP_POS_MSEC)

mouse = []
LASER = 1
NORMAL = 1
mouse_data_normal = MouseData(GENERAL_MOVEMENT_HISTORY)
mouse_data_laser = MouseData(GENERAL_MOVEMENT_HISTORY)
currentMouse = mouse_data_normal
remain_context_switch = NUM_OF_CYCLES * 2



def paint_graph():

    fig = plt.figure("Movement graph")
    fig.suptitle("{}, {} Cycles ".format(os.path.basename(filePath), NUM_OF_CYCLES), fontsize=12)
    plt.plot(mouse_data_normal.data['x'], mouse_data_normal.data['y'], mouse_data_laser.data['x'],
             mouse_data_laser.data['y'])
    plt.ylabel("Degree")
    plt.xlabel("Time")


    fig = plt.figure("All graphs")
    fig.suptitle("{}, {} Cycles ".format(os.path.basename(filePath), NUM_OF_CYCLES), fontsize=12)

    plt.subplot(221)
    mean = 100

    plt.title('Meadian angles {}'.format(mean))
    yy = medfilt1(mouse_data_normal.data['y'],mean)
    if len(mouse_data_laser.data['y']) < mean:
        plt.plot(mouse_data_normal.data['x'], tuple(yy), 'b')
    else:
        yy1 = medfilt1(mouse_data_laser.data['y'], mean)
        plt.plot(mouse_data_normal.data['x'], tuple(yy), 'b', mouse_data_laser.data['x'], tuple(yy1), 'g')
    plt.ylabel("Degree")
    plt.xlabel("Time")


    plt.subplot(222)
    plt.title('Angels')
    plt.plot(mouse_data_normal.data['x'], mouse_data_normal.data['y'], mouse_data_laser.data['x'],
             mouse_data_laser.data['y'])
    plt.ylabel("Degree")
    plt.xlabel("Time")

    plt.subplot(223)
    plt.title("Meadian speed".format(mean))
    vSize = medfilt1(mouse_data_normal.data['vSize'], mean)
    if len(mouse_data_laser.data['y']) < mean:
        plt.plot(mouse_data_normal.data['x'], tuple(vSize), 'b')
    else:
        vSize1 = medfilt1(mouse_data_laser.data['vSize'], mean)
        plt.plot(mouse_data_normal.data['x'], vSize, mouse_data_laser.data['x'], vSize1)
    plt.ylabel("speed")
    plt.xlabel("Time")


    plt.subplot(224)
    plt.title('Speed')
    plt.plot(mouse_data_normal.data['x'], mouse_data_normal.data['vSize'], mouse_data_laser.data['x'],
             mouse_data_laser.data['vSize'])
    plt.ylabel("speed")
    plt.xlabel("Time")


    #if mouse_data_laser.data['y'] != []:
    #    plt.hist(mouse_data_laser.data['y'],color='green')


    plt.show()


fpsCounter = FpsCounter()
while cap.isOpened() and remain_context_switch > 0:
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        keypoints = detector.detect(frame)

        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), COLOR_RED,
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        currentTime = cap.get(cv2.CAP_PROP_POS_MSEC)

        if len(keypoints) == 1 :
            center_point = tuple([int(i) for i in keypoints[0].pt])
            cv2.circle(im_with_keypoints, center_point, 3, COLOR_GREEN)

            if last_center_point is None:
                last_center_point = center_point

            # find vector direction
            v1 = np.subtract(center_point, last_center_point)
            v1 = tuple(v1 * VECTOR_SIZE_NORMALIZE)

            currentMouse.update_history(v1)
            super = currentMouse.get_super_avg()

            super_end_point = tuple(np.add(center_point, super))

            vector_end_point = tuple(np.add(center_point, tuple(v1)))

            if np.linalg.norm(v1) > SHORT_MOVEMENT_THRESHOLD:
                cv2.arrowedLine(im_with_keypoints, center_point, vector_end_point, COLOR_GREEN)
            if np.linalg.norm(super) > GENERAL_MOVEMENT_THRESHOLD:
                cv2.arrowedLine(im_with_keypoints, center_point, super_end_point, COLOR_ORANGE, 2)

            last_center_point = center_point
            res = angle_between(v1, super)


            if not np.isnan(res) and (np.linalg.norm(v1) > SHORT_MOVEMENT_THRESHOLD and np.linalg.norm(super) > GENERAL_MOVEMENT_THRESHOLD):
                currentMouse.movement_sum += np.linalg.norm(v1)

                assert angle_between(v1, super) >= 0

                #currentMouse.angle_sum += angle_between(v1,super)
                #currentMouse.avg_angle = currentMouse.angle_sum/currentMouse.movement_sum
                currentMouse.data['x'].append(currentTime/(1000*60))
                currentMouse.data['y'].append(angle_between(v1, super))
                currentMouse.data['vSize'].append(np.linalg.norm(super))

                #currentMouse.data['y'].append(currentMouse.avg_angle * 1000)

        fpsCounter.frame()

        cv2.putText(im_with_keypoints, "FPS {}".format(fpsCounter.current_fps), (0, 70), 0, .5, COLOR_RED)
        cv2.putText(im_with_keypoints, "movement sum {}".format(currentMouse.movement_sum), (0, 100), 0, .5, COLOR_RED)
        cv2.putText(im_with_keypoints, "Time " + str(currentTime), (0, 20), 0, .5, COLOR_RED)

        if (currentTime // (LASER_INTEVAL_MINUTES * 60 * 1000)) % 2 == 1:
            if currentMouse == mouse_data_normal :
                print "Context switch changing from mouse_data_normal to mouse_data_laser "
                remain_context_switch-=1
            currentMouse = mouse_data_laser
        else:
            if currentMouse == mouse_data_laser:
                print "Context switch changing from to mouse_data_laser mouse_data_normal"
                remain_context_switch -= 1
            currentMouse = mouse_data_normal


        cv2.imshow("Keypoints", im_with_keypoints)

        k = cv2.waitKey(1)
        if k != -1 :
            print k
            if k == 113:
                break
            elif k == 112:
                try:
                    paint_graph()
                except Exception, e:
                    print e
            elif k == 2555904:
                cap.set(cv2.CAP_PROP_POS_MSEC,currentTime + 30000)
            elif k == 2424832:
                cap.set(cv2.CAP_PROP_POS_MSEC, currentTime - 30000)


    else:
        break

print "file name= {}".format(filePath)
print "Finish total distance={} avg angle={}".format(currentMouse.movement_sum, currentMouse.avg_angle * 1000)


paint_graph()

cv2.destroyAllWindows()
cap.release()