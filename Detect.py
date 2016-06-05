import numpy as np
import cv2

import matplotlib.pyplot as plt

LASER_INTEVAL_MINUTES = 3

COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (51, 153, 255)
COLOR_GREEN = (0, 255, 0)

MOVEMENT_THRESHOLD = 10

#cap = cv2.VideoCapture('C:\\Users\\amotz\\PycharmProjects\\openCv\\4dms_cocaine_1st_laser.mpg')
filePath = "C:\\Users\\amotz\\PycharmProjects\\openCv\\4DMS_no_cocaine.mpg"
cap = cv2.VideoCapture(filePath)

print cap.isOpened()

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


def unit_vector(vector):
    """ Returns the unit vector of the vector.
    @type vector: tuple

    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

    @type v1: tuple
    @type v2: tuple
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class MouseData:



    def __init__(self,history_size):
        self.history = []

        self.movement_sum = 0
        self.angle_sum = 0
        self.avg_angle = 0
        self.currentTime = 0
        self.data = {'x': [], 'y': []}

        for i in range(history_size):
            self.history.append((0, 0))

    def update_history(self,point):
        self.history.pop()
        self.history.insert(0, point)

    def get_super_avg(self):
        super_post_arr = (0, 0)
        for v in self.history:
            super_post_arr = np.add(super_post_arr, v)
        return tuple(super_post_arr/len(self.history))



print params
detector = cv2.SimpleBlobDetector_create(params)
prev_frame = cap.read()
last_center_point = None
print cap.get(cv2.CAP_PROP_POS_MSEC)

mouse = []
LASER = 1
NORMAL = 1
mouse_data_normal = MouseData(5)
mouse_data_laser = MouseData(5)

currentMouse = mouse_data_normal


def paint_graph():
    plt.plot(mouse_data_normal.data['x'], mouse_data_normal.data['y'], mouse_data_laser.data['x'],
             mouse_data_laser.data['y'])
    plt.ylabel("Tremor average")
    plt.xlabel("Total distance")
    plt.show()


while cap.isOpened():
    ret ,frame = cap.read()


    if ret:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        keypoints = detector.detect(frame)

        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), COLOR_RED,
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if len(keypoints) == 1 :
            center_point = tuple([int(i) for i in keypoints[0].pt])
            cv2.circle(im_with_keypoints, center_point, 3, COLOR_GREEN)

            if last_center_point is None:
                last_center_point = center_point

            # find vector direction
            v1 = np.subtract(center_point, last_center_point)

            # "Normalize" vector (mostly for drawing purposes )
            v1 = tuple(v1*5)

            mouse_data_normal.update_history(v1)
            super = mouse_data_normal.get_super_avg()

            super_end_point = tuple(np.add(center_point, super))

            vector_end_point = tuple(np.add(center_point, v1))

            if np.linalg.norm(v1) > MOVEMENT_THRESHOLD:
                cv2.arrowedLine(im_with_keypoints, center_point, vector_end_point, COLOR_GREEN)
            if np.linalg.norm(super) > MOVEMENT_THRESHOLD:
                cv2.arrowedLine(im_with_keypoints, center_point, super_end_point, COLOR_ORANGE, 2)

            last_center_point = center_point
            res = angle_between(v1, super)


            if not np.isnan(res) and (np.linalg.norm(v1) > MOVEMENT_THRESHOLD and np.linalg.norm(super) > MOVEMENT_THRESHOLD):
                #print "angle_between(v1,super) {}".format(angle_between(v1,super))
                currentMouse.movement_sum += np.linalg.norm(v1)
                currentMouse.angle_sum += angle_between(v1,super)
                currentMouse.avg_angle = currentMouse.angle_sum/currentMouse.movement_sum
                currentMouse.data['x'].append(currentMouse.movement_sum)
                currentMouse.data['y'].append(currentMouse.avg_angle * 1000)

        currentTime = cap.get(cv2.CAP_PROP_POS_MSEC)
        cv2.putText(im_with_keypoints, "avg angle {}".format(currentMouse.avg_angle*1000), (0, 70), 0, .5, COLOR_RED)
        cv2.putText(im_with_keypoints, "movement sum {}".format(currentMouse.movement_sum), (0, 100), 0, .5, COLOR_RED)
        cv2.putText(im_with_keypoints, "Time " + str(currentTime), (0, 20), 0, .5, COLOR_RED)

        if (currentTime // (LASER_INTEVAL_MINUTES * 60 * 1000)) % 2 == 1:
            if currentMouse == mouse_data_normal :
                print "Context switch changing from mouse_data_normal to mouse_data_laser "
            currentMouse = mouse_data_laser
        else:
            if currentMouse == mouse_data_laser:
                print "Context switch changing from to mouse_data_laser mouse_data_normal"
            currentMouse = mouse_data_normal


        cv2.imshow("Keypoints", im_with_keypoints)

        k = cv2.waitKey(1)
        if k != -1 :
            print k
            if k == 113:
                break
            elif k == 112:
                paint_graph()

    else:
        break

print "file name= {}".format(filePath)
print "Finish total distance={} avg angle={} endTime={}".format(currentMouse.movement_sum, currentMouse.avg_angle * 1000, currentMouse.currentTime)


paint_graph()

cv2.destroyAllWindows()
cap.release()