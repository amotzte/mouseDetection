import numpy as np
import cv2

cap = cv2.VideoCapture('C:\\Users\\amotz\\PycharmProjects\\openCv\\vlc-record.mp4.mp4')
print cap.isOpened()


# cap.set(1000,cv2.CAP_PROP_POS_MSEC) = {bool} False
# getattr(cv2, "CAP_PROP_" + "POS_MSEC") = {int} 0
# take first frame of the video
#ret,frame = cap.read()

# setup initial location of window
#r,h,c,w = 250,90,400,125  # simply hardcoded the values
#track_window = (c,r,w,h)

# # set up the ROI for tracking
# roi = frame[r:r+h, c:c+w]
# hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
# term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 20
params.blobColor  = 0

params.maxArea=4000
params.minArea=1000
params.minThreshold = 0
params.maxThreshold = 30
params.minRepeatability=2

params.filterByConvexity = True
params.maxConvexity = 3
params.minConvexity = 0.6
filterByCircularity =  True

minCircularity = 0.800000011921

#params.filterByInertia = True
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


            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
            >>> angle_between((1, 0), (0, 1))
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class Direction:

    history = []

    def __init__(self,history_size):
        for i in range(history_size):
            self.history.append((0, 0))

    def update_history(self,point):
        self.history.pop()
        self.history.insert(0, point)

    def get_super_avg(self):
        super_post = (0, 0)
        for v in self.history:
            super_post = tuple(np.add(super_post, v))
        return tuple(np.divide(super_post, len(self.history)))


print params
#params.maxThreshold = 100
detector = cv2.SimpleBlobDetector_create(params)
prev_frame = cap.read()
last_center_point = (0, 0)

#print angle_between((0,5),(95,0))
general_direction = Direction(7)
imidiate_direction = Direction(7)
while(cap.isOpened()):
    ret ,frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    if ret:
        keypoints = detector.detect(frame)

        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.circle(im_with_keypoints, (int(keypoints[0].pt[0]), int(keypoints[0].pt[1])), 3, (0, 255, 0))
        #if (im_with_keypoints.)
        if (len(keypoints)) == 1 :
            center_point = tuple([int(i) for i in keypoints[0].pt])
            cv2.circle(im_with_keypoints, center_point, 3, (0, 255, 0))

            # find vector direction
            v1 = tuple(np.subtract(center_point, last_center_point))

            # "Normalize" vector (mostelly for drawing purposes )
            v1 = tuple(np.multiply(v1, 5))

            general_direction.update_history(v1)
            super = general_direction.get_super_avg()

            super_end_point = tuple(np.add(center_point, super))
            #print superposition

            vector_end_point = tuple(np.add(center_point, v1))

            if np.linalg.norm(v1) > 20:
                cv2.arrowedLine(im_with_keypoints, center_point, vector_end_point, (0, 255, 0))
                print "v1 {}".format(np.linalg.norm(v1))
            if np.linalg.norm(super) > 20:
                cv2.arrowedLine(im_with_keypoints, center_point, super_end_point, (255, 0, 0),2)
                print "super pos {}".format(np.linalg.norm(super))

            last_center_point = center_point
            res = angle_between(v1,super)
            if not np.isnan(res) and np.linalg.norm(v1) > 20 and np.linalg.norm(super) > 20 :
                print "angle_between(v1,super) {}".format(angle_between(v1,super))


            #if (np.linalg.norm(delta)):



        #cv2.namedWindow("", cv2.WINDOW_NORMAL)

        cv2.imshow("Keypoints", im_with_keypoints)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    else:
        break

cv2.destroyAllWindows()
cap.release()