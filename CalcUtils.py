import math
import time

import numpy as np



class FpsCounter:
    def __init__(self):
        self.num_of_frames = 0
        self.FPS_NUM = 5
        self.start = time.time()
        self.current_fps = 0

    def frame(self):
        self.num_of_frames+=1
        if self.num_of_frames == self.FPS_NUM:
            total_sec = time.time() - self.start
            self.current_fps = self.num_of_frames / total_sec
            self.start=time.time()
            self.num_of_frames=0
        return self.current_fps


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
    rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return math.degrees(rad)


def runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def medfilt1(x=None,L=None):

    '''
    a simple median filter for 1d numpy arrays.

    performs a discrete one-dimensional median filter with window
    length L to input vector x. produces a vector the same size
    as x. boundaries handled by shrinking L at edges; no data
    outside of x used in producing the median filtered output.
    (upon error or exception, returns None.)

    inputs:
        x, Python 1d list or tuple or Numpy array
        L, median filter window length
    output:
        xout, Numpy 1d array of median filtered result; same size as x

    bdj, 5-jun-2009
    '''

    # input checks and adjustments --------------------------------------------
    try:
        N = len(x)
        if N < 2:
            print 'Error: input sequence too short: length =',N
            return None
        elif L < 2:
            print 'Error: input filter window length too short: L =',L
            return None
        elif L > N:
            print 'Error: input filter window length too long: L = %d, len(x) = %d'%(L,N)
            return None
    except:
        print 'Exception: input data must be a sequence'
        return None

    xin = np.array(x)
    if xin.ndim != 1:
        print 'Error: input sequence has to be 1d: ndim =',xin.ndim
        return None

    xout = np.zeros(xin.size)

    # ensure L is odd integer so median requires no interpolation
    L = int(L)
    if L%2 == 0: # if even, make odd
        L += 1
    else: # already odd
        pass
    Lwing = (L-1)/2

    # body --------------------------------------------------------------------

    for i,xi in enumerate(xin):

        # left boundary (Lwing terms)
        if i < Lwing:
            xout[i] = np.median(xin[0:i+Lwing+1]) # (0 to i+Lwing)

        # right boundary (Lwing terms)
        elif i >= N - Lwing:
            xout[i] = np.median(xin[i-Lwing:N]) # (i-Lwing to N-1)

        # middle (N - 2*Lwing terms; input vector and filter window overlap completely)
        else:
            xout[i] = np.median(xin[i-Lwing:i+Lwing+1]) # (i-Lwing to i+Lwing)

    return xout