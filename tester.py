#!/bin/env python

import numpy as np
from CalcUtils import medfilt1
import CalcUtils as c
def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

if __name__ == '__main__':

    # 100 pseudo-random integers ranging from 1 to 100, plus three large outliers for illustration.
    x = list(np.ceil(np.random.rand(25)*100)) + [1000] + \
        list(np.ceil(np.random.rand(25)*100)) + [2000] + \
        list(np.ceil(np.random.rand(25)*100)) + [3000] + \
        list(np.ceil(np.random.rand(25)*100))

    #---------------------------------------------------------------
    L = 2

    print 'L:',L
    print 'x:',x

    xout = medfilt1(x,10)

    if xout != None:
        print 'xout:',list(xout)
        
        try:
            import pylab as pl
            pl.subplot(2,1,1)
            pl.plot(x)
            pl.plot(xout)
            pl.grid(True)
            y1min = np.min(xout)*.5
            y1max = np.max(xout)*2
            pl.legend(['x (pseudo-random)','xout'])
            pl.title('median filter with window length ' + str(L) + ' (removes outliers, tracks remaining signal)')
        except:
            print 'pylab exception: not plotting results.'
    #---------------------------------------------------------------
    L = 103

    print 'L:',L
    print 'x:',x

    xout = runningMeanFast(x,10)

    if xout != None:
        print 'xout:',list(xout)
        
        try:
            pl.subplot(2,1,2)
            pl.plot(x)
            pl.plot(xout)
            pl.grid(True)
            y2min = np.min(xout)*.5
            y2max = np.max(xout)*2
            pl.legend(['same x (pseudo-random)','xout'])
            pl.title('median filter with window length ' + str(L) + ' (removes outliers and noise)')
        except:
            pass
    #---------------------------------------------------------------
    try:
        pl.subplot(2,1,1)
        pl.ylim([min(y1min,y2min),max(y1max,y2max)])
        pl.subplot(2,1,2)
        pl.ylim([min(y1min,y2min),max(y1max,y2max)])
        pl.show()
    except:
        pass