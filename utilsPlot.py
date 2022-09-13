'''
Created by Francesco
29 November 2021
'''
#functions and script to compute correlations in space and time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
import os

#################################### plotting ##################################
def plotErrorBar(ax, x, y, err, xlabel, ylabel, logx = False, logy = False):
    ax.errorbar(x, y, err, marker='o', color='k', markersize=7, markeredgecolor='k', markeredgewidth=0.7, linewidth=1.2, elinewidth=1, capsize=4)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    if(logx == True):
        ax.set_xscale('log')
    if(logy == True):
        ax.set_yscale('log')
    plt.tight_layout()

def plotCorrelation(x, y, ylabel, xlabel = "$Distance,$ $r$", logy = False, logx = False, color = 'k', show = True):
    fig = plt.figure(0, dpi = 120)
    ax = fig.gca()
    ax.plot(x, y, linewidth=1.5, color=color, marker='.')
    if(logy == True):
        ax.set_yscale('log')
    if(logx == True):
        ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    plt.tight_layout()
    if(show == True):
        plt.pause(1)

def getStepList(numFrames, firstStep, stepFreq):
    maxStep = int(firstStep + stepFreq * numFrames)
    stepList = np.arange(firstStep, maxStep, stepFreq, dtype=int)
    if(stepList.shape[0] < numFrames):
        numFrames = stepList.shape[0]
    else:
        stepList = stepList[-numFrames:]
    return stepList

def computeTau(data, threshold=np.exp(-1)):
    relStep = np.argwhere(data[:,2]>threshold)[-1,0]
    if(relStep + 1 < data.shape[0]):
        t1 = data[relStep,0]
        t2 = data[relStep+1,0]
        ISF1 = data[relStep,2]
        ISF2 = data[relStep+1,2]
        slope = (ISF2 - ISF1)/(t2 - t1)
        intercept = ISF2 - slope * t2
        return (np.exp(-1) - intercept)/slope
    else:
        return data[relStep,0]

def computeDeltaChi(data):
    maxStep = np.argmax(data[:,3])
    maxChi = np.max(data[:,3])
    if(maxStep + 1 < data.shape[0]):
        # find values of chi above the max/2
        domeSteps = np.argwhere(data[:,3]>maxChi*0.5)
        t1 = domeSteps[0]
        t2 = domeSteps[-1]
        return t2 - t1
    else:
        return 0


if __name__ == '__main__':
    print("library for plotting utilities")
