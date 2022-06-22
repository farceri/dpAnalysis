'''
Created by Francesco
22 March 2022
'''
#functions and script to visualize a 2d dpm packing
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
import itertools
import sys
import os
import computeCorrelation
import shapeDescriptors

def getStepList(numFrames, firstStep, stepFreq):
    maxStep = int(firstStep + stepFreq * numFrames)
    stepList = np.arange(firstStep, maxStep, stepFreq, dtype=int)
    if(stepList.shape[0] < numFrames):
        numFrames = stepList.shape[0]
    else:
        stepList = stepList[-numFrames:]
    return stepList

def getContactDiff(dirName, numParticles, stepList):
    initialContacts = np.loadtxt(dirName + os.sep + "t" + str(stepList[0]) + "/contacts.dat", dtype=int)
    initialContacts = np.flip(np.sort(initialContacts, axis=1), axis=1)
    finalContacts = np.loadtxt(dirName + os.sep + "t" + str(stepList[-1]) + "/contacts.dat", dtype=int)
    finalContacts = np.flip(np.sort(finalContacts, axis=1), axis=1)
    contactdiff = np.zeros(numParticles)
    for i in range(numParticles):
        isdiff = True
        for c in initialContacts[i]:
            if(c != -1):
                for b in finalContacts[i]:
                    if(c == b):
                        isdiff = False
            if(isdiff == True):
                contactdiff[i] += 1
    return contactdiff

def plotErrorBar(ax, x, y, err, xlabel, ylabel, color = 'k', logx = False, logy = False):
    ax.errorbar(x, y, err, marker='.', color=color, markersize=5, markeredgecolor='k', markeredgewidth=0.7, linewidth=1.2, elinewidth=1, capsize=4)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    if(logx == True):
        ax.set_xscale('log')
    if(logy == True):
        ax.set_yscale('log')
    plt.tight_layout()

############################ plot shape correlations ###########################
def plotElongationVSActivity(dirName, figureName, numBins=20):
    dataSetList = np.array(["5e-04", "1e-03", "1.3e-03", "1.6e-03", "2e-03", "3e-03"])
    colorList = [[0.5,0,1], 'b', [0,1,0.5], 'g', 'r', [1,0.5,0]]
    fig = plt.figure(dpi = 120)
    ax = fig.gca()
    legendList  = []
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/corr-log.dat")):
            Teff = computeCorrelation.readFromParams(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/", "temperature")
            legendList.append("$T_{eff}=$" + str(np.format_float_scientific(Teff, 2)))
            boxSize = np.loadtxt(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/" + os.sep + "boxSize.dat")
            nv = np.loadtxt(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/" + os.sep + "numVertexInParticleList.dat", dtype=int)
            dirList = np.sort(computeCorrelation.getDirectories(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/"))
            elong = []
            for dir in dirList[-9:]:
                elong.append(shapeDescriptors.computeElongation(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/" + dir, boxSize, nv))
            data = np.array(elong)
            pdf, bins = shapeDescriptors.computePDF(data, np.linspace(np.min(data), np.max(data), numBins))
            ax.plot(bins, pdf, linewidth=1.2, color = colorList[i])
    ax.legend(legendList, loc ="upper right", fontsize=11)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$elongation$", fontsize=15)
    ax.set_ylabel("$PDF$", fontsize=15)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/elong-v0-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotElongationVSPhi(dirName, v0String, figureName, numBins=20):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    colorList = [[0.5,0,1], 'b', [0,0.5,1], 'g', [0.5,1,0], [1,0.8,0], [1,0.5,0], 'r', [1,0,0.5], [1,0.5,1]]
    colorList = cm.get_cmap('viridis', dataSetList.shape[0])
    fig = plt.figure(dpi = 120)
    ax = plt.gca()
    legendList  = []
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics/corr-log.dat")):
            phi = computeCorrelation.readFromParams(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics", "phi")
            legendList.append("$\\varphi=$" + str(np.format_float_positional(phi, 4)))
            boxSize = np.loadtxt(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics" + os.sep + "boxSize.dat")
            nv = np.loadtxt(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics" + os.sep + "numVertexInParticleList.dat", dtype=int)
            dirList = np.sort(computeCorrelation.getDirectories(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics"))
            elong = []
            for dir in dirList[-9:]:
                elong.append(shapeDescriptors.computeElongation(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics/" + dir, boxSize, nv))
            data = np.array(elong)
            pdf, bins = shapeDescriptors.computePDF(data, np.linspace(np.min(data), np.max(data), numBins))
            ax.plot(bins, pdf, linewidth=1.2, color = colorList(i/dataSetList.shape[0]))
    ax.legend(legendList, loc ="upper right", fontsize=11)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$elongation$", fontsize=15)
    ax.set_ylabel("$PDF$", fontsize=15)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/elong-phi-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotShapeMomentsVSActivity(dirName, figureName, numBins=20):
    dataSetList = np.array(["4e-03", "3e-03", "2e-03", "1e-03"])
    colorList = ['b', 'g', 'r', [1,0.5,0]]
    fig, ax = plt.subplots(1, 3, figsize = (14, 4), sharey = True, dpi = 120)
    legendList  = []
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/corr-log.dat")):
            Teff = computeCorrelation.readFromParams(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics", "temperature")
            legendList.append("$T_{eff}=$" + str(np.format_float_scientific(Teff, 2)))
            boxSize = np.loadtxt(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics" + os.sep + "boxSize.dat")
            nv = np.loadtxt(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics" + os.sep + "numVertexInParticleList.dat", dtype=int)
            dirList = np.sort(computeCorrelation.getDirectories(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics"))
            first = []
            second = []
            third = []
            for dir in dirList[-49:]:
                moments = shapeDescriptors.computeShapeMoments(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/" + dir, boxSize, nv)
                first.append(moments[0])
                second.append(moments[1])
                third.append(moments[2])
            data = np.array((np.array(first).flatten(), np.array(second).flatten(), np.array(third).flatten()))
            for m in range(data.shape[0]):
                pdf, bins = shapeDescriptors.computePDF(data[m], np.linspace(np.min(data[m]), np.max(data[m]), numBins))
                ax[m].plot(bins, pdf, linewidth=1.2, color = colorList[i])
    ax[2].legend(legendList, loc ="upper right", fontsize=12)
    for i in range(3):
        ax[i].tick_params(axis='both', labelsize=12)
    ax[0].set_xlabel("$shape$ $parameter$", fontsize=15)
    ax[1].set_xlabel("$shape$ $parameter$ $fluctuation$", fontsize=15)
    ax[2].set_xlabel("$shape$ $parameter$ $asymmetry$", fontsize=15)
    ax[0].set_ylabel("$PDF$", fontsize=15)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0)
    plt.savefig("/home/francesco/Pictures/dpm/shape-moments-v0-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotShapeMomentsVSPhi(dirName, v0String, figureName, numBins=20):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    colorList = [[0.5,0,1], 'b', [0,0.5,1], 'g', [0.5,1,0], [1,0.8,0], [1,0.5,0], 'r', [1,0,0.5], [1,0.5,1]]
    colorList = cm.get_cmap('viridis', dataSetList.shape[0])
    fig, ax = plt.subplots(1, 3, figsize = (12, 4), sharey = True, dpi = 120)
    legendList  = []
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics/corr-log.dat")):
            phi = computeCorrelation.readFromParams(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics", "phi")
            legendList.append("$\\varphi=$" + str(np.format_float_positional(phi, 4)))
            boxSize = np.loadtxt(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics" + os.sep + "boxSize.dat")
            nv = np.loadtxt(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics" + os.sep + "numVertexInParticleList.dat", dtype=int)
            dirList = np.sort(computeCorrelation.getDirectories(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics"))
            first = []
            second = []
            third = []
            for dir in dirList[-49:]:
                moments = shapeDescriptors.computeShapeMoments(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics/" + dir, boxSize, nv)
                first.append(moments[0])
                second.append(moments[1])
                third.append(moments[2])
            data = np.array((np.array(first).flatten(), np.array(second).flatten(), np.array(third).flatten()))
            for m in range(data.shape[0]):
                pdf, bins = shapeDescriptors.computePDF(data[m], np.linspace(np.min(data[m]), np.max(data[m]), numBins))
                ax[m].plot(bins, pdf, linewidth=1.2, color = colorList(i/dataSetList.shape[0]))
    ax[2].legend(legendList, loc ="upper right", fontsize=11)
    for i in range(3):
        ax[i].tick_params(axis='both', labelsize=12)
    ax[0].set_xlabel("$shape$ $parameter$", fontsize=15)
    ax[1].set_xlabel("$shape$ $parameter$ $fluctuation$", fontsize=15)
    ax[2].set_xlabel("$shape$ $parameter$ $asymmetry$", fontsize=15)
    ax[0].set_ylabel("$PDF$", fontsize=15)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0)
    plt.savefig("/home/francesco/Pictures/dpm/shape-moments-phi-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotInertiaVSActivity(dirName, figureName, numBins=20):
    dataSetList = np.array(["4e-03", "3e-03", "2e-03", "1e-03", "5e-04"])
    colorList = [[0.5,0,1], 'b', 'g', 'r', [1,0.5,0]]
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), sharey = True, dpi = 120)
    legendList  = []
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/corr-log.dat")):
            Teff = computeCorrelation.readFromParams(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics", "temperature")
            legendList.append("$T_{eff}=$" + str(np.format_float_scientific(Teff, 2)))
            boxSize = np.loadtxt(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics" + os.sep + "boxSize.dat")
            nv = np.loadtxt(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics" + os.sep + "numVertexInParticleList.dat", dtype=int)
            dirList = np.sort(computeCorrelation.getDirectories(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics"))
            first = []
            second = []
            for dir in dirList[-49:]:
                moments, _ = shapeDescriptors.computeInertiaTensor(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/" + dir, boxSize, nv, plot=False)
                first.append(moments[:,0])
                second.append(moments[:,1])
            data = np.array((np.array(first).flatten(), np.array(second).flatten()))
            for m in range(data.shape[0]):
                pdf, bins = shapeDescriptors.computePDF(data[m], np.linspace(np.min(data[m]), np.max(data[m]), numBins))
                ax[m].plot(bins, pdf, linewidth=1.2, color = colorList[i])
                ax[m].tick_params(axis='both', labelsize=14)
                #ax[m].set_xlim(-0.002,0.037)
                #ax[m].set_xticks((0, 0.01, 0.02, 0.03))
    ax[1].legend(legendList, loc ="upper right", fontsize=12)
    ax[0].set_xlabel("$e_x$", fontsize=17)
    ax[1].set_xlabel("$e_y$", fontsize=17)
    ax[0].set_ylabel("$PDF$", fontsize=17)
    #ax[0].set_ylim(-2,132)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.1)
    plt.savefig("/home/francesco/Pictures/dpm/shape-tensor-v0-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotInertiaVSPhi(dirName, v0String, figureName, numBins=20):
    dataSetList = np.array(["4", "5", "6", "7", "8", "9"])
    colorList = [[0.5,0,1], 'b', [0,0.5,1], 'g', [0.5,1,0], [1,0.8,0], [1,0.5,0], 'r', [1,0,0.5], [1,0.5,1]]
    colorList = cm.get_cmap('viridis', dataSetList.shape[0])
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), sharey = True, dpi = 120)
    legendList  = []
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics/corr-log.dat")):
            phi = computeCorrelation.readFromParams(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics", "phi")
            legendList.append("$\\varphi=$" + str(np.format_float_positional(phi, 4)))
            boxSize = np.loadtxt(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics" + os.sep + "boxSize.dat")
            nv = np.loadtxt(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics" + os.sep + "numVertexInParticleList.dat", dtype=int)
            dirList = np.sort(computeCorrelation.getDirectories(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics"))
            first = []
            second = []
            for dir in dirList[-49:]:
                moments, _ = shapeDescriptors.computeInertiaTensor(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics/" + dir, boxSize, nv, plot=False)
                first.append(moments[:,0])
                second.append(moments[:,1])
            data = np.array((np.array(first).flatten(), np.array(second).flatten()))
            for m in range(data.shape[0]):
                pdf, bins = shapeDescriptors.computePDF(data[m], np.linspace(np.min(data[m]), np.max(data[m]), numBins))
                ax[m].plot(bins, pdf, linewidth=1.2, color = colorList(i/dataSetList.shape[0]))
                ax[m].tick_params(axis='both', labelsize=14)
                #ax[m].set_xlim(-0.002,0.037)
                #ax[m].set_xticks((0, 0.01, 0.02, 0.03))
    ax[1].legend(legendList, loc ="upper right", fontsize=12)
    ax[0].set_xlabel("$e_x$", fontsize=17)
    ax[1].set_xlabel("$e_y$", fontsize=17)
    ax[0].set_ylabel("$PDF$", fontsize=17)
    #ax[0].set_ylim(-2,132)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.1)
    plt.savefig("/home/francesco/Pictures/dpm/inertia-phi-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotShapeCorrelation(dirName, figureName):
    dataSetList = np.array(["2e-04", "4e-04", "6e-04", "8e-04", "1e-03", "1.3e-03", "1.6e-03", "2e-03", "3e-03"])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0]+1)
    #colorList = [[0.5,0,1], 'b', 'g', 'r', [1,0.5,0]]
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(8,9), dpi = 100)
    #ax = plt.gca()
    legendList  = []
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/corr-shape.dat")):
            Teff = computeCorrelation.readFromParams(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/", "temperature")
            timeStep = computeCorrelation.readFromParams(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics", "dt")
            legendList.append("$v_0=$" + dataSetList[i])
            #legendList.append("$T_{eff}=$" + str(np.format_float_scientific(Teff, 2)))
            data = np.loadtxt(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/corr-shape.dat")
            #data = data[data[:,0]<8e05]
            ax[0].semilogx(data[:,0]*timeStep, data[:,1], linewidth=1.5, color = colorList(i/dataSetList.shape[0]))
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/corr-shape.dat")):
            data = np.loadtxt(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/corr-log.dat")
            ax[1].semilogx(data[:,0]*timeStep, data[:,2], linewidth=1.5, color = colorList(i/dataSetList.shape[0]))
    ax[1].legend(legendList, loc ="lower left", fontsize=15)
    #ax.set_ylim(-0.05, 1.05)
    ax[0].tick_params(axis='both', labelsize=17)
    ax[1].tick_params(axis='both', labelsize=17)
    ax[1].set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=20)
    ax[0].set_ylabel("$C_A (\\Delta t)$", fontsize=20)
    ax[1].set_ylabel("$ISF(\\Delta t)$", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/dpm/shapecorr-v0-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotShapeStressCorrelation(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04, numBins=30):
    stepList = getStepList(numFrames, firstStep, stepFreq)
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype=int)
    numParticles = nv.shape[0]
    # get list of most rearranging particles
    contactdiff = getContactDiff(dirName, numParticles, stepList)
    print(contactdiff[contactdiff>0])
    rearrangeList = np.argwhere(contactdiff>=2)
    rearrangeList = rearrangeList[:,0]
    particleList = np.arange(0,numParticles,1, dtype=int)
    print(rearrangeList)
    angles = []
    eigsShape = []
    eigsStress = []
    asphericity = []
    shapeparam = []
    eigvShapeMax = np.zeros((numParticles,2))
    eigvStressMax = np.zeros((numParticles,2))
    x = np.array([1,0])
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    #fig = plt.subplots(figsize=(9.5,4.5), dpi = 100)
    #ax = plt.gca()
    for step in stepList:
        if(os.path.isdir(dirName + os.sep + "t" + str(step))):
            eigs, eigvShape, _ = shapeDescriptors.computeInertiaTensor(dirName + os.sep + "t" + str(step), boxSize, nv, plot=False)
            asphericity.append(eigs[rearrangeList,1])
            perimeter = np.loadtxt(dirName + os.sep + "t" + str(step) + "/perimeters.dat")
            area = np.loadtxt(dirName + os.sep + "t" + str(step) + "/areas.dat")
            shapeparam.append(perimeter[rearrangeList]**2/(4*np.pi*area[rearrangeList]))
            eigs, eigvStress = shapeDescriptors.computeStressTensor(dirName + os.sep + "t" + str(step), nv, plot=False)
            eigsStress.append(eigs[rearrangeList,1])
            for i in rearrangeList:
                eigvShapeMax[i] = eigvShape[i,1]
                eigvStressMax[i] = eigvStress[i,1]
                angle = np.arccos(np.sum((eigvShapeMax[i]/np.linalg.norm(eigvShapeMax[i])) * (eigvStressMax[i]/np.linalg.norm(eigvStressMax[i]))))
                angles.append(np.degrees(angle))
    pdf, bins = shapeDescriptors.computePDF(angles, np.linspace(np.min(angles), np.max(angles), numBins))
    ax[0].plot(bins, pdf, linewidth=1.2, color = 'b')
    ax[1].scatter(shapeparam, eigsStress, color = 'b', linewidth=1.2, alpha=0.5, marker='o')
    ax[0].tick_params(axis="both", labelsize=14)
    ax[1].tick_params(axis="both", labelsize=14)
    ax[0].set_xlabel("$Stress-shape$ $angle,$ $\\theta$", fontsize=17)
    ax[0].set_ylabel("$PDF(\\theta)$", fontsize=17)
    ax[1].set_xlabel("$Shape$ $paramter,$ $A$", fontsize=17)
    ax[1].set_ylabel("$Stress,$ $s_{max}$", fontsize=17)
    ax[0].set_xlim(-5, 185)
    ax[0].set_ylim(-0.0002, 0.0132)
    ax[1].set_xlim(1.07, 1.49)
    ax[1].set_ylim(0.97, 1.92)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/shape-stress-corr-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]

    ############################ plot shape correlations ###########################
    if(whichPlot == "elongvsactivity"):
        figureName = sys.argv[3]
        numBins = int(sys.argv[4])
        plotElongationVSActivity(dirName, figureName, numBins)

    elif(whichPlot == "elongvsphi"):
        v0String = sys.argv[3]
        figureName = sys.argv[4]
        numBins = int(sys.argv[5])
        plotElongationVSPhi(dirName, v0String, figureName, numBins)

    elif(whichPlot == "shapevsactivity"):
        figureName = sys.argv[3]
        numBins = int(sys.argv[4])
        plotShapeMomentsVSActivity(dirName, figureName, numBins)

    elif(whichPlot == "shapevsphi"):
        v0String = sys.argv[3]
        figureName = sys.argv[4]
        numBins = int(sys.argv[5])
        plotShapeMomentsVSPhi(dirName, v0String, figureName, numBins)

    elif(whichPlot == "inertiavsactivity"):
        figureName = sys.argv[3]
        numBins = int(sys.argv[4])
        plotInertiaVSActivity(dirName, figureName, numBins)

    elif(whichPlot == "inertiavsphi"):
        v0String = sys.argv[3]
        figureName = sys.argv[4]
        numBins = int(sys.argv[5])
        plotInertiaVSPhi(dirName, v0String, figureName, numBins)

    elif(whichPlot == "shapecorr"):
        figureName = sys.argv[3]
        plotShapeCorrelation(dirName, figureName)

    elif(whichPlot == "shapestress"):
        figureName = sys.argv[3]
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        numBins = int(sys.argv[7])
        plotShapeStressCorrelation(dirName, figureName, numFrames, firstStep, stepFreq, numBins)

    else:
        print("Please specify the type of plot you want")
