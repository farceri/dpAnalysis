'''
Created by Francesco
12 October 2021
'''
#functions and script to visualize a 2d dpm packing
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
import itertools
import sys
import os
import spCorrelation as spCorr
import utilsCorr as ucorr
import utilsPlot as uplot

def plotSPCorr(ax, x, y, ylabel, color, legendLabel = None, logx = True, logy = False, linestyle = 'solid', alpha=1):
    ax.plot(x, y, linewidth=1., color=color, linestyle = linestyle, label=legendLabel, alpha=alpha)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_ylabel(ylabel, fontsize=18)
    if(logx == True):
        ax.set_xscale('log')
    if(logy == True):
        ax.set_yscale('log')


########################### plot and check compression #########################
def plotSPCompression(dirName, figureName, compute = "compute"):
    fig, ax = plt.subplots(2, 1, figsize = (6, 7), sharex = True, dpi = 120)
    if(compute=="compute"):
        phi = []
        pressure = []
        hop = []
        zeta = []
        for dir in os.listdir(dirName):
            if(os.path.isdir(dirName + os.sep + dir)):
                phi.append(ucorr.readFromParams(dirName + os.sep + dir, "phi"))
                pressure.append(ucorr.readFromParams(dirName + os.sep + dir, "pressure"))
                boxSize = np.loadtxt(dirName + os.sep + dir + "/boxSize.dat")
                psi6 = spCorr.computeHexaticOrder(dirName + os.sep + dir, boxSize)
                hop.append(np.mean(psi6))
                contacts = np.array(np.loadtxt(dirName + os.sep + dir + os.sep + "contacts.dat"))
                z = 0
                if(contacts.shape[0] != 0):
                    for p in range(contacts.shape[0]):
                        z += np.sum(contacts[p]>-1)
                    zeta.append(z/contacts.shape[0])
                else:
                    zeta.append(0)
        pressure = np.array(pressure)
        hop = np.array(hop)
        zeta = np.array(zeta)
        phi = np.array(phi)
        pressure = pressure[np.argsort(phi)]
        hop = hop[np.argsort(phi)]
        zeta = zeta[np.argsort(phi)]
        phi = np.sort(phi)
        np.savetxt(dirName + os.sep + "compression.dat", np.column_stack((phi, pressure, hop, zeta)))
    else:
        data = np.loadtxt(dirName + os.sep + "compression.dat")
        phi = data[:,0]
        pressure = data[:,1]
        hop = data[:,2]
        zeta = data[:,3]
    ax[0].semilogy(phi, pressure, color='k', linewidth=1.5)
    ax[1].plot(phi, zeta, color='k', linewidth=1.5)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax[0].set_ylabel("$pressure,$ $p$", fontsize=17)
    #ax[0].set_ylabel("$hexatic$ $order,$ $\\psi_6$", fontsize=17)
    ax[1].set_ylabel("$coordination$ $number,$ $z$", fontsize=17)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/soft/comp-control-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPPSI6P2Compression(dirName, figureName):
    fig, ax = plt.subplots(2, 1, figsize = (6, 7), sharex = True, dpi = 120)
    phi = []
    hop = []
    p2 = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir)):
            phi.append(ucorr.readFromParams(dirName + os.sep + dir, "phi"))
            boxSize = np.loadtxt(dirName + os.sep + dir + "/boxSize.dat")
            nv = np.loadtxt(dirName + os.sep + dir + "/numVertexInParticleList.dat", dtype=int)
            psi6 = spCorr.computeHexaticOrder(dirName + os.sep + dir, boxSize)
            hop.append(np.mean(psi6))
            eigvmax, _ = shapeDescriptors.getShapeDirections(dirName + os.sep + dir, boxSize, nv)
            angles = np.arctan2(eigvmax[:,1], eigvmax[:,0])
            p2.append(np.mean(2 * np.cos(angles - np.mean(angles))**2 - 1))
    phi = np.array(phi)
    hop = np.array(hop)
    p2 = np.array(p2)
    hop = hop[np.argsort(phi)]
    p2 = p2[np.argsort(phi)]
    phi = np.sort(phi)
    hop = hop[phi>0.65]
    p2 = p2[phi>0.65]
    phi = phi[phi>0.65]
    np.savetxt(dirName + os.sep + "compression.dat", np.column_stack((phi, pressure, zeta)))
    ax[0].plot(phi, p2, color='k', linewidth=1.5)
    ax[1].plot(phi, hop, color='k', linewidth=1.5)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax[0].set_ylabel("$nematic$ $order,$ $\\langle p2 \\rangle$", fontsize=17)
    ax[1].set_ylabel("$hexagonal$ $order,$ $\\langle \\psi_6 \\rangle$", fontsize=17)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/soft/comp-param-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPHOPCompression(dirName, figureName):
    dataSetList = np.array(os.listdir(dirName))
    phi = dataSetList.astype(float)
    dataSetList = dataSetList[np.argsort(phi)]
    phi = np.sort(phi)
    hop = np.zeros(phi.shape[0])
    err = np.zeros(phi.shape[0])
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    for i in range(dataSetList.shape[0]):
        psi6 = spCorr.computeHexaticOrder(dirName + os.sep + dataSetList[i])
        hop[i] = np.mean(psi6)
        err[i] = np.sqrt(np.var(psi6)/psi6.shape[0])
    ax.errorbar(phi[hop>0], hop[hop>0], err[hop>0], marker='o', color='k', markersize=5, markeredgecolor='k', markeredgewidth=0.7, linewidth=1, elinewidth=1, capsize=4)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax.set_ylabel("$hexatic$ $order$ $parameter,$ $\\psi_6$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/hop-comp-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotCompressionSet(dirName, figureName):
    #dataSetList = np.array(["kb1e-03", "kb1e-02", "kb1e-01", "kb2e-01", "kb4e-01", "kb5e-01-kakl", "kb6e-01", "kb8e-01"])
    dataSetList = np.array(["A1_1-sigma17", "A1_2-sigma17", "A1_3-sigma17"])
    phiJ = np.array([0.8301, 0.8526, 0.8242, 0.8205, 0.8176, 0.7785, 0.7722, 0.7707])
    colorList = ['k', [0.5,0,1], 'b', 'g', [0.8,0.9,0.2], [1,0.5,0], 'r', [1,0,0.5]]
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    for i in range(dataSetList.shape[0]):
        pressure = []
        phi = []
        for dir in os.listdir(dirName + dataSetList[i]):
            if(os.path.isdir(dirName + dataSetList[i] + os.sep + dir)):
                phi.append(ucorr.readFromParams(dirName + dataSetList[i] + os.sep + dir, "phi"))
                pressure.append(ucorr.readFromParams(dirName + dataSetList[i] + os.sep + dir, "pressure"))
        pressure = np.array(pressure)
        phi = np.array(phi)
        pressure = pressure[np.argsort(phi)]
        phi = np.sort(phi)
        phi = phi[pressure>0]
        pressure = pressure[pressure>0]
        np.savetxt(dirName + dataSetList[i] + os.sep + "compression.dat", np.column_stack((phi, pressure)))
        ax.semilogy(phi, pressure, color=colorList[i], linewidth=1.2, label=dataSetList[i])
    ax.legend(loc = "upper left", fontsize = 12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax.set_ylabel("$pressure,$ $p$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/compression-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPHOPDynamics(dirName, figureName):
    step = []
    hop = []
    err = []
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    for dir in os.listdir(dirName)[::10]:
        if(os.path.isdir(dirName + os.sep + dir)):
            step.append(float(dir[1:]))
            psi6 = spCorr.computeHexaticOrder(dirName + os.sep + dir)
            hop.append(np.mean(psi6))
            err.append(np.sqrt(np.var(psi6)/psi6.shape[0]))
    step = np.array(step)
    hop = np.array(hop)
    err = np.array(err)
    hop = hop[np.argsort(step)]
    err = err[np.argsort(step)]
    step = np.sort(step)
    plotErrorBar(ax, step, hop, err, "$simulation$ $step$", "$hexatic$ $order$ $parameter,$ $\\psi_6$")
    plt.savefig("/home/francesco/Pictures/soft/hexatic-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPPSI6P2Dynamics(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04):
    stepList = uplot.getStepList(numFrames, firstStep, stepFreq)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int)
    numParticles = nv.shape[0]
    hop = []
    p2 = []
    for i in stepList:
        psi6 = spCorr.computeHexaticOrder(dirName + os.sep + "t" + str(i), boxSize)
        hop.append(np.mean(psi6))
        eigvmax, _ = shapeDescriptors.getShapeDirections(dirName + os.sep + "t" + str(i), boxSize, nv)
        angles = np.arctan2(eigvmax[:,1], eigvmax[:,0])
        p2.append(np.mean(2 * np.cos(angles - np.mean(angles))**2 - 1))
    stepList -= stepList[0]
    stepList = np.array(stepList-stepList[0])/np.max(stepList-stepList[0])
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6,5), dpi=150)
    ax[0].plot(stepList, hop, linewidth=1.2, color='b')
    ax[1].plot(stepList, p2, linewidth=1.2, color='g')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$time$ $fraction,$ $t/t_{relax}$", fontsize=17)
    ax[0].set_ylabel("$\\langle \\psi_6 \\rangle$", fontsize=17)
    ax[1].set_ylabel("$\\langle P_2 \\rangle$", fontsize=17, labelpad=-5)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/soft/psi6-p2-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPHOPVSphi(dirName, figureName):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    phi = []
    hop = []
    err = []
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    for i in range(dataSetList.shape[0]):
        phi.append(ucorr.readFromParams(dirName + os.sep + dataSetList[i], "phi"))
        psi6 = spCorr.computeHexaticOrder(dirName + os.sep + dataSetList[i])
        hop.append(np.mean(psi6))
        err.append(np.sqrt(np.var(psi6)/psi6.shape[0]))
    plotErrorBar(ax, phi, hop, err, "$packing$ $fraction,$ $\\varphi$", "$hexatic$ $order$ $parameter,$ $\\psi_6$")
    plt.savefig("/home/francesco/Pictures/soft/hop-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotDeltaEvsDeltaV(dirName, figureName):
    #dataSetList = np.array(["1e-03", "3e-03", "5e-03", "7e-03", "9e-03", "1e-02", "1.3e-02", "1.5e-02", "1.7e-02", "2e-02", "2.3e-02", "2.5e-02", "2.7e-02", "3e-02", "4e-02", "5e-02", "6e-02"])
    dataSetList = np.array(["1e-03", "3e-03", "5e-03", "7e-03", "1e-02", "3e-02", "5e-02", "7e-02", "1e-01"])
    deltaE = []
    deltaV = []
    pressure = []
    fig = plt.figure(0, dpi=120)
    ax = fig.gca()
    energy0 = np.mean(np.loadtxt(dirName + os.sep + "dynamics-test/energy.dat")[:,2])
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + os.sep + "comp-delta" + dataSetList[i] + os.sep + "energy.dat")):
            energy = np.loadtxt(dirName + os.sep + "comp-delta" + dataSetList[i] + os.sep + "energy.dat")
            deltaE.append(np.mean(energy[:,2]) - energy0)
            deltaV.append(1 - (1-float(dataSetList[i]))**2)
            if(i < 5 and i > 0):
                pressure.append((deltaE[-1] - deltaE[0]) / (deltaV[-1] - deltaV[0]))
    ax.plot(deltaV, deltaE, lw=1.2, color='k', marker='.')
    print("average pressure: ", np.mean(pressure), "+-", np.std(pressure))
    x = np.linspace(0,0.1,100)
    m = np.mean(pressure)
    q = -10
    ax.plot(x, m*x + q, lw=1.2, color='g')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$\\Delta E$", fontsize=17)
    ax.set_xlabel("$\\Delta V$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pressure-" + figureName + ".png", transparent=False, format = "png")
    plt.show()


################################# plot dynamics ################################
def plotSPDynamics(dirName, figureName):
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    # plot brownian dynamics
    data = np.loadtxt(dirName + "/corr-log-q1.dat")
    timeStep = ucorr.readFromParams(dirName, "dt")
    ax.semilogx(data[1:,0]*timeStep, data[1:,2], color='b', linestyle='--', linewidth=1.2, marker="$T$", markersize = 10, markeredgewidth = 0.2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    ax.set_ylabel("$ISF(\\Delta t)$", fontsize=18)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pcorr-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDynamicsVSActivity(dirName, figureName):
    damping = 1e03
    meanRad = np.mean(np.loadtxt(dirName + "../particleRad.dat"))
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["10", "20", "40", "60", "80", "100"])
    colorList = cm.get_cmap('viridis', f0List.shape[0])
    markerList = ['v', 'o', 's']
    fig1, ax1 = plt.subplots(figsize = (7, 5), dpi = 120)
    fig2, ax2 = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(DrList.shape[0]):
        Diff = []
        tau = []
        deltaChi = []
        Pe = []
        for j in range(f0List.shape[0]):
            dirSample = dirName + "/Dr" + DrList[i] + "/Dr" + DrList[i] + "-f0" + f0List[j] + "/dynamics"
            if(os.path.exists(dirSample + os.sep + "corr-log-q1.dat")):
                data = np.loadtxt(dirSample + os.sep + "corr-log-q1.dat")
                timeStep = ucorr.readFromParams(dirSample, "dt")
                Diff.append(data[-1,1]/(4 * data[-1,0] * timeStep))
                tau.append(timeStep*ucorr.computeTau(data))
                deltaChi.append(ucorr.computeDeltaChi(data))
                Pe.append(((float(f0List[j])/damping) / float(DrList[i])) / meanRad)
                ax1.loglog(data[1:,0]*timeStep, data[1:,1], marker = markerList[i], color = colorList(j/f0List.shape[0]), fillstyle='none')
                #ax1.semilogx(data[1:,0]*timeStep, data[1:,2], marker = markerList[i], color = colorList(j/f0List.shape[0]), fillstyle='none')
        tau = np.array(tau)
        Diff = np.array(Diff)
        ax2.loglog(Pe, tau, linewidth=1.2, color='k', marker=markerList[i], fillstyle='none', markersize=10, markeredgewidth=1.5)
    #ax1.plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.5, color='k')
    #ax1.legend(("$D_r = 1$", "$D_r = 0.1$", "$D_r = 0.01$"), fontsize=14, loc="upper right")
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    ax1.set_ylabel("$MSD(\\Delta t)$", fontsize=18)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlabel("$Peclet$ $number,$ $v_0/(D_r \sigma)$", fontsize=18)
    ax2.set_ylabel("$Diffusivity,$ $D$", fontsize=17)
    #ax2.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=18)
    #ax2.set_ylabel("$Relaxation$ $interval,$ $\\Delta_\\chi$", fontsize=18)
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig("/home/francesco/Pictures/soft/pmsd-Drf0-" + figureName + ".png", transparent=True, format = "png")
    #fig2.savefig("/home/francesco/Pictures/soft/ptau-Drf0-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDynamicsVSTemp(dirName, figureName):
    T = []
    diff = []
    tau = []
    deltaChi = []
    dataSetList = np.array(["0.04", "0.05", "0.06", "0.07", "0.08", "0.09", #1e09
                            "0.1", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", #1e08
                            "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]) #1e07
    colorList = cm.get_cmap('plasma', dataSetList.shape[0])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/T" + dataSetList[i] + "/dynamics/corr-log-q1.dat")):
            data = np.loadtxt(dirName + "/T" + dataSetList[i] + "/dynamics/corr-log-q1.dat")
            timeStep = ucorr.readFromParams(dirName + "/T" + dataSetList[i] + "/dynamics/", "dt")
            #T.append(ucorr.readFromParams(dirName + "/T" + dataSetList[i] + "/dynamics/", "temperature"))
            energy = np.loadtxt(dirName + "/T" + dataSetList[i] + "/energy.dat")
            T.append(np.mean(energy[:,4]))
            diff.append(data[-1,1]/(4 * data[-1,0] * timeStep))
            tau.append(timeStep*ucorr.computeTau(data))
            deltaChi.append(timeStep*ucorr.computeDeltaChi(data))
            #print("T: ", T[-1], " diffusity: ", Deff[-1], " relation time: ", tau[-1], " tmax:", data[-1,0]*timeStep)
            #plotSPCorr(ax, data[:,0]*timeStep, data[:,1], "$MSD(\\Delta t)$", color = colorList(i/dataSetList.shape[0]), logy = True)
            #plotSPCorr(ax, data[:,0]*timeStep, data[:,1]/data[:,0]*timeStep, "$\\frac{MSD(\\Delta t)}{\\Delta t}$", color = colorList(i/dataSetList.shape[0]), logy = True)
            plotSPCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList(i/dataSetList.shape[0]))
            #plotSPCorr(ax, data[1:,0]*timeStep, data[1:,3], "$\\chi(\\Delta t)$", color = colorList(i/dataSetList.shape[0]))
    #ax.plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.5, color='k')
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    #ax.set_xlim(4e-04, 4e07)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pisf-T-" + figureName + ".png", transparent=True, format = "png")
    T = np.array(T)
    diff = np.array(diff)
    tau = np.array(tau)
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    #ax.loglog(T, diff, linewidth=1.5, color='k', marker='o')
    ax.loglog(1/T, tau, linewidth=1.5, color='k', marker='o')
    #ax.semilogx(T[2:], diff[2:]*tau[2:], linewidth=1.5, color='k', marker='o')
    #ax.semilogx(1/T, deltaChi, linewidth=1.5, color='k', marker='o')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Temperature,$ $T$", fontsize=17)
    #ax.set_ylabel("$Diffusivity,$ $D$", fontsize=17)
    ax.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    #ax.set_ylabel("$D$ $\\tau$", fontsize=17)
    #ax.set_ylabel("$Susceptibility$ $width,$ $\\Delta \\chi$", fontsize=17)
    plt.tight_layout()
    np.savetxt(dirName + "../diff-tau-vs-temp.dat", np.column_stack((T, diff, tau, deltaChi)))
    plt.savefig("/home/francesco/Pictures/soft/ptau-T-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDynamicsVSPhi(dirName, sampleName, figureName):
    phi = []
    tau = []
    Deff = []
    dirDyn = "/langevin/"
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7"])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0])
    fig, ax = plt.subplots(dpi = 150)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics/corr-log-q1.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + dirDyn  + "/T" + sampleName + "/dynamics/corr-log-q1.dat")
            timeStep = ucorr.readFromParams(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics", "dt")
            phi.append(ucorr.readFromParams(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics", "phi"))
            Deff.append(data[-1,1]/(4 * data[-1,0] * timeStep))
            tau.append(timeStep*ucorr.computeTau(data))
            print("phi: ", phi[-1], " Deff: ", Deff[-1], " tau: ", tau[-1])
            legendlabel = "$\\varphi=$" + str(np.format_float_positional(phi[-1],4))
            #plotSPCorr(ax, data[1:,0]*timeStep, data[1:,1], "$MSD(\\Delta t)$", color = colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), legendLabel = legendlabel, logy = True)
            plotSPCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), legendLabel = legendlabel)
    #ax.plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.5, color='k')
    #ax.set_ylim(3e-06,37100)#2.3e-04
    ax.legend(loc = "lower left", fontsize = 11, ncol = 2)
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    ax.set_xlim(3.8e-04, 8.13e05)
    #ax.set_ylim(7.5e-06, 8.8e03)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pisf-vsphi-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPTauVSActivity(dirName, figureName):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"])
    colorList = cm.get_cmap('plasma', dataSetList.shape[0])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/tauDiff.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + "/tauDiff.dat")
            ax.loglog(1/data[:,1], data[:,2], linewidth=1.5, color=colorList(i/dataSetList.shape[0]), marker='o')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Inverse$ $temperature,$ $1/T$", fontsize=17)
    #ax[1].set_ylabel("$Diffusivity,$ $D_{eff}$", fontsize=17)
    ax.set_ylabel("$log(\\tau)$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/ptau-active-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPTauVSTemp(dirName, figureName):
    phi0 = 0.8277#0.83867
    mu = 1.15#1.1
    delta = 1.05#1.2
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0]+10)
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/relaxationData.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + "/relaxationData.dat")
            phi = ucorr.readFromParams(dirName + dataSetList[i], "phi")
            #ax.loglog(1/data[:,0], np.log(data[:,2]), linewidth=1.5, color=colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), marker='o')
            ax.loglog(np.abs(phi - phi0)**(2/mu)/data[:,0], np.abs(phi0 - phi)**(delta) * np.log(np.sqrt(data[:,0])*data[:,2]), linewidth=1.5, color=colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), marker='o')
    ax.set_xlim(7.6e-04, 4.2e02)
    ax.set_ylim(6.3e-03, 1.8)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Inverse$ $temperature,$ $1/T$", fontsize=17)
    ax.set_xlabel("$|\\varphi - \\varphi_0|^{2/\\mu}/T$", fontsize=17)
    #ax.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    ax.set_ylabel("$|\\varphi - \\varphi_0|^\\delta \\log(\\tau T^{1/2})$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/ptau-vsT-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPTauVSPhi(dirName, sampleName, figureName):
    dataSetList = np.array(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0]+10)
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/active-langevin/T" + sampleName + "/tauDiff.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + "/active-langevin/T" + sampleName + "/tauDiff.dat")
            data = data[1:,:]
            ax.loglog(data[data[:,4]>0,1], data[data[:,4]>0,4], linewidth=1.5, color=colorList(i/dataSetList.shape[0]), marker='o')
    #ax.set_xlim(1.3, 15300)
    ax.plot(np.linspace(5,100,50), 2e04/np.linspace(5,100,50)**2, linestyle='--', linewidth=1.5, color='k')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Wave$ $vector$ $magnitude,$ $q$", fontsize=17)
    ax.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/ptau-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPDynamicsVSQ(dirName, figureName):
    dataSetList = np.array(["1", "2", "3", "5", "10", "20", "30", "50", "100"])
    colorList = cm.get_cmap('plasma', dataSetList.shape[0])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/corr-log-q" + dataSetList[i] + ".dat")):
            data = np.loadtxt(dirName + "/corr-log-q" + dataSetList[i] + ".dat")
            timeStep = ucorr.readFromParams(dirName, "dt")
            legendlabel = "$q=2\\pi/($" + dataSetList[i] + "$\\times d)$"
            plotSPCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList(i/dataSetList.shape[0]), legendLabel = legendlabel)
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    ax.legend(loc = "lower left", fontsize = 12)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pcorr-q-" + figureName + ".png", transparent=True, format = "png")
    plt.show()


############################## plot dynamics FDT ###############################
def plotSPEnergyScale(dirName, sampleName, figureName):
    Dr = []
    T = []
    pressure = []
    timeStep = 3e-04
    dataSetList = np.array(["1e03", "1e02", "1e01", "1", "1e-01", "1e-02", "1e-03", "1e-04"])
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    data = np.loadtxt(dirName + "../../T1/energy.dat")
    ax[0].semilogx(1/timeStep, np.mean(data[10:,4]), color='g', marker='$B$', markersize = 10, markeredgewidth = 0.2, alpha=0.2)
    ax[1].semilogx(1/timeStep, np.mean(data[10:,6]), color='g', marker='$B$', markersize = 10, markeredgewidth = 0.2, alpha=0.2)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/Dr" + dataSetList[i] + "-" + sampleName + "/dynamics/")):
            data = np.loadtxt(dirName + "/Dr" + dataSetList[i] + "-" + sampleName + "/dynamics/energy.dat")
            Dr.append(float(dataSetList[i]))
            T.append(np.mean(data[10:,4]))
            pressure.append(np.mean(data[10:,6]))
    ax[0].tick_params(axis='both', labelsize=15)
    ax[1].tick_params(axis='both', labelsize=15)
    ax[0].semilogx(Dr, T, linewidth=1.2, color='k', marker='o')
    ax[1].semilogx(Dr, pressure, linewidth=1.2, color='k', marker='o')
    ax[0].set_xlabel("$Persistence$ $time,$ $1/D_r$", fontsize=18)
    ax[1].set_xlabel("$Persistence$ $time,$ $1/D_r$", fontsize=18)
    ax[0].set_xlabel("$Rotational$ $diffusion,$ $D_r$", fontsize=18)
    ax[1].set_xlabel("$Rotational$ $diffusion,$ $D_r$", fontsize=18)
    ax[0].set_ylabel("$Temperature,$ $T$", fontsize=18)
    ax[1].set_ylabel("$Pressure,$, $p$", fontsize=18)
    ax[0].set_ylim(0.98,3.8)#1.15)#
    ax[1].set_ylim(5e-05,6.4e-03)#9.6e-04)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/soft-Tp-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPVelPDFVSMass(dirName, firstIndex, figureName):
    #dataSetList = np.array(["1e03", "5e03", "1e04", "5e04", "1e05", "5e05", "1e06"])
    #massList = np.array([1e03, 5e03, 1e04, 5e04, 1e05, 5e05, 1e06])
    dataSetList = np.array(["5e04", "1e05", "5e05", "1e06", "5e06"])
    massList = np.array([5e04, 1e05, 5e05, 1e06, 5e06])
    colorList = cm.get_cmap('plasma', massList.shape[0] + 1)
    fig = plt.figure(0, dpi=120)
    ax = fig.gca()
    for i in range(massList.shape[0]):
        scale = np.sqrt(massList[i])
        vel = []
        dirSample = dirName + os.sep + "dynamics-mass" + dataSetList[i]
        for dir in os.listdir(dirSample):
            if(os.path.isdir(dirSample + os.sep + dir)):
                vel.append(np.loadtxt(dirSample + os.sep + dir + os.sep + "particleVel.dat")[:firstIndex])
        vel = np.array(vel).flatten()
        mean = np.mean(vel) * scale
        Temp = np.var(vel) * scale**2
        alpha2 = np.mean((vel * scale - mean)**4)/(3 * Temp**2) - 1
        velPDF, edges = np.histogram(vel, bins=np.linspace(np.min(vel), np.max(vel), 60), density=True)
        edges = 0.5 * (edges[:-1] + edges[1:])
        print("Mass:", massList[i], " variance: ", Temp, " alpha2: ", alpha2)
        ax.semilogy(edges[velPDF>0] * scale, velPDF[velPDF>0] / scale, linewidth=1.5, color=colorList(i/massList.shape[0]), label="$m =$" + dataSetList[i])
    ax.legend(fontsize=10, loc="upper right")
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$P(v) / m^{1/2}$", fontsize=17)
    ax.set_xlabel("$v m^{1/2}$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/velSubSet-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPVelCorr(dirName, figureName, numBins):
    dirList = np.array(ucorr.getDirectories(dirName))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    meanRad = np.mean(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    bins = np.linspace(0, np.sqrt(boxSize[0]*boxSize[0] + boxSize[1]*boxSize[1]), numBins)
    binCenter = 0.5 * (bins[:-1] + bins[1:])
    speedCorr = np.zeros(bins.shape[0]-1)
    velCorr = np.zeros(bins.shape[0]-1)
    for dir in dirList[:1]:
        speedc, velc = spCorr.computeParticleVelCorr(dirName + os.sep + dir, meanRad, bins)
        speedCorr += speedc
        velCorr += velc
    speedCorr /= dirList.shape[0]
    velCorr /= dirList.shape[0]
    fig = plt.figure(0, dpi=120)
    ax = fig.gca()
    ax.plot(binCenter, velCorr, linewidth=1.5, color='k', marker='o')
    ax.plot(binCenter, speedCorr, linewidth=1.5, color='g', marker='*')
    ax.legend(("$\\langle \\sum_{ij} \\delta \\vec{v}_i \\cdot \\delta \\vec{v}_j \\rangle$", "$\\langle \\sum_{ij} \\delta v_i \\delta v_j \\rangle$"), loc = 'upper right', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Distance,$ $r/\\sigma$", fontsize=17)
    ax.set_ylabel("$Correlation,$ $C_v(r),$ $C_s(r)$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/velCorr-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPDensityVarVSTime(dirName, sampleName, numBins, figureName):
    dataSetList = np.array(["1", "1e-01", "1e-02"])
    if(sampleName == "10"):
        pressureList = np.array(["400", "400", "400"])
    else:
        pressureList = np.array(["485", "495", "560"])
    colorList = ['k', 'b', 'g']
    markerList = ['o', 'v', '*']
    fig, ax = plt.subplots(figsize = (8,4), dpi = 120)
    for i in range(dataSetList.shape[0]):
        var = []
        phi = []
        step = []
        dirSample = dirName + "Dr" + dataSetList[i] + "/Dr" + dataSetList[i] + "-f0" + sampleName + "/dynamics-ptot" + pressureList[i] + "/"
        #dirSample = dirName + "Dr" + dataSetList[i] + "/Dr" + dataSetList[i] + "-f0" + sampleName + "/dynamics-test/"
        for dir in os.listdir(dirSample):
            if(os.path.exists(dirSample + os.sep + dir + os.sep + "restAreas.dat")):
                if(float(dir[1:])%1e04 == 0):
                    localDensity = spCorr.computeLocalDensity(dirSample + os.sep + dir, numBins)
                    var.append(np.std(localDensity)/np.mean(localDensity))
                    phi.append(ucorr.readFromParams(dirSample + os.sep + dir, "phi"))
                    step.append(int(dir[1:]))
        var = np.array(var)
        phi = np.array(phi)
        step = np.array(step)
        var = var[np.argsort(step)]
        phi = phi[np.argsort(step)]
        step = np.sort(step)
        plt.plot(step, var, color=colorList[i], lw=1, marker=markerList[i], markersize=4)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel('$Simulation$ $step$', fontsize=18)
    ax.set_ylabel('$\\Delta \\varphi / \\varphi$', fontsize=18)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/localDensity-vsPhi-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDensityPDF(dirName, numBins, figureName):
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["10", "20", "40", "60", "80", "100"])
    colorList = cm.get_cmap('viridis', f0List.shape[0])
    markerList = ['v', 'o', 's']
    fig, ax = plt.subplots(dpi = 120)
    for i in range(DrList.shape[0]):
        for j in range(f0List.shape[0]):
            dirSample = dirName + "/Dr" + DrList[i] + "/Dr" + DrList[i] + "-f0" + f0List[j] + "/dynamics"
            if(os.path.exists(dirSample + os.sep + "localDensity-N" + numBins + ".dat")):
                data = np.loadtxt(dirSample + os.sep + "localDensity-N" + numBins + ".dat")
                data = data[data[:,1]>0]
                ax.plot(data[1:,0], data[1:,1], linewidth=1.2, marker=markerList[i], color=colorList((f0List.shape[0]-j)/f0List.shape[0]), fillstyle='none')
    data = np.loadtxt(dirName + "../langevin/T1e-01/dynamics/localDensity-N" + numBins + ".dat")
    data = data[data[:,1]>0]
    ax.plot(data[1:,0], data[1:,1], linewidth=1.2, marker='*', markersize=12, color='k', fillstyle='none', markeredgewidth=1.5)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_ylabel('$PDF(\\varphi)$', fontsize=18)
    ax.set_xlabel('$\\varphi$', fontsize=18)
    ax.set_yscale('log')
    #ax.set_xlim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/densityPDF-Drf0-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPLocalDensityPDFvsActivity(dirName, numBins, figureName):
    damping = 1e03
    meanRad = np.mean(np.loadtxt(dirName + "../particleRad.dat"))
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["10", "20", "40", "60", "80", "100"])
    markerList = ['v', 'o', 's']
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(DrList.shape[0]):
        T = []
        Tsubset = []
        deltaPhi = []
        for j in range(f0List.shape[0]):
            dirSample = dirName + "/Dr" + DrList[i] + "/Dr" + DrList[i] + "-f0" + f0List[j] + "/dynamics-mass1e06"
            if(os.path.exists(dirSample)):
                t, tsubset = spCorr.computeParticleVelPDFSubSet(dirSample, firstIndex=10, mass=1e06, plot=False)
                T.append(t)
                Tsubset.append(tsubset)
                deltaPhi.append(spCorr.computeLocalDensityPDF(dirSample, numBins))
        np.savetxt(dirName + "/Dr" + DrList[i] + "/localDensityData.dat", np.column_stack((T, Tsubset, deltaPhi)))
        Tsubset = np.array(Tsubset)
        ax.semilogx(Tsubset, deltaPhi, linewidth=1.2, color='k', marker=markerList[i], fillstyle='none', markersize=8, markeredgewidth=1.5)
    thermalData = np.loadtxt(dirName + "../../glassFDT/localDensityData.dat")
    ax.semilogx(thermalData[:,0], thermalData[:,1], linewidth=1.2, color='k', linestyle='--')
    ax.legend(("$D_r = 1$", "$D_r = 0.1$", "$D_r = 0.01$", "$thermal$"), fontsize=14, loc="upper left")
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Temperature$ $T,$ $T_{FDT}$", fontsize=18)
    ax.set_ylabel("$Variance$ $of$ $PDF(\\varphi)$", fontsize=18)
    fig.tight_layout()
    fig.savefig("/home/francesco/Pictures/soft/pPDFphi-Drf0-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPLocalDensityPDFvsTemp(dirName, numBins, figureName):
    T = []
    deltaPhi = []
    dataSetList = np.array(["0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1",
                            "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18",
                            "0.19", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
                            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    for i in range(dataSetList.shape[0]):
        dirSample = dirName + "/T" + dataSetList[i] + "/dynamics-mass1e06"
        if(os.path.exists(dirSample)):
            energy = np.loadtxt(dirSample + "/energy.dat")
            T.append(np.mean(energy[-20:,3]))
            deltaPhi.append(spCorr.computeLocalDensityPDF(dirSample, numBins, plot="plot"))
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    ax.loglog(T, deltaPhi, linewidth=1.5, color='k', marker='o')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Temperature,$ $T$", fontsize=17)
    ax.set_ylabel("$Variance$ $of$ $PDF(\\varphi)$", fontsize=17)
    plt.tight_layout()
    np.savetxt(dirName + "../localDensityData.dat", np.column_stack((T, deltaPhi)))
    plt.savefig("/home/francesco/Pictures/soft/pPDFphi-T-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPFDTSusceptibility(dirName, figureName, Dr, driving):
    tmeasure = 100
    fextStr = np.array(["2", "3", "4"])
    fext = fextStr.astype(float)
    mu = np.zeros((fextStr.shape[0],2))
    T = np.zeros((fextStr.shape[0],2))
    #fig0, ax0 = plt.subplots(dpi = 120)
    fig, ax = plt.subplots(1, 2, figsize = (12.5, 5), dpi = 120)
    corr = np.loadtxt(dirName + os.sep + "dynamics/corr-log-q1.dat")
    timeStep = ucorr.readFromParams(dirName + os.sep + "dynamics/", "dt")
    #plotSPCorr(ax0, corr[1:,0]*timeStep, corr[1:,1]/(corr[1:,0]*timeStep), "$MSD(\\Delta t) / \\Delta t$", color = 'k', logy = True)
    timeStep = ucorr.readFromParams(dirName + os.sep + "dynamics", "dt")
    diff = np.mean(corr[corr[:,0]*timeStep>tmeasure,1]/(2*corr[corr[:,0]*timeStep>tmeasure,0]*timeStep))
    for i in range(fextStr.shape[0]):
        sus = np.loadtxt(dirName + os.sep + "dynamics-fext" + fextStr[i] + "/susceptibility.dat")
        sus = sus[sus[:,0]>tmeasure,:]
        mu[i,0] = np.mean(sus[:,1]/sus[:,0])
        mu[i,1] = np.std(sus[:,1]/sus[:,0])
        energy = np.loadtxt(dirName + os.sep + "dynamics-fext" + fextStr[i] + "/energy.dat")
        energy = energy[energy[:,0]>tmeasure,:]
        T[i,0] = np.mean(energy[:,4])
        T[i,1] = np.std(energy[:,4])
    ax[0].errorbar(fext, mu[:,0], mu[:,1], color='k', marker='o', markersize=8, lw=1, ls='--', capsize=3)
    ax[1].errorbar(fext, T[:,0], T[:,1], color='b', marker='D', fillstyle='none', markeredgecolor = 'b', markeredgewidth = 1.5, markersize=8, lw=1, ls='--', capsize=3)
    ax[1].errorbar(fext, diff/mu[:,0], mu[:,1], color='k', marker='o', markersize=8, lw=1, ls='--', capsize=3)
    for i in range(ax.shape[0]):
        ax[i].tick_params(axis='both', labelsize=15)
        ax[i].set_xlabel("$f_0$", fontsize=18)
    ax[0].set_ylabel("$Mobility,$ $\\chi / t = \\mu$", fontsize=18)
    ax[1].set_ylabel("$Temperature$", fontsize=18)
    ax[1].legend(("$Kinetic,$ $T$", "$FDT,$ $D/ \\mu = T_{FDT}$"), loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    #fig.savefig("/home/francesco/Pictures/soft/pFDT-" + figureName + ".png", transparent=True, format = "png")
    pSize = 2 * np.mean(np.array(np.loadtxt(dirName + os.sep + "dynamics/particleRad.dat")))
    Pe = pSize * driving / 1e-02
    Pev = ((driving / 1e03) / Dr) / pSize
    print("Pe: ", Pev, " susceptibility: ",  np.mean(mu[i,0]), " diffusivity: ", diff, " T_FDT: ", diff/np.mean(mu[i,0]))
    np.savetxt(dirName + "FDTtemp.dat", np.column_stack((Pe, Pev, np.mean(T[:,0]), np.std(T[:,0]), np.mean(mu[:,0]), np.std(mu[:,0]), diff)))
    plt.show()

def plotSPFDTdata(dirName, firstIndex, mass, figureName):
    damping = 1e03
    meanRad = np.mean(np.loadtxt(dirName + "../particleRad.dat"))
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["10", "20", "40", "60", "80", "100"])
    markerList = ['v', 'o', 's']
    fig1, ax1 = plt.subplots(figsize = (7, 5), dpi = 120)
    fig2, ax2 = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(DrList.shape[0]):
        Dr = []
        f0 = []
        Pe = []
        T = []
        Tsubset = []
        diff = []
        tau = []
        deltaChi = []
        Treduced = []
        for j in range(f0List.shape[0]):
            dirSample = dirName + "/Dr" + DrList[i] + "/Dr" + DrList[i] + "-f0" + f0List[j] + "/dynamics-mass1e06"
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + os.sep + "../dynamics/corr-log-q1.dat")
                timeStep = ucorr.readFromParams(dirSample + os.sep + "../dynamics", "dt")
                diff.append(np.mean(data[-10:,1]/(4 * data[-10:,0] * timeStep)))
                tau.append(timeStep*ucorr.computeTau(data))
                deltaChi.append(ucorr.computeDeltaChi(data))
                Dr.append(float(DrList[i]))
                f0.append(float(f0List[j]))
                Pe.append(((float(f0List[j])/damping) / float(DrList[i])) / (2 * meanRad))
                t, tsubset = spCorr.computeParticleVelPDFSubSet(dirSample, firstIndex, mass, plot=False)
                T.append(t)
                Tsubset.append(tsubset)
                Treduced.append(Tsubset[-1]*f0[-1]/(Dr[-1] * damping * 2 * meanRad))
        np.savetxt(dirName + "/Dr" + DrList[i] + "/FDTdata.dat", np.column_stack((Dr, f0, Pe, T, Tsubset, tau, diff, deltaChi)))
        Pe = np.array(Pe)
        Tsubset = np.array(Tsubset)
        tau = np.array(tau)
        diff = np.array(diff)
        Dr = np.array(Dr)
        f0 = np.array(f0)
        Treduced = np.array(Treduced)
        ax1.loglog(Pe, Treduced, linewidth=1.2, color='k', marker=markerList[i], fillstyle='none', markersize=10, markeredgewidth=1.5)
        ax2.loglog(Treduced, tau*diff, linewidth=1.2, color='k', marker=markerList[i], fillstyle='none', markersize=8, markeredgewidth=1.5)
        print("energy scale: ", Tsubset/Treduced)
    thermalData = np.loadtxt(dirName + "../../glassFDT/relaxationData-test.dat")
    ax2.semilogx(thermalData[:,0], thermalData[:,1]*thermalData[:,2], linewidth=1.2, color='k', linestyle='--')
    ax2.legend(("$D_r = 1$", "$D_r = 0.1$", "$D_r = 0.01$", "$thermal$"), fontsize=14, loc="upper left")
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlabel("$Peclet$ $number,$ $v_0/(D_r \\sigma)$", fontsize=18)
    ax1.set_ylabel("$T_{FDT}/\\epsilon_A$", fontsize=18)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlabel("$T_{FDT}/\\epsilon_A, T/\\epsilon$", fontsize=18)
    #ax2.set_ylabel("$Diffusivity,$ $D$", fontsize=18)
    #ax2.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=18)
    ax2.set_ylabel("$D$ $\\tau$", fontsize=18)
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig("/home/francesco/Pictures/soft/pPeTfdt-Drf0-" + figureName + ".png", transparent=True, format = "png")
    fig2.savefig("/home/francesco/Pictures/soft/pdifftauTfdt-Drf0-" + figureName + ".png", transparent=True, format = "png")
    plt.show()


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]

########################### check and plot compression #########################
    if(whichPlot == "comp"):
        figureName = sys.argv[3]
        plotSPCompression(dirName, figureName)

    elif(whichPlot == "hexcomp"):
        figureName = sys.argv[3]
        plotSPHOPCompression(dirName, figureName)

    elif(whichPlot == "comppsi6p2"):
        figureName = sys.argv[3]
        plotSPPSI6P2Compression(dirName, figureName)

    elif(whichPlot == "compset"):
        figureName = sys.argv[3]
        plotCompressionSet(dirName, figureName)

    elif(whichPlot == "hop"):
        figureName = sys.argv[3]
        plotSPHOPDynamics(dirName, figureName)

    elif(whichPlot == "psi6p2"):
        figureName = sys.argv[3]
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        plotSPPSI6P2Dynamics(dirName, figureName, numFrames, firstStep, stepFreq)

    elif(whichPlot == "hopphi"):
        figureName = sys.argv[3]
        plotSPHOPVSphi(dirName, figureName)

    elif(whichPlot == "pressure"):
        figureName = sys.argv[3]
        plotDeltaEvsDeltaV(dirName, figureName)

################################# plot dynamics ################################
    elif(whichPlot == "pdyn"):
        figureName = sys.argv[3]
        plotSPDynamics(dirName, figureName)

    elif(whichPlot == "pdynactivity"):
        figureName = sys.argv[3]
        plotSPDynamicsVSActivity(dirName, figureName)

    elif(whichPlot == "pdyntemp"):
        figureName = sys.argv[3]
        plotSPDynamicsVSTemp(dirName, figureName)

    elif(whichPlot == "pdynphi"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        plotSPDynamicsVSPhi(dirName, sampleName, figureName)

    elif(whichPlot == "ptauactivity"):
        figureName = sys.argv[3]
        plotSPTauVSActivity(dirName, figureName)

    elif(whichPlot == "ptautemp"):
        figureName = sys.argv[3]
        plotSPTauVSTemp(dirName, figureName)

    elif(whichPlot == "ptauphi"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        plotSPTauVSPhi(dirName, sampleName, figureName)

    elif(whichPlot == "pdynq"):
        figureName = sys.argv[3]
        plotParticleDynamicsVSQ(dirName, figureName)

############################## plot dynamics FDT ###############################
    elif(whichPlot == "pscale"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        plotSPEnergyScale(dirName, sampleName, figureName)

    elif(whichPlot == "pvelmass"):
        firstIndex = int(sys.argv[3])
        figureName = sys.argv[4]
        plotSPVelPDFVSMass(dirName, firstIndex, figureName)

    elif(whichPlot == "pvelcorr"):
        figureName = sys.argv[3]
        numBins = int(sys.argv[4])
        plotSPVelCorr(dirName, figureName, numBins)

    elif(whichPlot == "pdensityvstime"):
        sampleName = sys.argv[3]
        numBins = int(sys.argv[4])
        figureName = sys.argv[5]
        plotSPDensityVarVSTime(dirName, sampleName, numBins, figureName)

    elif(whichPlot == "pdensitypdf"):
        numBins = sys.argv[3]
        figureName = sys.argv[4]
        plotSPDensityPDF(dirName, numBins, figureName)

    elif(whichPlot == "pdensityvsactivity"):
        numBins = int(sys.argv[3])
        figureName = sys.argv[4]
        plotSPLocalDensityPDFvsActivity(dirName, numBins, figureName)

    elif(whichPlot == "pdensityvstemp"):
        numBins = int(sys.argv[3])
        figureName = sys.argv[4]
        plotSPLocalDensityPDFvsTemp(dirName, numBins, figureName)

    elif(whichPlot == "pfdtsus"):
        figureName = sys.argv[3]
        Dr = float(sys.argv[4])
        driving = float(sys.argv[5])
        plotSPFDTSusceptibility(dirName, figureName, Dr, driving)

    elif(whichPlot == "pfdtdata"):
        firstIndex = int(sys.argv[3])
        mass = float(sys.argv[4])
        figureName = sys.argv[5]
        plotSPFDTdata(dirName, firstIndex, mass, figureName)

    else:
        print("Please specify the type of plot you want")
