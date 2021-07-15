from numpy import concatenate
from numpy.lib.function_base import append
import haus
from lsh import *
import time
from joblib import Parallel, delayed
from os.path import exists
from os.path import splitext
import matplotlib.cm as cm


SAVE_SPHERES = False
LOAD_SPHERES = False

datasetFilename = "taxi1000_3d.npy"

RsStart = 3000 
RsStop = 6000
RsStepSize = 100
nStart = 50
nStop = 100
nStepSize = 10

numTrials = 1

def outer_h(traj,trajectories):
    return [haus.distanceBetweenSurfaces(traj,tr) for tr in trajectories]

datasetFilenameShort = splitext(datasetFilename)[0]
hausArrayFilename = "hausArray_" + datasetFilenameShort + ".npy"
baseFolderPath = datasetFilenameShort + "_results/"

mkdir_p(baseFolderPath)

trajectories = np.load("trajectories/" + datasetFilename, allow_pickle=True)
queries = trajectories[:20][:]
trajectories = trajectories[20:][:]

t1 = time.time()

# make trajectories into proper numpy 3d arrays to be compatible with Numba
trajectoriesNumba = fillTrajNumba(trajectories)

if(exists(hausArrayFilename) == False):
    hausArray = Parallel(n_jobs = 10)(delayed(outer_h)(trajectoryA, trajectoriesNumba) for trajectoryA in trajectoriesNumba)
    np.save(hausArrayFilename, hausArray)
else:
    hausArray = np.load(hausArrayFilename)

t2 = time.time()
print("time taken for haus algo = " + str(t2-t1) + " sec")

# n equals the number of discs, Rs the radius for each discs
num_n = int((nStop-nStart)/nStepSize + 1)
num_Rs = int((RsStop-RsStart)/RsStepSize + 1)
nArray = np.linspace(nStart, nStop, num=num_n, dtype = int)
RsArray = np.linspace(RsStart, RsStop, num=num_Rs, dtype = int)
all_correlations = np.zeros((num_n,num_Rs), dtype=float)
for i in range(num_n):
    correlationArray = np.zeros(num_Rs, dtype=float)
    for RsIndex in range(num_Rs):
        RsRepeat = np.repeat(RsArray[RsIndex], nArray[i])
        correlcoTotal = 0
        for trial in range(numTrials):
            sphereFilepath = "spheres/spheres_n_" + str(nArray[i]) + "_Rs_" + str(RsRepeat[0]) + "_trial_" + str(trial) + ".npy"
            if(LOAD_SPHERES == False):
                spheres = findSpheres(trajectories, nArray[i])
                if(SAVE_SPHERES == True):
                    mkdir_p(baseFolderPath + "spheres")
                    np.save(baseFolderPath + sphereFilepath, spheres)
            else:
                spheres = np.load(baseFolderPath + sphereFilepath)

            #plotTrajectoriesAndSpheres(trajectories, spheres, RsRepeat)

            bv = binaryVectorsSpheres(trajectoriesNumba, spheres, RsRepeat)
            bv = bv.astype(int)

            # create a edit distance signatures for each trajectory, and compute distances
            ev = editVector3d(trajectoriesNumba, spheres, RsRepeat)

            editedArray = editArray(trajectoriesNumba, ev)

            editPairs = findPairs(editedArray, nArray[i])
            meanHaus, max_DID = findMean(editPairs, hausArray)
            correlation = correlco(meanHaus)[0][1]

            t3 = time.time()
            print("time taken so far = " + str(t3-t1) + " sec")

            plt.clf()
            plotCorr(meanHaus, max_DID)
            ax = plt.gca()
            ax.set_xlim([0,nArray[i]])
            ax.set_ylim([0, 6000])
            ax.legend(['correlation = ' + str(round(correlation, 4))], loc ="lower right")
            plt.title(['n = ' + str(nArray[i]) + ', Rs = ' + str(RsRepeat[0])])
            plt.xlabel("Binary Sketch Distance (levenshtein)")
            plt.ylabel("Mean Hausdorf Distance")

            mkdir_p(baseFolderPath + "plots")
            imgnameFilepath = "plots/n_" + str(nArray[i]) + "_Rs_" + str(RsRepeat[0]) + "_trial_"+ str(trial) + ".png"
            plt.savefig(baseFolderPath + imgnameFilepath)
            #plt.show()

            print(str(correlation) + " for " + imgnameFilepath)
            correlcoTotal = correlcoTotal + correlation
        correlationArray[RsIndex] = correlcoTotal/numTrials
    all_correlations[i] = correlationArray
        

    mkdir_p(baseFolderPath + "correlationArrays")
    correlationFilepath = "correlationArrays/correlationArray_n_" + str(nArray[i]) + ".txt"
    np.save(baseFolderPath + correlationFilepath, correlationArray)

    plt.clf()
    ax = plt.gca()
    plt.xlabel("Sphere radius (m)")
    plt.ylabel("Pearson correlation coefficient")
    plt.plot(RsArray, correlationArray, '-o')
    ax.legend([str(nArray[i]) + " spheres"])

    imgnameFilepath = "plots/correlco_n_" + str(nArray[i]) + ".png"
    plt.savefig(baseFolderPath + imgnameFilepath)

plt.clf()
ax = plt.gca()
plt.xlabel("Sphere radius (m)")
plt.ylabel("Pearson correlation coefficient")
colors = cm.rainbow(np.linspace(0, 1, num_n))
for i in range(num_n):
    plt.plot(RsArray, all_correlations[i], '-o', linewidth=1.5, markersize=3, color=colors[i], label=str(nArray[i]) + " spheres")
ax.set_xlim([RsStart-RsStepSize, RsStop+RsStepSize])
ax.set_ylim([0, 1])
plt.legend()
plt.savefig(baseFolderPath + "plots/correlco_all.png")