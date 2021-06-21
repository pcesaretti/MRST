from lsh import *
from DSBF import *
import time

a = open("trajectories/taxi1000.txt", 'rb')
trajectories = cPickle.load(a)
a.close()

addRandomTimeDim(trajectories)

noTraj = len(trajectories)

#n equals the number of discs, Rs the radius for each discs
t1 = time.time()

n = 70
Rs = np.repeat(1000,n)

spheres = findSpheres(trajectories, n)
plotTrajectoriesAndSpheres(trajectories, spheres, Rs)

#create the binayVectors and corresponding DID array
max_length = max(map(len, trajectories))
trajectories = np.array([xi+[[-1,-1,-1]]*(max_length-len(xi)) for xi in trajectories], dtype=float)
bv = binaryVectorsSpheres(trajectories, spheres, Rs)

bv = bv.astype(int)
hamArray = makeCdist(trajectories, spheres, bv)

#create a edit distnace signatures for each trajectory, and compute distances
ev = editVector(trajectories, spheres, Rs)
editArray = editArray(trajectories, ev)

#create set of pairs of trajectories whose DID assumes n and find the mean edit distances of those trajectories
pairs = findPairs(hamArray, n)
meanEditValues = findMean(pairs, editArray)

t2 = time.time()
print("time taken for MRST algo = " + str(t2-t1) + " sec")

#print(meanEditValues)

#ploting the mean edit distances for each possible value of DID
plotCorr(meanEditValues)
ax = plt.gca()
ax.set_xlim([0, n])
ax.set_ylim([0, n])
plt.xlabel("Disc intersection Distance")
plt.ylabel("Mean Edit Distance")
plt.show()

#plotting the values of edit distnaces vs DID 

plt.scatter(hamArray, editArray, c='r')
ax = plt.gca()
ax.set_xlim([0, n])
ax.set_ylim([0, n])
plt.xlabel("Disc Intersection Distance")
plt.ylabel("Edit Distance")
plt.show()