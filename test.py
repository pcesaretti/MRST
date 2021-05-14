from lsh import *

a = open("trajectories/taxi500.txt", 'rb')
trajectories = cPickle.load(a)
a.close()
noTraj = len(trajectories)

plotTrajectories(trajectories)

#n equals the number of discs, Rs the radius for each discs
n = 60
Rs = np.repeat(1000,n)

circles = findCircles(trajectories, n)

#create the binayVectors and corresponding DID array
bv = binaryVectors(trajectories, circles, Rs)
hamArray = makeCdist(trajectories, circles, bv)

#create a edit distnace signatures for each trajectory, and compute distances
ev = editVector(trajectories, circles, Rs)
editArray = editArray(trajectories, ev)

#create set of pairs of trajectories whose DID assumes n and find the mean edit distances of those trajectories
pairs = findPairs(hamArray, n)
meanEditValues = findMean(pairs, editArray)


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


