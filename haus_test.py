from lsh import *
from haus import *

a = open("trajectories/taxi500.txt", 'rb')
trajectories = cPickle.load(a)
a.close()
noTraj = len(trajectories)

n = 60
Rs = np.repeat(1000, n)

circles = findCircles(trajectories, n)

ev = editVector(trajectories, circles, Rs)
editArray = editArray(trajectories, ev)

hausArray = distanceMatrixOfCurves(trajectories)
editPairs = findPairs(editArray, n)
meanHaus = findMean(editPairs, hausArray)
correlation = correlco(meanHaus)[0][1]

#plot correlation graph
plotCorr(meanHaus)
ax = plt.gca()
ax.set_xlim([0,n])
ax.set_ylim([0, 6000])
ax.legend([correlation])
plt.xlabel("Edit Distance")
plt.ylabel("Mean Hausdorf Distance")
plt.show()

print(correlco(meanHaus))
