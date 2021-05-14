from lsh import *
from frechet import *

a = open("trajectories/taxi200.txt", 'rb')
trajectories = cPickle.load(a)
a.close()
noTraj = len(trajectories)

n = 60
Rs = np.repeat(1000, n)

circles = findCircles(trajectories, n)

ev = editVector(trajectories, circles, Rs)
editArray = editArray(trajectories, ev)

frechetArray = distanceMatrixOfCurves(trajectories)

editPairs = findPairs(editArray, n)
meanFrechet = findMean(editPairs, frechetArray)
correlation = correlco(meanFrechet)[0][1]

plotCorr(meanFrechet)
ax = plt.gca()
ax.set_xlim([0, n])
ax.set_ylim([0, 6000])
ax.legend([correlation])
plt.xlabel("Edit Distance")
plt.ylabel("Mean Frechet Distance")
plt.show()


