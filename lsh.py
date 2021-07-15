import random
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import sqrt

from numpy.lib.function_base import percentile

colors = [ 'r', 'b','c', 'g', 'y', 'k', 'm', 'r']
colorsNew = ['c', 'g', 'y', 'm']

def plotQueryAndClosest(query, trajectories, possibleClosestTrajIndices, closestTrajIdx, trueClosestTrajIdx):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter([a for [a,b,c] in query],[b for [a,b,c] in query], [c for [a,b,c] in query], color = 'black', s=20, label='query')
    ax.scatter([a for [a,b,c] in trajectories[trueClosestTrajIdx]],[b for [a,b,c] in trajectories[trueClosestTrajIdx]], [c for [a,b,c] in trajectories[trueClosestTrajIdx]], color = 'blue', s=20, label='true closest traj')

    c = 0
    for index in possibleClosestTrajIndices:
        if index == -1:
            break
        if(index == closestTrajIdx):
            colorval = 'red'
            sval = 20
            ax.scatter([a for [a,b,c] in trajectories[index]],[b for [a,b,c] in trajectories[index]], [c for [a,b,c] in trajectories[index]], color = colorval, s=sval, label='found closest traj')
        else:
            sval = 3
            colorval = colorsNew[c%4]
            c = c+1
            ax.scatter([a for [a,b,c] in trajectories[index]],[b for [a,b,c] in trajectories[index]], [c for [a,b,c] in trajectories[index]], color = colorval, s=sval)
    
    plt.axis('auto')
    ax.legend()
    plt.show()

def scale_time(trajectories, S):  
    for i in range(len(trajectories)):
        for j in range(len(trajectories[i])):
            trajectories[i][j][2] = trajectories[i][j][2] * S
    
    return trajectories

@njit
def scale_time_numba(trajectories, S):  
    for i in range(len(trajectories)):
        for j in range(len(trajectories[i])):
            if(trajectories[i][j][0] == -1):
                break
            trajectories[i][j][2] = trajectories[i][j][2] * S
    
    return trajectories

@njit
def find_avg_speed(trajectories):

    avgSpeedAllTraj = 0
    for i in range(len(trajectories)):
        distanceCoveredByTraj = 0
        startTime = trajectories[i][0][2]
        endTime = -1
        speedTraj = 0
        for j in range(len(trajectories[i])-1):
            if(trajectories[i][j+1][0] == -1):
                endTime = trajectories[i][j][2]
                break
            distanceCoveredByTraj =  distanceCoveredByTraj + sqrt( (trajectories[i][j+1][0]-trajectories[i][j][0])**2 +
                                                                       (trajectories[i][j+1][1]-trajectories[i][j][1])**2 )
        if endTime == -1:
            endTime = trajectories[i][j][2]
        if endTime != startTime:
            speedTraj = distanceCoveredByTraj/(endTime-startTime)
        avgSpeedAllTraj = avgSpeedAllTraj + speedTraj

    avgSpeedAllTraj = avgSpeedAllTraj/len(trajectories)
    return avgSpeedAllTraj


def hamming_distance(a, b):
    ret = 0
    if len(a) > len(b):
        b = np.pad(b, (len(a) - len(b), 0), 'constant')
    else:
        a = np.pad(a, (len(b) - len(a), 0), 'constant')
    
    for i in range(len(a)):
        if a[i] != b[i]:
            ret = ret + 1

    return ret

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    from errno import EEXIST
    from os import makedirs,path
    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def fillTrajNumba(trajectories):
    max_length = max(map(len, trajectories))
    trajectoriesNumba = np.empty((len(trajectories), max_length, len(trajectories[0][0])), dtype='float')
    trajectoriesNumba.fill(-1)

    for i in range(len(trajectories)):
        for j in range(len(trajectories[i])):
            trajectoriesNumba[i][j] = trajectories[i][j]

    return trajectoriesNumba

# meant to be temporary until real time data is gotten
def addRandomTimeDim(trajectories):

    for trajectory in trajectories:
        for i in range(len(trajectory)):
            if (i == 0):
                trajectory[i].append( random.uniform(0,1000) )
                continue
            randomTime = trajectory[i-1][2] + sqrt( (trajectory[i][0]-trajectory[i-1][0])**2 +
                                                    (trajectory[i][1]-trajectory[i-1][1])**2 ) * (random.uniform(0.8, 1.2))
            trajectory[i]. append(randomTime)

def findCircles(trajectories, n):
    
    cs = np.empty((n,2), dtype='float')

    xmax, ymax =  map(max, zip(*[x for l in trajectories for x in l])) 
    xmin, ymin = map(min, zip(*[x for l in trajectories for x in l]))
    
    A = (xmax-xmin)*(ymax-ymin)
    
    for i in range(n):

        xcircle = random.uniform(xmin,xmax)
        ycircle = random.uniform(ymin,ymax)

        cs[i][0] = xcircle
        cs[i][1] = ycircle

    return cs

def findSpheres(trajectories, n):
    
    cs = np.empty((n,3), dtype='float')

    xmax, ymax, tmax =  map(max, zip(*[x for l in trajectories for x in l])) 
    xmin, ymin, tmin = map(min, zip(*[x for l in trajectories for x in l]))


    for i in range(n):

        xcircle = random.uniform(xmin,xmax)
        ycircle = random.uniform(ymin,ymax)
        tcircle = random.uniform(tmin,tmax)

        cs[i][0] = xcircle
        cs[i][1] = ycircle
        cs[i][2] = tcircle

    return cs

def plotTrajectories(trajectories):
    
    fig, ax = plt.subplots(figsize=(15,7))
    
    for c, trajectory in enumerate(trajectories):
        ax.scatter([a for [a,b] in trajectory],[b for [a,b] in trajectory], color = colors[c%8], s = 5)
    plt.axis('equal')

def plotCircles(cs, Rs):

    for i,[xcircle,ycircle] in enumerate(cs):
        circle = plt.Circle((xcircle,ycircle), Rs[i], color='r', alpha=0.2)
        plt.gcf().gca().add_artist(circle)
    
    plt.show()

def plotTrajectoriesAndSpheres(trajectories, cs, Rs):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    for c, trajectory in enumerate(trajectories):
        ax.scatter([a for [a,b,c] in trajectory],[b for [a,b,c] in trajectory], [c for [a,b,c] in trajectory], color = colors[c%8], s = 5)

    for i,[xcircle,ycircle, tcircle] in enumerate(cs[:20]):
        
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = xcircle + Rs[i] * np.cos(u)*np.sin(v)
        y = ycircle + Rs[i] * np.sin(u)*np.sin(v)
        t = tcircle + Rs[i] * np.cos(v)
        ax.plot_wireframe(x, y, t, color="r", rstride=1, cstride=1)
    
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
    
    plt.show()

@njit        
def binaryVectors(trajectories, cs, Rs):

    n = len(cs)
    vs = np.zeros((len(trajectories),n), dtype='bool')
    
    for i in range(len(trajectories)):
        for j in range(n):
            xcircle,ycircle = cs[j]
            for x,y in trajectories[i]:
                if(x == 0 and y == 0):
                    break
                elif (x - xcircle)**2 + (y - ycircle)**2 < Rs[j]**2:
                    vs[i][j] = True

    return vs

@njit        
def binaryVectorsSpheres(trajectories, cs, Rs):

    n = len(cs)
    vs = np.zeros((len(trajectories),n), dtype='bool')
    
    for i in range(len(trajectories)):
        for j in range(n):
            xcircle,ycircle,tcircle = cs[j]
            for x,y,t in trajectories[i]:
                if(x == -1 and y == -1 and t == -1):
                    break
                elif (x - xcircle)**2 + (y - ycircle)**2 + (t - tcircle)**2 < Rs[j]**2:
                    vs[i][j] = True

    return vs

@njit
def makeCdist(trajectories, cs, vs):

    cdist = np.zeros((len(trajectories),len(trajectories)))
    
    for i in range(0,len(trajectories)):
        for j in range(i,len(trajectories)):
            cdist[i,j] = cdist[j,i] = len(np.bitwise_xor(vs[i],vs[j]).nonzero()[0])
    
    return cdist


# create a signature that will note the order of circles in which a trajectory is located;
# if a coordinate of the trajectory is located within a circle, c[i] with radius Rs[i], then
# we append the index of that circle to an array called edit_string.
@njit
def editVector(trajectories, cs, Rs):
    
    n = len(cs)
    ed = np.empty((len(trajectories),n+1), dtype='int')
    ed.fill(-1)

    for j in range(len(trajectories)):
        v = np.zeros(n)
        es_index = 1
        for i in range(n):
            xcircle, ycircle = cs[i]
            for x, y in trajectories[j]:
                if(x == 0 and y == 0):
                    break
                elif ((x - xcircle)**2 + (y - ycircle)**2 < Rs[i]**2
                    and v[i] != 1):
                        v[i] = 1
                        ed[j][es_index] = i
                        es_index = es_index + 1
        ed[j][0] = es_index

    return ed


@njit
def editVector3d(trajectories, cs, Rs):
    
    n = len(cs)
    ed = np.empty((len(trajectories),n+1), dtype='int')
    ed.fill(-1)

    for j in range(len(trajectories)):
        v = np.zeros(n)
        es_index = 1
        for i in range(n):
            xcircle, ycircle, tcircle = cs[i]
            for x, y, t in trajectories[j]:
                if(x == -1 and y == -1 and t == -1):
                    break
                elif ((x - xcircle)**2 + (y - ycircle)**2 + (t - tcircle)**2 < Rs[i]**2
                    and v[i] != 1):
                        v[i] = 1
                        ed[j][es_index] = i
                        es_index = es_index + 1
        ed[j][0] = es_index

    return ed

# create an array, listing the edit distances between trajectories. Location A[i,j] will
# list the distance between trajectory i and j
@njit
def editArray(ts, edit_vector):

    array = np.zeros((len(ts), len(ts)))

    for i in range(0, len(ts)):
        for j in range(i+1, len(ts)):
            ev_i_last_index = edit_vector[i][0]
            ev_j_last_index = edit_vector[j][0]
            array[i, j] = array[j, i] = levenshtein(edit_vector[i][1:ev_i_last_index], edit_vector[j][1:ev_j_last_index])

    return array


# edit distance test for error; this algorithm uses dynamic programming to keep track of previously computed distances
@njit
def levenshtein(s, t):
    if len(s) == 0: return len(t)
    elif len(t) == 0: return len(s)
    v0 = np.empty(len(t) + 1, dtype='int')
    v1 = np.empty(len(t) + 1, dtype='int')
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]
            
    return v1[len(t)]

# find the pairs of trajectories for which the  distance measure value
# between those trajectories is m, for all values m; eA is the array of
# distance measure values between trajectories i and j. 
# What is returned is a collection or set of tuples, (k, i, j), where k 
# is the value on a specific distance measure (i.e. edit distance) 
# and i and j are trajectories.
def findPairs(eA, m):

    answer = []
    added = np.zeros((len(eA), len(eA)))

    for k in range(m+1):
        for i in range(len(eA)):
            v = []
            for j in range(len(eA)):
                if (added[i,j] == 0):
                    if (eA[i,j] == k):
                        added[i, j] = added[j, i] = 1
                        v.append((k, i, j))
            if v:
                answer.append(v)
    return np.array(answer, dtype='object')

# NOT USED AS THIS IS SIGNIFICANTLY SLOWER 
@njit
def findPairs2(eA, m):

    answer_new = np.empty((0, 150, 3), dtype='int')

    added = np.zeros((len(eA), len(eA)), dtype='bool')

    max_answer_part_index = 0
    for k in range(m+1):
        for i in range(len(eA)):
            answer_part = np.empty((1, 150, 3), dtype='int')
            answer_part.fill(-1)
            answer_part_index = 0
            for j in range(len(eA)):
                if (added[i,j] == False):
                    if (eA[i,j] == k):
                        added[i, j] = added[j, i] = True
                        answer_part[0, answer_part_index, 0] = k
                        answer_part[0, answer_part_index, 1] = i
                        answer_part[0, answer_part_index, 2] = j
                        if(max_answer_part_index < answer_part_index):
                            max_answer_part_index = answer_part_index
                        answer_part_index = answer_part_index + 1
            if answer_part_index:
                answer_new = np.concatenate((answer_new, answer_part), axis=0)

    return answer_new


# find the mean edit distance value for those set of trajectories, given by the argument 'answers', 
# whose distance measure value equals m. Return an array of tuples, (DID, average), where DID
# corresponds to the specific distance measure values and the average equals the mean of another 
# distance metric for that set of trajectories.
def findMean(answers, dA):

    avg = []

    max_DID = 0
    for answer in answers:
        value = 0	
        for i in range(len(answer)):
            DID, row, col = answer[i]
            value += dA[row][col]

        average = value/len(answer)
        avg.append((DID, average))
        if(max_DID < DID):
            max_DID = DID

    return np.array(avg), max_DID

# NOT USED AS THIS IS SIGNIFICANTLY SLOWER 
@njit
def findMean2(answers, dA):

    avg_new = np.empty((0,2), dtype='float')

    for j in range(answers.shape[0]):
        value = 0	
        for i in range(answers.shape[1]):
            if(answers[j][i][0] == -1):
                break
            DID = answers[j][i][0]
            row = answers[j][i][1]
            col = answers[j][i][2]
            value += dA[row][col]

        DID_and_average = np.zeros((1,2), dtype='float')
        DID_and_average[0][0] = DID
        DID_and_average[0][1] = value/i
        avg_new = np.concatenate((avg_new, DID_and_average), axis=0)

    return avg_new

# plot the average values of the edit distance for each disc intersection value
def plotCorr(x_y_pair, max_DID):

    x, y = map(list, (zip(*x_y_pair)))

    binVals = dict()
    for avgVal in x_y_pair:
        binVals.setdefault(avgVal[0], []).append(avgVal[1])

    meanVals = []
    pertile_5 = []
    pertile_95 = []

    pertile_10 = []
    pertile_90 = []

    pertile_25 = []
    pertile_75 = []

    pertile_x = []
    for i in range(max_DID+1):
        if i in binVals:
            pertile_5.append(np.percentile(binVals[i], 5))
            pertile_95.append(np.percentile(binVals[i], 95))

            pertile_10.append(np.percentile(binVals[i], 10))
            pertile_90.append(np.percentile(binVals[i], 90))

            pertile_25.append(np.percentile(binVals[i], 25))
            pertile_75.append(np.percentile(binVals[i], 75))

            meanVals.append(np.mean(binVals[i]))
            pertile_x.append(i)

    #plt.scatter(x,y, c ='r')
    plt.plot(pertile_x, meanVals, color = 'b')
    plt.fill_between(pertile_x, pertile_95, pertile_5, color='red', alpha=0.2)
    plt.fill_between(pertile_x, pertile_90, pertile_10, color='blue', alpha=0.2)
    plt.fill_between(pertile_x, pertile_75, pertile_25, color='green', alpha=0.2)

def correlco(mean):
	
	x_distance, y_mean = map(list, (zip(*mean)))

	return np.corrcoef(x_distance, y_mean)