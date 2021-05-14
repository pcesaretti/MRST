import _pickle as cPickle
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

colors = [ 'r', 'b','c', 'g', 'y', 'k', 'm', 'r']

def plotTrajectories(trajectories):
    
    fig, ax = plt.subplots(figsize=(15,7))
    
    for c, trajectory in enumerate(trajectories):
        ax.scatter([a for [a,b] in trajectory],[b for [a,b] in trajectory], color = colors[c%8], s = 5)
    plt.axis('equal')

    
    
def findCircles(trajectories, n):
    
    cs = []

    xmax, ymax =  map(max, zip(*[x for l in trajectories for x in l])) 
    xmin, ymin = map(min, zip(*[x for l in trajectories for x in l]))
    
    A = (xmax-xmin)*(ymax-ymin)
    
    for i in range(n):

        xcircle = random.uniform(xmin,xmax)
        ycircle = random.uniform(ymin,ymax)

        cs.append([xcircle,ycircle])

    return cs

def plotCircles(cs, Rs):

    for i,[xcircle,ycircle] in enumerate(cs):
        circle = plt.Circle((xcircle,ycircle), Rs[i], color='r', alpha=0.2)
        plt.gcf().gca().add_artist(circle)
        
def binaryVectors(trajectories, cs, Rs):

    vs = []
    n = len(cs)
    
    for trajectory in trajectories:
        v = np.zeros(n)
        for i in range(n):
            xcircle,ycircle = cs[i]
            for x,y in trajectory:
                if (x - xcircle)**2 + (y - ycircle)**2 < Rs[i]**2:
                    v[i] = 1

        vs.append(v)

    return np.array(vs)

def makeCdist(trajectories, cs, vs):

    cdist = np.zeros((len(trajectories),len(trajectories)))
    
    for i in range(0,len(trajectories)):
        for j in range(i,len(trajectories)):
            cdist[i,j] = cdist[j,i] = len(np.bitwise_xor(vs[i].astype(int),vs[j].astype(int)).nonzero()[0])
    
    return cdist


# create a signature that will note the order of circles in which a trajectory is located;
# if a coordinate of the trajectory is located within a circle, c[i] with radius Rs[i], then
# we append the index of that circle to an array called edit_string.
 	
def editVector(trajectories, cs, Rs):

	ed = []
	n = len(cs)

	for trajectory in trajectories:
		edit_string = []
		v = np.zeros(n)
		for i in range(n):
			xcircle, ycircle = cs[i]
			for x, y in trajectory:
				if ((x - xcircle)**2 + (y - ycircle)**2 < Rs[i]**2
					and v[i] != 1):
						v[i] = 1
						edit_string.append(i)

		ed.append(edit_string)

	return np.array(ed) 	

# create an array, listing the edit distances between trajectories. Location A[i,j] will
# list the distance between trajectory i and j

def editArray(ts, edit_vector):

	array = np.zeros((len(ts), len(ts)))

	for i in range(0, len(ts)):
		for j in range(i, len(ts)):
			array[i, j] = array[j, i] = levenshtein(edit_vector[i], edit_vector[j])

	return array


# edit distance test for error; this algorithm uses dynamic programming to keep track of previously computed distances

def levenshtein(s, t):
        if False: return 0
        #if s == t: return 0
        elif len(s) == 0: return len(t)
        elif len(t) == 0: return len(s)
        v0 = [None] * (len(t) + 1)
        v1 = [None] * (len(t) + 1)
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

	return np.array(answer)

# find the mean edit distance value for those set of trajectories, given by the argument 'answers', 
# whose distance measure value equals m. Return an array of tuples, (DID, average), where DID
# corresponds to the specific distance measure values and the average equals the mean of another 
# distance metric for that set of trajectories.

def findMean(answers, dA):

	avg = []

	for answer in answers:
		value = 0	
		for i in range(len(answer)):
			DID, row, col = answer[i]
			value += dA[row][col]
		
		average = value/len(answer)
		avg.append((DID, average))



	return np.array(avg)

# plot the average values of the edit distance for each disc intersection value

def plotCorr(x_y_pair):

	x, y = map(list, (zip(*x_y_pair)))

	plt.scatter(x,y, c ='r')

def correlco(mean):
	
	x_distance, y_mean = map(list, (zip(*mean)))

	return np.corrcoef(x_distance, y_mean)