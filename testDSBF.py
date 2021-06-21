from lsh import *
from DSBF import *
import time
from numba.typed import List
from numba import njit

#n equals the number of discs, Rs the radius for each discs
n = 500
Rs = np.repeat(500,n)

query_size = 100
c = 2
r = 5
epsilon = 0.2

delta = 1
psi = r
c_mod = 2
c_div = 1

a = open("trajectories/taxi1000.txt", 'rb')
trajectories = cPickle.load(a)
a.close()
noTraj = len(trajectories)

plotTrajectories(trajectories)

circles = findCircles(trajectories, n)

plotCircles(circles, Rs)

#create the binayVectors and corresponding DID array
max_length = max(map(len, trajectories))
trajectories = np.array([xi+[[0,0]]*(max_length-len(xi)) for xi in trajectories], dtype=float)
bv = binaryVectors(trajectories, circles, Rs)

queries = bv[len(bv)-query_size:]
bv = bv[:len(bv)-query_size]
#np.savetxt('test.txt',bv, fmt='%d', delimiter=' ')
#np.savetxt('test_queries.txt',queries, fmt='%d', delimiter=' ')

t1 = time.time()
rows_M = math.ceil( 24 * c**2/(c-1) * max(r, 2/(c-1) * math.log(1/epsilon, 2)) )
sorted_trajectories, ones_indices = sortInput(bv)

filter_binary = np.zeros((len(sorted_trajectories), rows_M), dtype='bool')
matrix_M = np.zeros((rows_M, n), dtype='int')
initializeFilterAndM(sorted_trajectories, rows_M, c_mod, c_div, delta, n, matrix_M, filter_binary)
t2 = time.time()
print("time taken for DSBF init = " + str(t2-t1) + " sec")

t1 = time.time()
answers = [ False for i in range(len(queries)) ]
for i in range(len(queries)):
    answers[i] = checkSimilarity(queries[i], rows_M, c_mod, c_div, psi, ones_indices, len(sorted_trajectories)-1, n, matrix_M, filter_binary)
t2 = time.time()
print("time taken for "+ str(len(queries)) + " DSBF queries = " + str(t2-t1) + " sec")
print("query results: " + str(answers))