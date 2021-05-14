from joblib import Parallel, delayed
import time
import frechet
import haus
import _pickle as cPickle

a = open("trajectories/pkdd5000-utm-box",'rb')
trajectories = cPickle.load(a)

def outer_h(traj,trajectories):
    return [haus.distanceBetweenCurves(traj,tr) for tr in trajectories]


start = time.time()
results = Parallel(n_jobs = 10)(delayed(outer_h)(trajectoryA, trajectories) for trajectoryA in trajectories)
end = time.time()

print(end-start)
cPickle.dump(results, open( "haus5000-utm-box", "wb" ) )