from lsh import addRandomTimeDim
from os.path import splitext
import numpy as np

datasetFilename = "taxi1000.txt"

datasetFilenameShort = splitext(datasetFilename)[0]
datasetFilename3d = datasetFilenameShort + '_3d.npy'

trajectories = np.load("trajectories/" + datasetFilename, allow_pickle=True)
addRandomTimeDim(trajectories)
np.save("trajectories/" + datasetFilename3d, trajectories)