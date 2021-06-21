import scipy.spatial as sp
import numpy as np
from numba import njit

def distanceBetweenCurves(C1, C2):
    D = sp.distance.cdist(C1, C2, 'euclidean')

    #none symmetric Hausdorff distances
    H1 = np.max(np.min(D, axis=1))
    H2 = np.max(np.min(D, axis=0))

    return  np.maximum(H1,H2)

    #return (H1 + H2) / 2.

def distanceMatrixOfCurves(Curves):
    numC = len(Curves)

    D = np.zeros((numC, numC))
    for i in range(0, numC-1):
        for j in range(i+1, numC):
            #print (i,j)
            D[i, j] = D[j, i] = distanceBetweenCurves(Curves[i], Curves[j])
    return D

@njit
def distanceBetweenSurfaces(vol_a,vol_b):
    dist_lst = np.empty(0)
    for idx in range(len(vol_a)):
        if(vol_a[idx][0] == -1):
            break
        dist_min = 100000.0        
        for idx2 in range(len(vol_b)):
            if(vol_b[idx2][0] == -1):
                break
            diff = np.array([vol_a[idx][0]-vol_b[idx2][0], vol_a[idx][1]-vol_b[idx2][1], vol_a[idx][2]-vol_b[idx2][2]])
            dist= np.linalg.norm(diff)
            if dist_min > dist:
                dist_min = dist
        dist_lst = np.append(dist_lst, dist_min)
    return np.max(dist_lst)

@njit
def distanceMatrixOfSurfaces(Surfaces):
    numC = len(Surfaces)

    D = np.zeros((numC, numC))
    for i in range(0, numC-1):
        for j in range(i+1, numC):
            D[i, j] = D[j, i] = distanceBetweenSurfaces(Surfaces[i], Surfaces[j])
    return D

# def distance_on_sphere_numpy(coordinate_array):
#     """
#     Compute a distance matrix of the coordinates using a spherical metric.
#     :param coordinate_array: numpy.ndarray with shape (n,2); latitude is in 1st col, longitude in 2nd.
#     :returns distance_mat: numpy.ndarray with shape (n, n) containing distance in km between coords.
#     """
#     # Radius of the earth in km (GRS 80-Ellipsoid)
#     EARTH_RADIUS = 6371.007176 

#     # Unpacking coordinates
#     latitudes = coordinate_array[:, 0]
#     longitudes = coordinate_array[:, 1]
#     n_pts = coordinate_array.shape[0]

#     # Convert latitude and longitude to spherical coordinates in radians.
#     degrees_to_radians = np.pi/180.0
#     phi_values = (90.0 - latitudes)*degrees_to_radians
#     theta_values = longitudes*degrees_to_radians

#     # Expand phi_values and theta_values into grids
#     theta_1, theta_2 = np.meshgrid(theta_values, theta_values)
#     theta_diff_mat = theta_1 - theta_2

#     phi_1, phi_2 = np.meshgrid(phi_values, phi_values)

#     # Compute spherical distance from spherical coordinates
#     angle = (np.sin(phi_1) * np.sin(phi_2) * np.cos(theta_diff_mat) + 
#            np.cos(phi_1) * np.cos(phi_2))
#     arc = np.arccos(angle)

#     # Multiply by earth's radius to obtain distance in km
#     return arc * EARTH_RADIUS    