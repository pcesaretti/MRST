import scipy.spatial as sp
import numpy as np

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