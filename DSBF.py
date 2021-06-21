import random
import math
import numpy as np
import time
from numba.typed import List
from numba import njit
from numpy.core.numeric import ones

@njit
def sum_ones(l):
    sum = 0
    for x in l:
        sum += x
    return sum

@njit
def mergeSort(arr):
    if len(arr) > 1:
 
         # Finding the mid of the array
        mid = len(arr)//2
 
        # Dividing the array elements
        L = arr[:mid][:]
 
        # into 2 halves
        R = arr[mid:][:]

        L = L.copy()
        R = R.copy()
 
        # Sorting the first half
        mergeSort(L)
 
        # Sorting the second half
        mergeSort(R)
 
        i = j = k = 0
 
        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if sum_ones(L[i]) < sum_ones(R[j]):
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
 
        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
 
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

@njit
def sortInput(input):
    
    mergeSort(input)
    row = len(input)
    col = len(input[0])
    ones_indices = np.empty(col+1, dtype='int')
    ones_indices.fill(-1)
    for i in range(row):
        num_ones = sum_ones(input[i])
        if(ones_indices[int(num_ones)] == -1):
            ones_indices[int(num_ones)] = i

    return input, ones_indices

@njit
def EUCMOD(a, b):
    return (((a % b) + b) % b) if a < 0 else a % b

@njit
def ModifiedModulo(a, c_mod):
    return EUCMOD((a + math.floor(c_mod/2)), c_mod) - math.floor(c_mod/2)

@njit
def dotProduct(x, y):
    result = 0
    for i in range(len(x)):
        result = result + x[i]*y[i]

    return result

@njit
def computeSignature(x, c_mod, c_div, rows_M, matrix_M):

    signature = np.zeros(rows_M, dtype='bool')
    
    for i in range(rows_M):
        temp = math.floor( ModifiedModulo(dotProduct(matrix_M[i], x), c_mod) / c_div )
        if(temp == -1):
            signature[i] = True

    return signature

@njit
def initializeFilterAndM(sorted_trajectories, rows_M, c_mod, c_div, delta, num_circles, matrix_M, filter_binary):
    
    for i_col in range(num_circles):
        for j in range(delta):
            num = random.randint(0, 1)
            if(num == 0):
                num = -1
            row = random.randint(0, rows_M-1)
            matrix_M[row][i_col] += num

    for i in range(len(sorted_trajectories)):
        filter_binary[i][:] = computeSignature(sorted_trajectories[i], c_mod, c_div, rows_M, matrix_M)

@njit
def findIndices(ones_indices, num_ones, psi, max_data_index, max_ones):
    m = 0
    no_near = False
    first_index = -1
    last_index = -1
    min_val = int(num_ones - psi)
    max_val = int(num_ones + psi)

    while(min_val < 0):
        min_val = min_val + 1
    while(max_val > max_ones):
        max_val = max_val - 1
    
    while ((min_val + m) < max_val and (ones_indices[min_val + m] == -1)):
        m = m + 1

    if ((min_val + m) == max_val):
        if (ones_indices[max_val] != -1):
            first_index = ones_indices[max_val]
        else:
            no_near = True
    else:
        first_index = ones_indices[min_val + m]
    
    if(no_near == False):
        is_set = False
        j = max_val + 1
        while(is_set == False and j <= max_ones):
            if(ones_indices[j] != -1):
                last_index = ones_indices[j] - 1
                is_set = True
            j = j + 1
        if(is_set == False):
            last_index = max_data_index

    return first_index, last_index, no_near

@njit
def computeGap(sig_x, sig_y, c_mod, c_div, rows_M):
    gap = 0
    difference = 0

    for i in range(rows_M):
        if(sig_x[i] == True):
            if(sig_y[i] == True):
                difference = 0
            else:
                difference = -1
        else:
            if(sig_y[i] == True):
                difference = 1
            else:
                difference = 0
        gap = gap + abs(c_div * ModifiedModulo(difference, c_mod))

    return gap

@njit
def checkSimilarity(query,  rows_M, c_mod, c_div, psi, ones_indices, max_data_index, num_circles, matrix_M, filter_binary):
    similarity = False

    num_ones = sum_ones(query)
    first_index, last_index, no_near = findIndices(ones_indices, num_ones, psi, max_data_index, num_circles)
    if(no_near == False):
        valid_size = last_index - first_index + 1
        valid_input = [ [ 0 for i in range(rows_M) ] for j in range(valid_size) ]

        for k in range(valid_size):
            for l in range(rows_M):
                valid_input[k][l] = filter_binary[k + first_index][l]

        valid_input, ones_indices_sig = sortInput(valid_input)

        signature = computeSignature(query, c_mod, c_div, rows_M, matrix_M)

        num_ones_sig = sum_ones(signature)
        first_index2, last_index2, no_near_sig = findIndices(ones_indices_sig, num_ones_sig, psi, valid_size-1, rows_M)

        if(no_near_sig == False):
            i = first_index2
            gap = 0
            while(similarity == False and i <= last_index2):
                gap = computeGap(signature, valid_input[i], c_mod, c_div, rows_M)
                if(gap <= psi):
                    similarity = True
                i = i + 1
                
    return similarity