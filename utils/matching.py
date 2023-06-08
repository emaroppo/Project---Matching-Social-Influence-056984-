import numpy as np
from scipy.optimize import linear_sum_assignment


'''
--------------------
Professor's solution; won't work with our code for some reason
--------------------
'''
#Step 1. Subtract the smallest entry in each row from all the entries of its row.
def step1(m):
    for i in range(m.shape[0]):
        m[i,:] -= m[i,:].min()
    return m

#Step 2. Subtract the smallest entry in each column from all the entries of its column.
def step2(m):
    for i in range(m.shape[1]):
        m[:,i] -= m[:,i].min()
    return m

#Step 3. Cover all zeros in the matrix using the minimum number of horizontal and vertical lines.
def step3(m):
    dim = m.shape[0]
    assigned = np.array([])
    assignments = np.zeros(m.shape, dtype=int)

    for i in range(dim):
        for j in range(0,dim):
            if m[i,j] == 0 and np.sum(assignments[i,:]) == 0 and np.sum(assignments[:,j]) == 0:
                assignments[i,j] = 1
                assigned=np.append(assigned, i)
    
    rows = np.linspace(0,dim-1,dim).astype(int)
    marked_rows = np.setdiff1d(rows, assigned.astype(int))
    new_marked_rows = marked_rows.copy()

    marked_cols = np.array([])

    while len(new_marked_rows) > 0:
        new_marked_cols = np.array([], dtype=int)
        for nr in new_marked_rows:
            zeros_cols = np.argwhere(m[nr,:] == 0).flatten()
            new_marked_cols = np.append(new_marked_cols, np.setdiff1d(zeros_cols, marked_cols))
        marked_cols = np.append(marked_cols, new_marked_cols)
        new_marked_rows = np.array([], dtype=int)

        for nc in new_marked_cols:
            new_marked_rows = np.append(new_marked_rows, np.argwhere(assignments[:,nc] == 1).flatten())
        marked_rows = np.unique(np.append(marked_rows, new_marked_rows))

    return np.setdiff1d(rows, marked_rows).astype(int), np.unique(marked_cols)

def step_5(m, covered_rows, covered_cols):
    uncovered_rows = np.setdiff1d(np.linspace(0,m.shape[0]-1,m.shape[0]), covered_rows)
    uncovered_cols = np.setdiff1d(np.linspace(0,m.shape[1]-1,m.shape[1]), covered_cols)
    min_val = np.max(m)
    for i in uncovered_rows.astype(int):
        for j in uncovered_cols.astype(int):
            if m[i,j] < min_val:
                min_val = m[i,j]
    for i in uncovered_rows.astype(int):
        m[i,:] -= min_val
    for j in covered_cols.astype(int):
        m[:,j] += min_val
    return m

def find_rows_single_zero(matrix):
    for i in range(matrix.shape[0]):
        if np.sum(matrix[i,:] == 0) == 1:
            j = np.argwhere(matrix[i,:] == 0).flatten()[0]
            return i,j
    return False

def find_cols_single_zero(matrix):
    for i in range(matrix.shape[1]):
        if np.sum(matrix[:,i] == 0) == 1:
            j = np.argwhere(matrix[:,i] == 0).flatten()[0]
            return i,j
    return False

def assignment_single_zero(m, assignment):
    val= find_rows_single_zero(m)

    while val:
        i,j = val
        m[i,j] += 1
        m[:,j] += 1
        assignment[i,j] = 1
        val = find_rows_single_zero(m)
    val= find_cols_single_zero(m)

    while val:
        i,j = val
        m[i,j] += 1
        m[i,:] += 1
        assignment[i,j] = 1
        val = find_cols_single_zero(m)
    return assignment

def first_zero(m):
    return np.argwhere(m == 0)[0][0], np.argwhere(m == 0)[0][1]

def final_assignment(initial_matrix, m):
    assignment=np.zeros(m.shape, dtype=int)
    assignment=assignment_single_zero(m, assignment)
    while np.sum(m == 0) > 0:
        i,j = first_zero(m)
        assignment[i,j] = 1
        m[i,:] += 1
        m[:,j] += 1
        assignment=assignment_single_zero(m, assignment)
    return assignment*initial_matrix, assignment

def hungarian_algorithm_prof(matrix):
    m =matrix.copy()
    step1(m)
    step2(m)
    n_lines = 0
    max_len = np.maximum(m.shape[0], m.shape[1])
    while n_lines != max_len:
        lines=step3(m)
        n_lines = len(lines[0]) + len(lines[1])
        if n_lines != max_len:
            step_5(m, lines[0], lines[1])
    return final_assignment(matrix, m)

'''
--------------------
END Professor's solution
--------------------
'''

'''
--------------------
Online Solution - To do: check if valid
--------------------
'''
def hungarian_algorithm(matrix):
    m = matrix.copy()
    n_rows, n_cols = m.shape
    max_val = np.max(m)

    if n_rows > n_cols:
        m = np.pad(m, ((0, 0), (0, n_rows - n_cols)), mode='constant', constant_values=max_val)
    elif n_cols > n_rows:
        m = np.pad(m, ((0, n_cols - n_rows), (0, 0)), mode='constant', constant_values=max_val)

    assigned_rows, assigned_cols = linear_sum_assignment(m)

    assignment = np.zeros_like(m, dtype=int)
    assignment[assigned_rows, assigned_cols] = 1

    return assignment[:n_rows, :n_cols] * matrix, assignment[:n_rows, :n_cols]
