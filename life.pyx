cimport cython
import numpy as np

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef int calc_neighs(unsigned char[:, :] field, int i, int j, int n):
    cdef:
        int neighs = 0;
        int k, row_idx, col_idx;
    neighs = 0
    if i - 1 >= 0 and j - 1 >= 0 and field[i - 1, j - 1]:
        neighs += 1
    if i - 1 >= 0 and field[i - 1, j]:
        neighs += 1
    if i - 1 >= 0 and j + 1 < n and field[i - 1, j + 1]:
        neighs += 1
    if j - 1 >= 0 and field[i, j - 1]:
        neighs += 1
    if j + 1 < n and field[i, j + 1]:
        neighs += 1
    if i + 1 < n and j - 1 >= 0 and field[i + 1, j - 1]:
        neighs += 1
    if i + 1 < n and field[i + 1, j]:
        neighs += 1
    if i + 1 < n and j + 1 < n and field[i + 1, j + 1]:
        neighs += 1
    return neighs

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef make_move(unsigned char[:, :] field, int moves):
    cdef:
        int _, i, j, neighs;
        int n;
        int switch = 0;
        unsigned char[:, :] cur_field;
        unsigned char[:, :] next_field;
    cur_field = np.copy(field)
    next_field = np.zeros_like(field, 'uint8')
    n = len(field)
    for _ in range(moves):
        if switch == 0:
            for i in range(n):
                for j in range(n):
                    neighs = calc_neighs(cur_field, i, j, n)
                    if cur_field[i, j] and neighs == 2:
                        next_field[i, j] = 1
                    elif neighs == 3:
                        next_field[i, j] = 1
                    else:
                        next_field[i, j] = 0
        else:
            for i in range(n):
                for j in range(n):
                    neighs = calc_neighs(next_field, i, j, n)
                    if next_field[i, j] and neighs == 2:
                        cur_field[i, j] = 1
                    elif neighs == 3:
                        cur_field[i, j] = 1
                    else:
                        cur_field[i, j] = 0
        switch = (switch + 1) % 2
    return np.array(next_field if switch else cur_field)