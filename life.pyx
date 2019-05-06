cimport cython
import numpy as np

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef int calc_neighs(unsigned char[:, :] field, int i, int j, int n, int k):
    cdef:
        int neighs = 0;
        int i_min = i - 1;
        int i_pl = i + 1;
        int j_min = j - 1;
        int j_pl = j + 1;
    neighs = 0
    if i_min >= 0:
        if j_min >= 0:
            neighs += field[i_min, j_min]
        neighs += field[i_min, j]
        if j_pl < k:
            neighs += field[i_min, j_pl]
    if j_min >= 0:
        neighs += field[i, j_min]
    if j_pl < k:
        neighs += field[i, j_pl]
    if i_pl < n:
        if j_min >= 0:
            neighs += field[i_pl, j_min]
        neighs += field[i_pl, j]
        if j_pl < k:
            neighs += field[i_pl, j_pl]
    return neighs

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef make_move(unsigned char[:, :] field, int moves):
    cdef:
        int _, i, j, neighs;
        int n, k;
        int switch = 0;
        unsigned char[:, :] cur_field;
        unsigned char[:, :] next_field;
    cur_field = np.copy(field)
    next_field = np.zeros_like(field, 'uint8')
    n = field.shape[0]
    k = field.shape[1]
    for _ in range(moves):
        if switch == 0:
            for i in range(n):
                for j in range(k):
                    neighs = calc_neighs(cur_field, i, j, n, k)
                    if cur_field[i, j] and neighs == 2:
                        next_field[i, j] = 1
                    elif neighs == 3:
                        next_field[i, j] = 1
                    else:
                        next_field[i, j] = 0
        else:
            for i in range(n):
                for j in range(k):
                    neighs = calc_neighs(next_field, i, j, n, k)
                    if next_field[i, j] and neighs == 2:
                        cur_field[i, j] = 1
                    elif neighs == 3:
                        cur_field[i, j] = 1
                    else:
                        cur_field[i, j] = 0
        switch = (switch + 1) % 2
    return np.array(next_field if switch else cur_field)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef float fitness_score(unsigned char[:, :] gene, unsigned char[:, :]Y, int delta):
    return (make_move(gene, moves=delta) == Y).sum() / 400.0