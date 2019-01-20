import numpy as np

M = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

def calc_neighs(field, i, j):
    neighs = 0
    n = len(field)
    for m in M:
        row_idx = m[0] + i
        col_idx = m[1] + j
        if 0 <= row_idx < n and 0 <= col_idx < n:
            if field[row_idx][col_idx]:
                neighs += 1
    return neighs

def make_move(field, moves=1):
    n = len(field)
    cur_field = field
    for _ in range(moves):
        new_field = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                neighs = calc_neighs(cur_field, i, j)
                if cur_field[i][j] and neighs == 2:
                    new_field[i][j] = 1
                if neighs == 3:
                    new_field[i][j] = 1
        cur_field = new_field
    return cur_field

def generate_field(delta):
    field = np.random.randint(0, 2, size=(20, 20))
    field = make_move(field, moves=5)
    return field

def generate_data_row(delta):
    start_field = generate_field(delta)
    end_field = make_move(start_field, delta)
    return np.hstack((np.array(delta).reshape(1, -1), start_field.reshape(1, -1), end_field.reshape(1, -1))).ravel()