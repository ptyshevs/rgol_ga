import numpy as np

M = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
NROW = NCOL = 20

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
    field = make_move(field, moves=delta)
    return field

def generate_data_row(delta):
    start_field = generate_field(delta)
    end_field = make_move(start_field, delta)
    return np.hstack((np.array(delta).reshape(1, -1), start_field.reshape(1, -1), end_field.reshape(1, -1))).ravel()
  
def generate_sample(delta=1, skip_first=5, ravel=True):
    """
    Generate training sample
    
    @return: (end_frame, start_frame)
    """
    start_frame = generate_field(skip_first)
    end_frame = make_move(start_frame, delta)
    return (end_frame, start_frame) if not ravel else (end_frame.ravel(), start_frame.ravel())

def generate_samples(delta=1, n=32):
    """
    Generate batch of samples
    
    @return: (end_frames, start_frames)
    """
    X = np.zeros((n, NROW * NCOL))
    Y = np.zeros((n, NROW * NCOL))
    for i in range(n):
        x, y = generate_sample(delta)
        X[i, :] = x
        Y[i, :] = y
    return X, Y

def data_generator(delta=1, batch_size=32):
    """
    Can be used along with .fit_generator to generate training samples on the fly
    """
    while True:
        yield generate_samples(delta=delta, n=batch_size)
