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
        new_field = np.zeros((n, n), dtype='uint8')
        for i in range(n):
            for j in range(n):
                neighs = calc_neighs(cur_field, i, j)
                if cur_field[i][j] and neighs == 2:
                    new_field[i][j] = 1
                if neighs == 3:
                    new_field[i][j] = 1
        cur_field = new_field
    return cur_field

def generate_field(delta, flat=False):
    field = np.random.randint(0, 2, size=(20, 20), dtype='uint8')
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

def train_row_to_windowed_data(row):
    delta, start_field, end_field = row[0], row[1:401].reshape(20, 20), row[401:].reshape(20, 20)
    padded = np.pad(end_field, delta, mode="constant", constant_values=-1)
    rows = []
    labels = []
    n = len(start_field)
    for i in range(n):
        for j in range(n):
            window = padded[i:i+2*delta+1, j:j+2*delta+1]
            cell_status = start_field[i][j]
            rows.append(window.ravel())
            labels.append(cell_status)
    return (np.array(rows), np.array(labels).reshape(-1, 1))

def extract_features_from_raw_data(raw_data):
    X, y = [], []
    for row_idx in range(raw_data.shape[0]):
        field_X, field_y = train_row_to_windowed_data(raw_data[row_idx, :])
        X.append(field_X)
        y.append(field_y)
    return np.vstack(X), np.vstack(y)

def field_to_window_rows(end_field, delta):
    padded = np.pad(end_field, delta, mode="constant", constant_values=-1)
    rows = []
    
    n = len(end_field)
    for i in range(n):
        for j in range(n):
            window = padded[i:i+2*delta+1, j:j+2*delta+1]
            rows.append(window.ravel())
    return np.array(rows)

def predict_field(end_field, delta, model):

    
    rows = field_to_window_rows(end_field, delta)
    
    field = model.predict(rows)
    return field

def train_sample_windowize(field, delta=1, n=20):
    """ Same as the above, but with custom delta """
    padded = np.pad(field, delta, mode='constant', constant_values=-1)
    X = np.zeros((n * n, (1 + delta * 2) ** 2))
    for i in range(n):
        for j in range(n):
            X[i * n + j] = padded[i:i + 2 * delta + 1, j:j + 2 * delta + 1].ravel()
    return X

def window_data_proc(X, y=None, delta=1):
    """ Reformat data in window form and prepare for training """
    vectorize_windowing = lambda row: train_sample_windowize(row.reshape(20, 20), delta=delta)

    X = np.vstack(np.apply_along_axis(vectorize_windowing, 1, X))
    if y is not None:
        y = np.vstack(y.ravel())
        return X, y
    return X
