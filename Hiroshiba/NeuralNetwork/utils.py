import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    pass

## draw
def draw_weight(data, size):
    Z = data.reshape(size).T
    plt.imshow(Z, interpolation='none')
    plt.xlim(0,size[0])
    plt.ylim(0,size[1])
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

## make small movies from all movies
def makeInputsAsMovie(data, n_frame):
    frame = data.shape[0]
    n_pximage = data.shape[1] * data.shape[2]
    n_pxmovie = n_pximage * n_frame
    n_movie = frame // n_frame
    movies = np.zeros((n_movie, n_pxmovie), dtype=np.float32)
    for i in range(n_movie):
        i_frame = i * n_frame
        movies[i,:] = np.reshape( data[i_frame:i_frame+n_frame, : , :], (1, -1) )
    return movies

def bindInputs(data, n_frame):
    return np.reshape(data, (data.shape[0]/n_frame, -1))

def splitInputs(data, n_frame):
    return np.reshape(data, (data.shape[0], n_frame, -1))

## util numpy
def vstack_(a, b):
    if isinstance(a, np.ndarray) and a.size==0:
        return b
    else:
        return np.vstack((a, b))