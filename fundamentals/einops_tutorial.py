
import numpy as np
from numpy.lib.stride_tricks import as_strided
x = np.array([[0, 1, 2, 3, 4],
              [5, 6, 7, 8, 9]], dtype=np.int32)

x.strides

y = np.arange(2*3*4).reshape(2, 3, 4)
y.strides

y[1,1,2]
offset = np.sum(y.strides * np.array([1, 1, 2]))
offset/y.itemsize

#ex 1
x = np.arange(1, 26, dtype=np.int8).reshape(5, 5)
xp = as_strided(x, shape=(3,), strides=(1, ))
x[0,:3]

# ex 2 slice first 8 elements
xp = as_strided(x, shape=(8,), strides=(1, ))
xp

# flatten
xp = as_strided(x, shape=(np.prod(x.shape), ), strides=(1,))
x.ravel()

# skip

as_strided(x, shape=(3,), strides=(2,))

# column
x = np.arange(1, 26, dtype=np.int16).reshape(5, 5)
as_strided(x, shape=(4,), strides=(x.strides[0],))

# diagonal

as_strided(x, shape=(x.shape[0],), strides=(x.strides[0]+x.strides[1],))
x.diagonal()

# repeate

as_strided(x, shape=(x.shape[0], ), strides=(0, ))
np.broadcast_to(x[0,0], (x.shape[0], ))

# 2d

as_strided(x, shape=(3, 4), strides=x.strides)
x[:3, :4]

# 2 diagonal
x = np.arange(1, 26, dtype=np.int64).reshape(5, 5)
as_strided(x, shape=(4, 2), strides=(x.strides[0]+x.strides[1], 1*x.itemsize))

# sparse matrix

as_strided(x, shape=(3, 3), strides=(x.strides[0]*2, x.strides[1]*2))
x[::2, ::2]

# transpose

x = np.arange(1, 26, dtype=np.int8).reshape(5, 5)
as_strided(x, shape=(3, 3), strides=(x.strides[1], x.strides[0]))
x.T[:3, :3]

# repeat

as_strided(x, shape=(x.shape[0], 4), strides=(x.strides[0], 0))
np.broadcast_to(x[:, 0, None], (x.shape[0], 4))

x[:,0].shape
x[:,0, None].shape

# reshape

x = np.arange(1, 26, dtype=np.int16)
as_strided(x, shape=(5, 5), strides=(5*x.itemsize, x.itemsize))
x.reshape(5, 5)

# slide 1d window

x = np.arange(1, 26, dtype=np.int16)

w = 3
as_strided(x, shape=(x.size // w, w), strides=(x.itemsize, x.itemsize))

# slide 2d window then flatten
x = np.asarray([0,1,10,11,20,21,30,31,40,41,50,51], np.int8).reshape(6,2)

w = 2
as_strided(x, shape=(x.shape[1] * w, x.shape[0]), strides=(w*x.itemsize, x.itemsize))
as_strided(x, shape=(4,6), strides=(2,1))

# remove last axes
x = np.asarray(range(1,13), np.int8).reshape(3,2,2)
new_shape = (x.shape[0], x.shape[1] * x.shape[2])
new_strides = (x.strides[0], x.strides[-1])
as_strided(x, shape=new_shape, strides=new_strides)
x.reshape(3, 4)

# 2 corner
x = np.asarray(range(1,26), np.int8).reshape(5,5)
as_strided(x, shape=(2,2,2), strides=(x.strides[0]*3,
                                      x.strides[0],
                                      x.strides[-1]))
x.strides

# staggered slicing
as_strided(x, shape=(2,2,3), strides=(x.strides[0]*2,
                                      x.strides[0]+1,
                                      x.strides[-1]))

# repeate 2d array

as_strided(x, shape=(3,2,4), strides=(0, x.strides[0], x.strides[1]))

# 3d transpose
x = np.asarray(range(1,13), np.int32).reshape(3,2,2)
as_strided(x, shape=x.shape, strides=(x.strides[0], x.strides[-1], x.strides[1]))
x.swapaxes(1,2)


# slide 3d window
x = np.asarray(range(1,21), np.int8).reshape(4,5)
w = (2, x.shape[1])
as_strided(x, shape=(3, w[0], w[1]), strides=(x.strides[0], x.strides[0], x.strides[1]))
x.strides

# reshape
x = np.asarray(range(1,13), np.int8)
x.reshape(2, 2, 3)
es = x.strides[-1]
ns = (2, 2, 3)
as_strided(x, shape=(2, 2, 3), strides=(es*ns[1]*ns[2], es*ns[2], es))

# conv
x = np.asarray(range(1,26), np.int8).reshape(5,5)

as_strided(x, shape=(2, 2, 3, 3), strides=(10, 2, 5, 1))


a = np.array([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]], dtype=np.int8)

sub_shape = (3,3)
view_shape = tuple(np.subtract(a.shape, sub_shape) + 1) + sub_shape
strides = a.strides + a.strides

sub_matrices = np.lib.stride_tricks.as_strided(a,view_shape,strides)

conv_filter = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
m = np.einsum('ij,ijkl->kl',conv_filter,sub_matrices)

# repeate
x = np.asarray(range(1,13), np.int64).reshape(2,2,3)

as_strided(x, shape=(2, 2, x.shape[1], x.shape[2]),
           strides=(x.strides[0], 0, x.strides[1], x.strides[-1]))

# reshape
x = np.asarray(range(1,17), np.int64)
shape = 2,2,2,2
as_strided(x, shape=shape, strides=(x.strides[-1]*shape[1]*shape[2]*shape[3],
                                    x.strides[-1]*shape[1]*shape[2],
                                    x.strides[-1]*shape[1],
                                    x.strides[-1],))


# trace

x = np.arange(1, 26, dtype=np.int8).reshape(5, 5)
x.strides
as_strided(x, shape=(x.shape[0],1), strides=(x.strides[0]+x.strides[1], x.strides[-1]))

