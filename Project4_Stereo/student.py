import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    from numpy.linalg import pinv
    from numpy.linalg import inv
    from numpy.linalg import norm
    L = np.array(lights)
    dim1 = np.array(images).shape[1]
    dim2 = np.array(images).shape[2]
    dim3 = np.array(images).shape[3]

    I = np.array(images).reshape((-1, dim1 * dim2 * dim3))
    G = np.dot(pinv(L),I)
    k = norm(G,2,0).reshape((1,-1)) #column based 2 norm

    k[k<1e-7] = 0
    Nravel = np.divide(G, k,out=np.zeros_like(G), where=k!=0, dtype=np.float64)

    N = Nravel.reshape((-1, dim1, dim2, dim3))
    N = np.mean(N, axis=3)

    N = np.transpose(N, (1, 2, 0))
    k = k.reshape(-1).reshape((dim1, dim2, dim3))

    return k, N

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    # print (K.shape, K)
    # print (Rt.shape, Rt)
    # print (points.shape, points)
    from numpy.linalg import multi_dot
    # print (points)
    # print (points.shape)
    a = points.reshape((points.shape[0]*points.shape[1], -1))
    b = np.ones(((points.shape[0]*points.shape[1], 1)))
    points_proj = np.hstack((a, b))
    print(points_proj.shape)
    # res1 = np.dot(K, Rt)
    # print(res1.shape)
    res = multi_dot((K, Rt, points_proj.T))
    print(res.shape)

    c1 = res[0,:] / res[2,:]
    c2 = res[1,:] / res[2,:]
    c = np.vstack((c1, c2)).T
    projections = c.reshape((points.shape[0], points.shape[1], -1))
    print (projections.shape)
    return projections


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v11 = [ x111, x121, x211, x221, x112, x122, x212, x222 ]
    v12 = [ x111, x121, x211, x221, x112, x122, x212, x222 ]
    v21 = [ x111, x121, x211, x221, x112, x122, x212, x222 ]
    v22 = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    2 X 2 X (2 X 2**2)

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:              81    *      ([1,2,3]*9)

                        9       9          1           25
                          (2,2)
        normalized -- heigth x width x (channels * ncc_size**2) array
    """

    """
    from numpy.linalg import norm
    def partition(image,ncc_size):
        overlapR = image.shape[0] % ncc_size
        overlapC = image.shape[1] % ncc_size
        if (overlapR) == 0 and (overlapC) == 0:
            partition = image
        elif (overlapR) == 0 and (not (overlapC) == 0):
            partition = image[:-overlapC,:]
        elif (not (overlapR) == 0) and (overlapC) == 0:
            partition = image[:,:-overlapR]
        else:
            partition = image[:-overlapC,:-overlapR]
        return partition

    p = partition(image, ncc_size)

    p = p.reshape(p.shape[0]//ncc_size, 
                  ncc_size, p.shape[1]//ncc_size, 
                  ncc_size, p.shape[2]).swapaxes(1, 2).reshape(-1, 5, 5, p.shape[2])
    """
    from numpy.linalg import norm

    p = image

    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


    offset = ncc_size//2
    windows = np.zeros((image.shape[0] - offset*2, image.shape[1] - offset*2,image.shape[2],ncc_size,ncc_size))

    print(windows.shape)

    print(image.shape, "image")

    for i in range(image.shape[0]-ncc_size):
        windows[i, :, 0, :, :] = np.transpose(rolling_window(image[i:i+ncc_size,:,0], ncc_size),(1,0,2))
        windows[i, :, 1, :, :] = np.transpose(rolling_window(image[i:i+ncc_size,:,1], ncc_size),(1,0,2))
        windows[i, :, 2, :, :] = np.transpose(rolling_window(image[i:i+ncc_size,:,2], ncc_size),(1,0,2))

    p = windows

    if p.shape[2] == 1:
        # Mean
        pMeanVal = np.mean(np.mean(p, axis = 3),axis=3)
        means = np.array(list(map(lambda x : np.full((ncc_size,ncc_size),x), pMeanVal[:,:].reshape(-1)))).reshape(p.shape[0],p.shape[1],ncc_size,ncc_size)
        p = np.subtract(p,means)
    
        # Norm
        normedP = norm(norm(norm(p,axis=4),axis=3),axis=2)

        normedP[np.where(normedP < 1e-6)] = 0

        normMapped = np.array(list(map(lambda x : np.full((p.shape[2],ncc_size,ncc_size),x), normedP[:,:].reshape(-1)))).reshape(p.shape[0],p.shape[1],p.shape[2],ncc_size,ncc_size)

        ans = np.divide(p,normMapped)

    elif p.shape[2] == 3:
        # Mean
        pMeanVal = np.mean(np.mean(p, axis = 3),axis=3)
        print(pMeanVal.shape)

        meansR = np.array(list(map(lambda x : np.full((ncc_size,ncc_size),x), pMeanVal[:,:,0].reshape(-1)))).reshape(p.shape[0],p.shape[1],ncc_size,ncc_size)
        meansG = np.array(list(map(lambda x : np.full((ncc_size,ncc_size),x), pMeanVal[:,:,1].reshape(-1)))).reshape(p.shape[0],p.shape[1],ncc_size,ncc_size)
        meansB = np.array(list(map(lambda x : np.full((ncc_size,ncc_size),x), pMeanVal[:,:,2].reshape(-1)))).reshape(p.shape[0],p.shape[1],ncc_size,ncc_size)

        p[:,:,0] = np.subtract(p[:,:,0],meansR)
        p[:,:,1] = np.subtract(p[:,:,1],meansG)
        p[:,:,2] = np.subtract(p[:,:,2],meansB)
    
        # Norm
        normedP = norm(norm(norm(p,axis=4),axis=3),axis=2)


        normedP[np.where(normedP < 1e-6)] = 0

        normMapped = np.array(list(map(lambda x : np.full((p.shape[2],ncc_size,ncc_size),x), normedP[:,:].reshape(-1)))).reshape(p.shape[0],p.shape[1],p.shape[2],ncc_size,ncc_size)

        ans = np.zeros((p.shape[0],p.shape[1],p.shape[2],ncc_size,ncc_size))

        np.divide(p,normMapped,out=ans,where=normMapped!=0)

    pad = np.zeros((image.shape[0], image.shape[1],image.shape[2],ncc_size,ncc_size))

    pad[offset:-offset,offset:-offset,:,:,:] = ans

    pad = pad.reshape((pad.shape[0],pad.shape[1],-1))

    return pad

    #raise NotImplementedError()


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    raise NotImplementedError()
