import numpy as np


def solve_homography(u, v): # 計算兩個平面之間的3x3映射矩陣 (兩個N x 2的矩陣u和v作為輸入)
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N: # 檢查u和v的維度是否相同
        print('u and v should have the same size')
        return None
    if N < 4: # 至少需要4個點才能進行映射計算
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((2 * N, 9)) # 形成2N x 9的矩陣，其中N是點的數量
    for i in range(N): # 通過迴圈所有的點，並且將u和v的值結合起來得到。每個點將會形成兩行，每行9列，最後得到2N x 9的矩陣
        A[2 * i] = np.array([u[i, 0], u[i, 1], 1, 0, 0, 0, -v[i, 0] * u[i, 0], -v[i, 0] * u[i, 1], -v[i, 0]])
        A[2 * i + 1] = np.array([0, 0, 0, u[i, 0], u[i, 1], 1, -v[i, 1] * u[i, 0], -v[i, 1] * u[i, 1], -v[i, 1]])

    # TODO: 2.solve H with A
    _, _, V = np.linalg.svd(A) # 使用numpy的linalg.svd()函數對A進行SVD分解，得到3個矩陣：U, S和V
    H = V[-1].reshape((3, 3)) # V的最後一列就是homography matrix的解。將它重塑成3 x 3的形狀就得到了homography matrix H

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    dst_pixels = np.vstack((x.flatten(), y.flatten(), np.ones((1, (xmax-xmin)*(ymax-ymin))))).T

    if direction == 'b':    
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        src_pixels = np.matmul(H_inv, dst_pixels.T).T
        # u = np.reshape(src_pixels[:, 0] / src_pixels[:, 2], (ymax-ymin, xmax-xmin)).astype(int)
        # v = np.reshape(src_pixels[:, 1] / src_pixels[:, 2], (ymax-ymin, xmax-xmin)).astype(int)
        u = np.reshape(src_pixels[:, 0] / src_pixels[:, 2], (ymax-ymin, xmax-xmin))
        v = np.reshape(src_pixels[:, 1] / src_pixels[:, 2], (ymax-ymin, xmax-xmin))
        # print('u: ', u)
        # print('u_shape: ', u.shape)
        # print('v: ', v)
        # print('v_shape: ', v.shape)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = np.logical_and.reduce((u >= 0, v >= 0, u < w_src-1, v < h_src-1))
        # print('mask: ', mask)
        # print('mask_shape: ', mask.shape)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        u = u[mask]
        v = v[mask]
        # print('u: ', u)
        # print('u_size: ',u.shape)
        # print('v: ', v)
        # print('v_size: ', v.shape)

        mVxi = u.astype(int)
        mVyi = v.astype(int)
        dX = (u - mVxi).reshape((-1,1)) # delta X
        dY = (v - mVyi).reshape((-1,1)) # delta Y
        # print('dX: ', dX)
        # print('dX_type: ',dX.shape)
        # print('dY: ', dY)
        # print('dY_Type: ', dY.shape)

        p = np.zeros((h_src, w_src, ch))
        # print('p: ',p)
        # print('p_shape: ',p.shape)
        
        p[mVyi, mVxi, :] += (1-dY)*(1-dX)*src[mVyi, mVxi, :]
        p[mVyi, mVxi, :] += (dY)*(1-dX)*src[mVyi+1, mVxi, :]
        p[mVyi, mVxi, :] += (1-dY)*(dX)*src[mVyi, mVxi+1, :]
        p[mVyi, mVxi, :] += (dY)*(dX)*src[mVyi+1, mVxi+1, :]


        # TODO: 6. assign to destination image with proper masking
        dst[ymin:ymax,xmin:xmax][mask] = p[mVyi,mVxi]


        # pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        src_pixels = np.matmul(H, dst_pixels.T).T
        u = np.reshape(src_pixels[:, 0] / src_pixels[:, 2], (ymax-ymin, xmax-xmin)).astype(int)
        v = np.reshape(src_pixels[:, 1] / src_pixels[:, 2], (ymax-ymin, xmax-xmin)).astype(int)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        # mask = np.zeros((h_dst, w_dst), dtype=bool)
        mask = np.logical_and.reduce((u >= 0, v >= 0, u < w_dst, v < h_dst))

        # TODO: 5.filter the valid coordinates using previous obtained mask
        u = u[mask]
        v = v[mask]

        # TODO: 6. assign to destination image using advanced array indicing
        dst[ymin+v, xmin+u] = src[mask]

        # pass

    return dst