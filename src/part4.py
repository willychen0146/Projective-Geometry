import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)
np.random.seed(999)

'''def alpha_blend(im1, im2, H): # fail
    """Function to perform alpha blending of two images."""
    h_max = max([x.shape[0] for x in imgs]) # 矮的疊高的上
    w_max = sum([x.shape[1] for x in imgs]) # 寬度是所有圖寬之總和

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8) # 第一張輸入圖像的顏色通道數 (黑色畫布，用來存儲拼接後的圖像)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0] # values of imgs[0] to a sub-region of the dst array. imgs[0] is being assigned to the top-left corner of dst, with the same height and width as imgs[0]

    _, mask = cv2.threshold(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    mask = cv2.merge((mask, mask, mask))

    # find the overlapping region between im1 and im2
    im2_top_left = (0, 0) # im2的左上角在dst中的位置
    im2_bottom_right = (im2.shape[1], im2.shape[0]) # im2的右下角在dst中的位置
    im1_top_left_warped = solve_homography(H, np.array([0, 0, 1]))[:2].astype(int) # im1左上角在im2中的位置
    im1_bottom_right_warped = solve_homography(H, np.array([im1.shape[1], im1.shape[0], 1]))[:2].astype(int) # im1右下角在im2中的位置
    im1_top_left = (im2_top_left[0] + im1_top_left_warped[0], im2_top_left[1] + im1_top_left_warped[1]) # im1的左上角在dst中的位置
    im1_bottom_right = (im2_top_left[0] + im1_bottom_right_warped[0], im2_top_left[1] + im1_bottom_right_warped[1]) # im1的右下角在dst中的位置


    # select the overlapping regions
    overlap_im1 = dst[im1_top_left[1]:im1_bottom_right[1], im1_top_left[0]:im1_bottom_right[0]]
    overlap_im2 = im2[im2_top_left[1]:im2_bottom_right[1], im2_top_left[0]:im2_bottom_right[0]]    

    im1_warped = warping(overlap_im1, overlap_im2, H, 0, h_max, 0, w_max, direction='b')
    blended = cv2.addWeighted(im1_warped, 0.95, im1, 0.95, 0)
    blended = cv2.bitwise_and(blended, mask)
    return blended'''

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs]) # 矮的疊高的上
    w_max = sum([x.shape[1] for x in imgs]) # 寬度是所有圖寬之總和

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8) # 第一張輸入圖像的顏色通道數 (黑色畫布，用來存儲拼接後的圖像)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0] # values of imgs[0] to a sub-region of the dst array. imgs[0] is being assigned to the top-left corner of dst, with the same height and width as imgs[0]
    last_best_H = np.eye(3) # initializing a 3x3 identity matrix, used to store the best homography matrix that was found in previous iterations of the loop
    out = None


    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)): # 迭代所有要拼接的圖像
        im1 = imgs[idx + 1] # im1是下一張圖像
        im2 = imgs[idx] # im2是當前圖像

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create() # Initiate ORB detector, 檢測當前和前一張圖像中的特徵點
        # sift = cv2.SIFT_create() # if use SIFT

        kp1, des1 = orb.detectAndCompute(im1, None) # current image
        kp2, des2 = orb.detectAndCompute(im2, None) # previous image

        # des1 = cv2.convertScaleAbs(des1, alpha=(255.0/float(des1.max()))) # if use SIFT, convert data type
        # des2 = cv2.convertScaleAbs(des2, alpha=(255.0/float(des2.max()))) # if use SIFT, convert data type

        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True) # 暴力解 matched points
        matches = bf.match(des1, des2) 

        # Sorting to make that top matches at the front
        choose_n_matches = 100 # 只考慮前n個最匹配的點
        sorted_matches = sorted(matches, key=lambda x: x.distance) # 將匹配點按距離遞增排序，只保留前n_best_matches個匹配點
        matches = []
        for match in sorted_matches:
            if len(matches) >= choose_n_matches:
                break
            matches.append(match)

        q_idx, t_idx = [], []
        for match in matches:
            q_idx.append(match.queryIdx)
            t_idx.append(match.trainIdx)

        # 透過索引以找到對應的關鍵點 (src_pts和dst_pts分別代表當前圖像和前一張圖像中的匹配點，將關鍵點的位置保存到src_pts和dst_pts)
        src_pts = np.array([kp1[idx].pt for idx in q_idx])
        dst_pts = np.array([kp2[idx].pt for idx in t_idx])

        # TODO: 2. apply RANSAC to choose best H
        '''RANSAC parameter'''
        inlier_threshold = 0.37 # RANSAC
        iterations = 10000 # sampling iterations for RANSAC
        keypoints_for_H = 15 # number of keypoints to sample to estimate H
        best_homography = None
        '''RANSAC parameter'''

        # Begin RANSAC Iterating
        max_inliers = 0
        for i in range(iterations):
            # Randomly selects a set of num_kps_for_H keypoints from the source (src_pts) and destination (dst_pts) images to estimate the homography matrix H
            sampled_indices = np.random.permutation(len(src_pts))[:keypoints_for_H]
            p1, p2 = src_pts[sampled_indices], dst_pts[sampled_indices]
            H = solve_homography(p1, p2) # Uses H to compute the predicted coordinates (pred) of the keypoints in the source image
            
            # Compute the predicted coordinates using the estimated homography matrix
            U = np.concatenate((src_pts.T, np.ones((1, src_pts.shape[0]))), axis=0)
            pred = np.dot(H, U)
            pred = (pred / pred[-1]).T[:, :2]
            # pred = cv2.perspectiveTransform(np.array([src_pts]), H).squeeze()
            
            # Compute the Euclidean distance between the predicted coordinates and the destination points
            distances = np.linalg.norm(pred - dst_pts, axis=1)
            
            # Count the number of inliers based on a distance threshold
            inliers = np.sum(distances < inlier_threshold) # Number of inliers is counted based on threshold
        
            # Update the best homography matrix and the maximum number of inliers if necessary
            # Best homography matrix and the maximum number of inliers are updated if the current inlier ratio is greater than the previous maximum
            inlier_ratio = inliers / keypoints_for_H
            if inlier_ratio > max_inliers: 
                best_homography = H.copy()
                max_inliers = inlier_ratio

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_homography)

        # TODO: 4. apply warping
        # Warping to the current image using the overall homography matrix, which effectively aligns the current image with the reference image
        out = warping(im1, dst, last_best_H, 0, h_max, 0, w_max, direction='b')

        '''
        # TODO: 5. automatic weight adjustment
        a = 0.3 # starting value of alpha
        b = 0.9 # ending value of alpha
        step = 0.1 # step size
        best_out = None
        min_error = float('inf')
        out_copy = out.copy() # create a copy of out
        for alpha in np.arange(a, b, step):
            # blend the two images with alpha value
            temp_out = cv2.addWeighted(out_copy, alpha, dst, 1 - alpha, 0) # use the copy instead of original out

            # calculate the error
            border = int(im1.shape[1] * alpha) # estimate the border of the overlap region
            out_border = temp_out[:, :border, :]
            im1_border = im1[:, :border, :]
            if out_border.shape != im1_border.shape:
                out_border = np.pad(out_border, ((0, 0), (0, im1_border.shape[1] - out_border.shape[1]), (0, 0)), 'constant')
            error = np.sum((out_border - im1_border) ** 2)

            # update the best output and the minimum error
            if error < min_error:
                best_out = temp_out
                min_error = error

        out = best_out # update out with the best output'''

    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)