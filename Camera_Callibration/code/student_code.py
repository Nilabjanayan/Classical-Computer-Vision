import numpy as np


def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
    M = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],
                    [0.6750, 0.3152, 0.1136, 0.0480],
                    [0.1020, 0.1725, 0.7244, 0.9932]])

    ###########################################################################
    # TODO: YOUR PROJECTION MATRIX CALCULATION CODE HERE
    n=np.shape(points_2d)[0]
    #print(n)
    A=np.zeros((2*n,11))
    c = 0
    for i in range(0,2*n,2):
        
        A[i][:4] = np.append(points_3d[c], [1])  
        A[i][-3:] = -points_2d[c][0]*points_3d[c]
        
        A[i+1][4:8] = np.append(points_3d[c], [1])  
        A[i+1][-3:] = -points_2d[c][1]*points_3d[c]
        
        c += 1
    u = []

    for i in points_2d:
        u.append(i[0])
        u.append(i[1])
        
    u = np.array(u).reshape(len(u),1)
    
    M=np.linalg.lstsq(A,u)
    M=np.append(M[0],[1.]).reshape(3,4)
    M=M/np.linalg.norm(M)
        
    
    ###########################################################################

#     raise NotImplementedError('`calculate_projection_matrix` function in ' +
#         '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.
    cc = np.asarray([1, 1, 1])

    ###########################################################################
    # TODO: YOUR CAMERA CENTER CALCULATION CODE HERE
    Q = M[:3,:3]
#     for i in range(3):
#         for j in range(3):
#             Q[i][j]=M[i][j]
    m4=M[:,3]
    inv_Q=np.linalg.inv(Q)
    cc=np.array(-inv_Q@m4).reshape(1,3)
    
    ###########################################################################

#     raise NotImplementedError('`calculate_camera_center` function in ' +
#         '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return cc






def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    # Placeholder fundamental matrix
    F = np.asarray([[0, 0, -0.0004],
                    [0, 0, 0.0032],
                    [0, -0.0044, 0.1034]])

    ###########################################################################
    # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
    
    mean_points_a = np.mean(points_a,axis=0)
    u_bar=mean_points_a[0]
#     print(u_bar)
    v_bar=mean_points_a[1]
#     print(v_bar)
    
    
    new_points_a =points_a-mean_points_a
    sum_of_square=np.sum(new_points_a**2,axis=1)
    mean=np.mean(sum_of_square)
    s=np.sqrt(2/mean)
    
    
    mean_points_b = np.mean(points_b,axis=0)
    u1_bar=mean_points_b[0]
    
    v1_bar=mean_points_b[1]
    
    
    new_points_b = points_b-mean_points_b
    sum_of_square1 = np.sum(new_points_b**2,axis=1)
    mean1 = np.mean(sum_of_square1)
    s1 = np.sqrt(2/mean1)
    
    #constrat Ta and Tb
    K1 = np.array([[s,0,0],[0,s,0],[0,0,1]])
#     print(K1.shape)
    t1=np.array([[1,0,-u_bar],[0,1,-v_bar],[0,0,1]])
#     print(t1.shape)
    Ta = np.matmul(K1,t1)
    
    
    K2 = np.array([[s1,0,0],[0,s1,0],[0,0,1]])
    t2 = np.array([[1,0,-u1_bar],[0,1,-v1_bar],[0,0,1]])
    Tb = np.matmul(K2,t2)
    
    points_a1 = np.column_stack((points_a,[1]*points_a.shape[0]))
    
    points_b1 = np.column_stack((points_b,[1]*points_b.shape[0]))
    
    points_a_new = np.matmul(Ta , points_a1.T)
    
    points_b_new = np.matmul(Tb , points_b1.T)
    
    point1 = np.delete(points_a_new.T,2,1)
    
    point2 = np.delete(points_b_new.T,2,1)
    
    points_a = point1
    
    points_b = point2
    
    
    
    #############################################
    a = np.column_stack((points_a,[1]*points_a.shape[0]))
    
    b = np.column_stack((points_b,[1]*points_b.shape[0]))
    
    a = np.tile(a,3)
    
    b = b.repeat(3,axis=1)
    
    A=np.multiply(a,b)
    
    U,S,V=np.linalg.svd(A)
    
    F=V[-1].reshape(3,3)
    
    F = F/np.linalg.norm(F)
    
        
    U1,S1,V1 = np.linalg.svd(F)
    S1[2] = 0
    F =  U1 @ np.diagflat(S1) @ V1
    F = Tb.T @ F @ Ta

    
    ###########################################################################

#     raise NotImplementedError('`estimate_fundamental_matrix` function in ' +
#         '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return F

def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values
    best_F = estimate_fundamental_matrix(matches_a[:10, :], matches_b[:10, :])
    inliers_a = matches_a[:100, :]
    inliers_b = matches_b[:100, :]

    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    iterator = 10000
    
    threshold = 0.001
    
    best_F_matrix = np.zeros((3, 3))
    
    max_inlier = 0
    
    a = np.column_stack((matches_a, [1]*matches_a.shape[0]))

    b = np.column_stack((matches_b, [1]*matches_b.shape[0]))

    a = np.tile(a, 3)

    b = b.repeat(3, axis=1)

    A = np.multiply(a , b)
    
    for i in range(iterator):
        random_index= np.random.randint(matches_a.shape[0], size=8)
        
        F = estimate_fundamental_matrix(matches_a[random_index, :], matches_b[random_index, :])
        
        error = np.abs(np.matmul(A, F.reshape((-1))))
        
        current_inlier = np.sum(error <= threshold)
        
        if current_inlier > max_inlier:
            best_F_matrix = F.copy()
            max_inlier = current_inlier

    error = np.abs(np.matmul(A,best_F_matrix.reshape((-1))))
    index = np.argsort(error)
    # print(best_F_matrix)
    # print(np.sum(err <= threshold), "/", err.shape[0])
    return best_F_matrix, matches_a[index[:max_inlier]], matches_b[index[:max_inlier]]
    
#     ###########################################################################

#     raise NotImplementedError('`ransac_fundamental_matrix` function in ' +
#         '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

#     return best_F, inliers_a, inliers_b