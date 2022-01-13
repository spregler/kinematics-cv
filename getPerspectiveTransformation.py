import numpy as np
import cv2 as cv
import scipy.signal as sig

def getPerspectiveTransformation(src, dst):
    """ 
    Calculates 3x3 Homography Matrix to transform the src points to the dst points
    *
    * Coefficients are calculated by solving linear system:
    *             P                          H  = 0
    * / -x1 -y1 -1 0 0 0 x1x1' y1x1' x1' \ /h11\ / 0 \
    * | 0 0 0 -x1 -y1 -1 x1y1' y1y1' y1' | |h12| | 0 |
    * | -x1 -y1 -1 0 0 0 x1x1' y1x1' x1' | |h13| | 0 |
    * | 0 0 0 -x1 -y1 -1 x1y1' y1y1' y1' |.|h21|=| 0 |,
    * | -x1 -y1 -1 0 0 0 x1x1' y1x1' x1' | |h22| | 0 |
    * | 0 0 0 -x1 -y1 -1 x1y1' y1y1' y1' | |h23| | 0 |
    * | -x1 -y1 -1 0 0 0 x1x1' y1x1' x1' | |h31| | 0 |
    * | 0 0 0 -x1 -y1 -1 x1y1' y1y1' y1' | |h32| | 0 |
    * \ 0   0   0   0   0   0   0   0  1 / \(1)/ \ 1 /
    *
    */
    """

    if (src.shape != (4,2) or dst.shape != (4, 2) ):
        raise ValueError("There must be four source and destination points")

    p = np.zeros((9, 9))
    px = np.zeros((4, 9))
    py = np.zeros((4, 9))
    b = np.zeros(9)

    for i in range(4):
        px[i][0] = -src[i][0]
        px[i][1] = -src[i][1]
        px[i][2] = -1
        px[i][3] = p[i][4] = p[i][5] = 0
        px[i][6] = src[i][0] * dst[i][0]
        px[i][7] = src[i][1] * dst[i][0]
        px[i][8] = dst[i][0]
        py[i][0] = py[i][1] = py[i][2] = 0
        py[i][3] = -src[i][0]
        py[i][4] = -src[i][1]
        py[i][5] = -1
        py[i][6] = src[i][0] * dst[i][1]
        py[i][7] = src[i][1] * dst[i][1]
        py[i][8] = dst[i][1]

    p[0] = px[0]
    p[1] = py[0]
    p[2] = px[1]
    p[3] = py[1]
    p[4] = px[2]
    p[5] = py[2]
    p[6] = px[3]
    p[7] = py[3]
    p[8] = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    b[8] = 1

    h = np.linalg.solve(p, b)
    h.resize((9,), refcheck=False)
    return h.reshape(3, 3)


def harris(img, threshold=0.55):
    # Sobel x-axis kernel
    SOBEL_X = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int32")

    # Sobel y-axis kernel
    SOBEL_Y = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int32")

    # Gaussian kernel
    GAUSS = np.array((
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]), dtype="float32")

    img_cpy = img.copy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5,5), 0)
    # Convolve w/ Sobel operator on both axis
    dx = sig.convolve2d(img_gray, SOBEL_X)
    dy = sig.convolve2d(img_gray, SOBEL_Y)
    # Square of derivatives
    dx2 = dx * dx
    dy2 = dy * dy
    dxdy = dx * dy
    # Convolve w/ Gaussian kernel
    dx2_g = sig.convolve2d(dx2, GAUSS)
    dy2_g = sig.convolve2d(dy2, GAUSS)
    dxdy_g = sig.convolve2d(dxdy, GAUSS)
    # Calculate score: R = det(M) - k*trace^2(M)
    R = dx2_g*dy2_g - np.square(dxdy_g) - 0.05*np.square(dx2_g + dy2_g)
    # Normalize
    cv.normalize(R, R, 0, 1, cv.NORM_MINMAX)
    # Nonmax supression
    sup = np.where(R >= threshold) # sup[0] is x-axis and sup[1] is y-axis

    for px in zip(*sup[::-1]):
        cv.circle(img_cpy, px, 2, (0, 0, 255), -1)

    return img_cpy, sup



if __name__ == "__main__":
    # Create a transform to change table coordinates in inches to projector coordinates
    cap = cv.VideoCapture(0)

    while True:
        isTrue, frame = cap.read()
        # print(frame.shape) Frame size is 480, 640

        # cv.circle(frame, (232, 250), 2, (0, 0, 255), -1)
        # cv.circle(frame, (393, 250), 2, (0, 0, 255), -1)
        # cv.circle(frame, (213, 442), 2, (0, 0, 255), -1)
        # cv.circle(frame, (431, 437), 2, (0, 0, 255), -1)

        corners, sup = harris(frame)
        cv.imshow("Result", corners)

        # pts1 = np.float32([[232, 250], [393, 250], [213, 442], [431, 437]])
        # pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
        # matrix = getPerspectiveTransformation(pts1, pts2)

        # result = cv.warpPerspective(frame, matrix, (400, 600) )

        # cv.imshow("Frame", frame)
        # cv.imshow("result", result)

    # img = cv.imread("sample.PNG")
    # result, interestPoints = harris(img)

    
    # pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])

    # cv.imshow("result", result)
    # cv.waitKey(0)

        
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()



    