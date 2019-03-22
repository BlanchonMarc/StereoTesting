import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':

    imgL = cv2.imread(
        "/Users/marc/Documents/Dataset/AquisitionImagesFinal/Intensity-Polar/image_00001.png")
    imgL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)

    imgR = cv2.imread(
        "/Users/marc/Documents/Dataset/AquisitionImagesFinal/Kinect/image_00001.png")
    imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)

    imgPIL = Image.fromarray(imgL)
    imgL = np.array(imgPIL.resize(
        (imgR.shape[1], imgR.shape[0]), Image.ANTIALIAS))
    print(imgL.shape)
    print(imgR.shape)
    plt.imshow(imgL)
    plt.show()
    plt.imshow(imgR)
    plt.show()
    stereo = cv2.StereoSGBM_create(
        numDisparities=32, blockSize=16, uniquenessRatio=1)
    disparity = stereo.compute(imgL, imgR)
    plt.imshow(disparity, 'gray')
    plt.show()

    leftCameraMatrix = np.array(
        [[1144.841, 0, 327.035], [0, 1147.317, 165.249], [0, 0, 1]])
    leftDistortionCoefficients = np.array([0.1091, -0.1214, -0.0096, 0.0041])

    rightCameraMatrix = np.array([[1604.333, 0, 652.697], [
                                 0, 1607.913, 179.403], [0, 0, 1]])
    rightDistortionCoefficients = np.array([-0.1834, 0.2543, -0.0073, 0.0007])

    TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30,
                            0.001)
    OPTIMIZE_ALPHA = 0.250

    rot1 = np.array([[0.9998407,  0.0132871, -0.0119166], [-0.0111050,
                                                           0.9858181,  0.1674495], [0.0139725, -0.1672904,  0.9858086]])
    trans1 = np.array([-0.04573187, 0.1099181, -0.05943449])

    rot2 = np.array([[0.9989325, -0.0037325,  0.0460420], [0.0124329,
                                                           0.9816737, -0.1901633], [-0.0444885,  0.1905327,  0.9806722]])

    trans2 = np.array([0.13645018, -0.09783982, 0.08449779])

    rot3 = np.array([[0.9999570,  0.0006858, -0.0092435], [-0.0005168,
                                                           0.9998329,  0.0182760], [0.0092544, -0.0182705,  0.9997903]])

    trans3 = np.array([0.05267882, -0.00432335, 0.01241419])

    rotationMatrix = rot1 * rot2 * rot3
    rotationMatrix = rot1
    translationVector = trans1 + trans2 + trans3
    translationVector = trans1
    imageSize = imgL.shape

    (leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
        leftCameraMatrix, leftDistortionCoefficients,
        rightCameraMatrix, rightDistortionCoefficients,
        imageSize, rotationMatrix, translationVector,
        None, None, None, None, None,
        cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        leftCameraMatrix, leftDistortionCoefficients, leftRectification, leftProjection, imageSize, cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        rightCameraMatrix, rightDistortionCoefficients, rightRectification, rightProjection, imageSize, cv2.CV_32FC1)

    stereoMatcher = cv2.StereoBM_create()
    REMAP_INTERPOLATION = cv2.INTER_LINEAR
    fixedLeft = cv2.remap(imgL, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(imgR, rightMapX, rightMapY, REMAP_INTERPOLATION)

    depth = stereoMatcher.compute(fixedLeft, fixedRight)

    DEPTH_VISUALIZATION_SCALE = 2048
    while (True):
        cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    stereo = cv2.StereoBM_create()
    stereoMatcher.setMinDisparity(4)
    stereoMatcher.setNumDisparities(128)
    stereoMatcher.setBlockSize(21)
    stereoMatcher.setROI1(leftROI)
    stereoMatcher.setROI2(rightROI)
    stereoMatcher.setSpeckleRange(16)
    stereoMatcher.setSpeckleWindowSize(45)

    fixedLeft = cv2.remap(imgL, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(imgR, rightMapX, rightMapY, REMAP_INTERPOLATION)

    depth = stereoMatcher.compute(fixedLeft, fixedRight)
    DEPTH_VISUALIZATION_SCALE = 2048
    while(True):
        cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
