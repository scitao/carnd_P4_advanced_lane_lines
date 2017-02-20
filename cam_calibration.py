import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


class CamCalibration:
    path_cal_pics = str()
    ret = False
    mtx = []
    dist = []
    rvecs = 0
    tvecs = 0

    def __init__(self):
        print('# Calibration Class initialized ...')

    def calibrate(self, path_cal_pics, nx, ny, plot):
        # Load images for calibration and find chessboard corners
        self.path_cal_pics = path_cal_pics
        cam_pics = os.listdir(path_cal_pics)
        cam_pics.remove('.DS_Store')
        objpoints = []
        imgpoints = []

        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        if plot:
            fig = plt.figure()
            fig.canvas.set_window_title('Calibration Pictures')

        for i in range(len(cam_pics)):
            # Load camera picture
            cam_img = mpimg.imread(path_cal_pics + cam_pics[i])
            # Convert to gray scale
            gray = cv2.cvtColor(cam_img, cv2.COLOR_RGB2GRAY)
            # Find the corners (ret == True if corners are found)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny),
                                                     None)
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)

                if plot:
                    plt.subplot(4, 5, i + 1)
                    plt.title(str(i + 1))
                    plt.xticks([]), plt.yticks([])
                    cv2.drawChessboardCorners(cam_img, (nx, ny), corners, ret)
                    plt.imshow(cam_img)

        if plot:
            plt.show(fig)

        # Calibrate the camera
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                None, None)
        print('# Successful with: {} of {} pictures'.format(len(imgpoints),
                                                            len(cam_pics)))

    def undistort(self, img, plot):
        """
        Corrects distortion of camera
        :param img: Distorted image
        :param plot: Plots original and undistorted picture
        :return: Undistorted picture
        """
        if not self.ret:
            print('# Calibrate camera first!')
            return 0
        else:
            dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
            if plot:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.canvas.set_window_title('Distorted and Undistorted Image')
                ax1.imshow(img)
                ax2.imshow(dst)
                plt.show(fig)
            return dst

    @staticmethod
    def warp_img(img):
        """
        Perspective transform calibrated to undistorted 'straight_lines2.jpg'
        :param img: undistorted image
        :return: warped image
        """
        img_shape = (img.shape[1], img.shape[0])
        offset_x = int(img.shape[1] * 0.2)
        src = np.float32([[272, 680],
                          [1043, 680],
                          [688, 450],
                          [596, 450]])
        dst = np.float32([[offset_x, img_shape[1]],
                          [img_shape[0]-offset_x, img_shape[1]],
                          [img_shape[0]-offset_x, 0],
                          [offset_x, 0]])

        m = cv2.getPerspectiveTransform(src, dst)

        return cv2.warpPerspective(img, m, img_shape, flags=cv2.INTER_LINEAR)

    @staticmethod
    def unwarp_img(warp):
        img_shape = (warp.shape[1], warp.shape[0])
        offset_x = int(warp.shape[1] * 0.2)
        dst = np.float32([[272, 680],
                          [1043, 680],
                          [688, 450],
                          [596, 450]])
        src = np.float32([[offset_x, img_shape[1]],
                          [img_shape[0] - offset_x, img_shape[1]],
                          [img_shape[0] - offset_x, 0],
                          [offset_x, 0]])

        m = cv2.getPerspectiveTransform(src, dst)

        return cv2.warpPerspective(warp, m, img_shape, flags=cv2.INTER_LINEAR)
