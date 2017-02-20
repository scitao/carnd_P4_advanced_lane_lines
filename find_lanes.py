import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import pickle
from cam_calibration import CamCalibration


def cvt_grayscale(img):
    """
    Convert image from RGB to gray scale
    :param img: Image in RGB color space
    :return: Image in gray scale
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def cvt_hls(img):
    """
    Convert image from RGB to HLS color space
    :param img: Image in RGB color space
    :return: Image in HLS color space
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def get_rgb_single_ch(img, rgb):
    """
    Get one single channel from image in RGB color space
    :param img: Image in RGB color space
    :param rgb: Channel information
    :return: Image reduced to single channel information
    """
    channel = img
    if rgb == 'R':
        channel = channel[:, :, 0]
    elif rgb == 'G':
        channel = channel[:, :, 1]
    elif rgb == 'B':
        channel = channel[:, :, 2]
    else:
        print('Error:', str(rgb), 'not found!')

    return channel


def get_hls_single_ch(img, hls):
    """
    Get one single channel from image in HLS color space
    :param img: Image in RGB color space
    :param hls: Channel information
    :return: Image reduced to single channel information
    """
    channel = cvt_hls(img)
    if hls == 'H':
        channel = channel[:, :, 0]
    elif hls == 'L':
        channel = channel[:, :, 1]
    elif hls == 'S':
        channel = channel[:, :, 2]
    else:
        print('Error:', str(hls), 'not found!')

    return channel


def get_thresh_bin(img, thresh):
    """
    Converts single channel image to a binary image according threshold
    :param img:
    :param thresh:
    :return:
    """
    thresh_img = np.zeros_like(img)
    thresh_img[(img >= thresh[0] * np.max(img)) &
               (img <= thresh[1] * np.max(img))] = 1

    return thresh_img


def get_sobel_magnitude(img, kernel_size, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobel_xy = np.sqrt(sobel_x**2, sobel_y**2)
    sobel_xy = np.uint8(255*sobel_xy / np.max(sobel_xy))

    binary = np.zeros_like(sobel_xy)
    binary[(sobel_xy >= thresh[0]) & (sobel_xy <= thresh[1])] = 1

    return binary


def bitwise_add(img1, img2):
    """
    Adds positive pixels of two binary images
    :param img1: Image one (binary)
    :param img2: Image two (binary)
    :return: Addition of binary images
    """
    return (img1 == 1) | (img2 == 1)


def lines_bin(img):
    """
    Pipeline to convert RGB image to binary image for line finding
    :param img: Image in RGB color space
    :return: binary image
    """
    # RGB and HLS channel picks
    red = get_rgb_single_ch(img, 'R')
    gre = get_rgb_single_ch(img, 'G')
    sat = get_hls_single_ch(img, 'S')

    red_thresh_bin = get_thresh_bin(red, thresh=(.9, 1))
    gre_thresh_bin = get_thresh_bin(gre, thresh=(.9, 1))
    sat_thresh_bin = get_thresh_bin(sat, thresh=(.6, 1))

    # Addition of picked channels
    add_red_green = bitwise_add(red_thresh_bin, gre_thresh_bin)
    add_red_green_sat = bitwise_add(add_red_green, sat_thresh_bin)

    # Sobels
    magnitude = get_sobel_magnitude(img, 3, thresh=(30, 100))

    # Colors and sobel magnitude
    binary_img = bitwise_add(add_red_green_sat, magnitude)

    return binary_img


def find_lanes(bin_warp, nwindows=9, window_width=200, min_pixels=10, plot=False):
    """
    Finds the lanes in a binary bird-eye (warped) image
    :param bin_warp: Binary warped image
    :return:
    """
    # Sum the positive pixels in the lower half of the image vertically
    histogram = np.sum(bin_warp[int(bin_warp.shape[0]/2):, :], axis=0)
    # Identify the peaks in left and right half
    peak_left = np.int(np.argmax(histogram[:len(histogram) // 2]))
    peak_right = np.int(np.argmax(histogram[len(histogram) // 2:]) +
                        len(histogram) // 2)

    # Window height and width
    window_height = bin_warp.shape[0] // nwindows
    margin = window_width // 2

    # Find indices where binary is not zero
    nonzero = bin_warp.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Find peaks in left and right half of histogram (starting point)
    mid_x_left = peak_left
    mid_x_right = peak_right

    # Initialize empty arrays to store indices for lane pixels
    inds_left = []
    inds_right = []

    # Initialize and output image: stack 3 times binary and multiply by 255
    img_out = np.dstack((bin_warp.astype(np.uint8),) * 3) * 255

    # Loop through each window to identify lane pixels
    for window in range(nwindows):
        # Set boarders for window
        win_y_low = bin_warp.shape[0] - (window + 1) * window_height
        win_y_high = bin_warp.shape[0] - window * window_height
        win_left_x_left = mid_x_left - margin
        win_left_x_right = mid_x_left + margin
        win_right_x_left = mid_x_right - margin
        win_right_x_right = mid_x_right + margin

        cv2.rectangle(img_out, (win_left_x_left, win_y_low),
                      (win_left_x_right, win_y_high), (50, 25*window, 0), 2)
        cv2.rectangle(img_out, (win_right_x_left, win_y_low),
                      (win_right_x_right, win_y_high), (50, 25*window, 0), 2)

        good_inds_left = ((nonzero_y >= win_y_low) &
                          (nonzero_y < win_y_high) &
                          (nonzero_x >= win_left_x_left) &
                          (nonzero_x < win_left_x_right)).nonzero()[0]
        good_inds_right = ((nonzero_y >= win_y_low) &
                           (nonzero_y < win_y_high) &
                           (nonzero_x >= win_right_x_left) &
                           (nonzero_x < win_right_x_right)).nonzero()[0]

        # Append good indices of current window to the indices list
        inds_left.append(good_inds_left)
        inds_right.append(good_inds_right)

        # If more pixels than minimum: slide middle of next window
        if len(good_inds_left) > min_pixels:
            mid_x_left = np.int(np.mean(nonzero_x[good_inds_left]))
        if len(good_inds_right) > min_pixels:
            mid_x_right = np.int(np.mean(nonzero_x[good_inds_right]))

    # Concatenate the arrays of indices
    inds_left = np.concatenate(inds_left)
    inds_right = np.concatenate(inds_right)

    # Extract left and right line pixel positions
    y_left = nonzero_y[inds_left]
    x_left = nonzero_x[inds_left]
    y_right = nonzero_y[inds_right]
    x_right = nonzero_x[inds_right]

    # Fit a second order polynomial to each
    fit_left = np.polyfit(y_left, x_left, 2)
    fit_right = np.polyfit(y_right, x_right, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, bin_warp.shape[0] - 1, bin_warp.shape[0])
    fit_lane_left = fit_left[0] * ploty ** 2 + fit_left[1] * \
                                               ploty + fit_left[2]
    fit_lane_right = fit_right[0] * ploty ** 2 + fit_right[1] * \
                                                 ploty + fit_right[2]

    # Prepare for plotting
    img_out[nonzero_y[inds_left], nonzero_x[inds_left]] = [255, 0, 0]
    img_out[nonzero_y[inds_right], nonzero_x[inds_right]] = [255, 0, 0]

    img_lane = np.zeros_like(img_out)
    lane_left = np.array([np.transpose(np.vstack([fit_lane_left, ploty]))])
    lane_right = np.array(
        [np.flipud(np.transpose(np.vstack([fit_lane_right, ploty])))])
    lane = np.hstack((lane_left, lane_right))

    cv2.fillPoly(img_lane, np.int_([lane]), (0, 255, 0))

    # Calculations for offset to lane center
    middle = 0.5*bin_warp.shape[1]
    x_l = np.mean(fit_lane_left[-20:-1])
    x_r = np.mean(fit_lane_right[-20:-1])
    off = middle - 0.5*(x_r - x_l) - x_l
    #print('Poly Left offset:', off)

    # Plot found lane pixels on binary warped image
    if plot:

        result = cv2.addWeighted(img_out, 1, img_lane, 0.3, 0)
        plt.imshow(result)
        plt.plot(fit_lane_left, ploty, color='yellow')
        plt.plot(fit_lane_right, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.plot([0.5*bin_warp.shape[1], 0.5*bin_warp.shape[1]],
                 [bin_warp.shape[0], bin_warp.shape[0] - 100], '--r')
        plt.show()



    # Transform to real world
    ym_per_pix = 3/80
    xm_per_pix = 3.7/760

    fit_left_rw = np.polyfit(y_left*ym_per_pix, x_left*xm_per_pix, 2)
    fit_right_rw = np.polyfit(y_right*ym_per_pix, x_right*xm_per_pix, 2)

    off_rw = off * xm_per_pix

    return img_lane, fit_left_rw, fit_right_rw, off_rw


def draw_line(img, img_lane):
    return cv2.addWeighted(img, 1, img_lane, 0.3, 0)


def background_for_text(img, height):
    img[:height, :, :] = img[:height, :, :] * 0.4

    return img


def text_on_image(img, rad_left, rad_right, off_m):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255,) * 3

    # Radius left lane
    cv2.putText(img, ('Left:'), (20, 40), font, 1.2, color, 2)
    cv2.putText(img, ('%.1f m' %rad_left), (160, 40), font, 1.2, color, 2)
    # Radius right lane
    cv2.putText(img, ('Right:'), (20, 80), font, 1.2, color, 2)
    cv2.putText(img, ('%.1f m' %rad_right), (160, 80), font, 1.2, color, 2)
    # Offset to lane center
    cv2.putText(img, ('Offset:'), (500, 40), font, 1.2, color, 2)
    cv2.putText(img, ('%.1f m' %off_m), (500, 80), font, 1.2, color, 2)
    return img

def measuring_curvature(img, fit_left, fit_right):
    ym_per_pix = 3 / 80
    xm_per_pix = 3.7 / 760

    eval_point = img.shape[0]
    rad_left = ((1 + (2*fit_left[0]*eval_point*ym_per_pix + fit_left[1])**2)**1.5) / np.absolute(2*fit_left[0])
    rad_right = ((1 + (2*fit_right[0]*eval_point*ym_per_pix + fit_right[1])**2)**1.5) / np.absolute(2*fit_right[0])
    #print('Left Radius:', rad_left, 'Right Radius', rad_right)

    return rad_left, rad_right


# Camera calibration
cal_path = './camera_cal/'
cam = CamCalibration()
cam.calibrate(cal_path, 9, 6, False)

# Processing area
test_imgs = os.listdir('./test_images')

image = mpimg.imread('./test_images/' + test_imgs[4])
dst = cam.undistort(image, False)

warp = cam.warp_img(dst)
test = lines_bin(warp)

warp_lane, fit_l, fit_r, off = find_lanes(test, 9, 200, 10, False)
lane = cam.unwarp_img(warp_lane)
r_left, r_right = measuring_curvature(warp_lane, fit_l, fit_r)

final = draw_line(dst, lane)
final = background_for_text(final, 100)
final = text_on_image(final, r_left, r_right, off)


plt.imshow(final)
plt.show()

from moviepy.editor import VideoFileClip
from IPython.display import HTML

vid = False
if vid:
    def pipeline(dist_img):
        dst = cam.undistort(dist_img, False)
        warp = cam.warp_img(dst)
        bin_warp = lines_bin(warp)
        lane_warp, fit_l, fit_r, off = find_lanes(bin_warp)
        r_left, r_right = measuring_curvature(warp_lane, fit_l, fit_r)
        lane = cam.unwarp_img(lane_warp)
        final = draw_line(dst, lane)
        final = background_for_text(final, 100)
        final = text_on_image(final, r_left, r_right, off)
        return final

    output = 'test.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    clip = clip1.fl_image(pipeline)
    clip.write_videofile(output, audio=False)
