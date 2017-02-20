# Udacity SDCND Project 4: Advanced Lane Lines
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

With Project Nr. 4 students of the Self-Driving Car Nanodegree are challenged to apply more advanced robotics techniques to find lane lines, calculate the lanes curvature as well as the cars offset to the center of the lane. This writeup will give a summary of the approach and reflect my experience.

## Content
This writeup will be sturctured according to the goals of this project:

1. Camera calibration and undistortion
2. Computer vision techniques to extract the lane lines
3. Perspective transform for birds-eye view
4. Lane detection
5. Calculation of lane curvature and offset to the lane center
6. Transformation and visualization of all gathered information to original view

## 1. Camera calibration and undistortion
Every camera has a distorion due to its lenses. Thus, objects might appear in the wrong size or shape depending on where they are placed in the image. To avoid this, the camera should be properly calibrated and images should get undistored.
For this project I set up a file called cam_calibration.py with a class which is responsible for calibration, undistortion, warping and unwarping images (more on that later).
To calibrate the camera, pictures of a chess board have been taken and provided by Udacity. Built-in function of openCV makes it easy to calibrate (cam_calibration.py, line 19). Results of the calibration are the distortion matrix and vectors for positioning of the camera in the real world. Those parameters and vectors can be used to undistort the picture (cam_calibration.py, line 63)
#### Chessboard example
![alt text](https://github.com/jxkxb/carnd_P4_advanced_lane_lines/blob/master/writeup/chess_original_vs_undistored.png "Chessboard")
#### Picture example
![alt text](https://github.com/jxkxb/carnd_P4_advanced_lane_lines/blob/master/writeup/pic_original_vs_undistorted.png "Picture")

## 2. Computer vision techniques to extract the lane lines
The approach to extrac the lane lines was to apply color space transformation and gradient caluclation (find_lines.py, line 10 - 93). The best practice was to use the seperated color channels for red and green (from RGB color space), the saturation channel (from HLS color space) and the calculated magnitude of sobel x and sobel y (find pipeline in find_lines.py, line 134).
To all these channels a threshold was applied to get a binary mask for each of them. At the end, all of those binary mask where added together.
#### All channels separated
![alt text](https://github.com/jxkxb/carnd_P4_advanced_lane_lines/blob/master/writeup/chess_original_vs_undistored.png "All Channels")
#### Result after adding R, G, S and Sobel Magnitude
![alt text](https://github.com/jxkxb/carnd_P4_advanced_lane_lines/blob/master/writeup/pic_result_extracted_lines.png "Result")

## 3. Perspective Transform for Birds-Eye View
With the assumption that the road is flat the image was transformed in its perspective to have a birds-eye view. This birds-eye view makes it much easier to identify the 'hot pixels' belonging to the lane and caluclating lane curvature and offset to the center of the line.
The transformation is a function of the class CamCalibration (cam_calibration.py, line 84). It is calibrated to the provides example image 'straigt_lines2.jpg'.
#### Birds-Eye View Transformation
![alt text](https://github.com/jxkxb/carnd_P4_advanced_lane_lines/blob/master/writeup/pic_original_vs_warped.png "Birds-Eye View")

## 4. Lane Detection
Since the algorithm needs a starting point for each line, an vertical accumulation of all white pixels of the lower half of the picture was applied. The result is comparable with an histogram with the same number ob bins than datapoints (length of x-axis). The peak in the left and the right half are the starting points for the algorithm to search for white pixel which belong to the line (find_lines.py, line 134).
From here the picture got vertically devided in 9 windows with a width of 200 px. each white pixel inside this window was added to a list of lane pixels. The algorithm loops through all the windows (buttom to top). If there are more than a minimum of pixels (10) inside one window, the center of the new window gets adjusted to the mean x value of the current window. That ensured that curvy lines were detected properly too.
A polynomial fit (squared) was applied to all lane pixels found as continuous representation of the lane.
#### Lane Pixel Search and Polinominal Fit
![alt text](https://github.com/jxkxb/carnd_P4_advanced_lane_lines/blob/master/writeup/pic_lane_find.png "Found Lanes")

