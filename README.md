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
