# Udacity SDCND Project 4: Advanced Lane Lines
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
With Project Nr. 4 students of the Self-Driving Car Nanodegree are challenged to apply more advanced robotics techniques to find lane lines, calculate the lanes curvature as well as the cars offset to the center of the lane. This writeup will give a summary of the approach and reflect my experience.

## Content
This writeup will be sturctured according to the goals of this project:
1. Camera calibration and undistortion
2. Computer vision techniques to extract the lane lines out of the image/video
3. Perspective transform for birds-eye view
4. Lane detection
5. Calculation of lane curvature and offset to the lane center
6. Transformation and visualization of all gathered information to original view

## 1. Camera calibration and undistortion
Every camera has a distorion due to its lenses. Thus, objects might appear in the wrong size or shape depending on where they are placed in the image. To avoid this, the camera should be properly calibrated and images should get undistored.
For this project I set up a file called cam_calibration.py with a class which is responsible for calibration, undistortion, warping and unwarping images (more on that later).
To calibrate the camera, pictures of a chess board have been taken and provided by Udacity. Built-in function of openCV makes it easy to calibrate (cam_calibration.py, line 19). Results of the calibration are the distortion matrix and vectors for positioning of the camera in the real world. Those parameters and vectors can be used to undistort the picture (cam_calibration.py, line 63)
[alt text](https://github.com/jxkxb/carnd_P4_advanced_lane_lines/blob/master/writeup/chess_original_vs_undistored.png "Chessboard")
[alt text](https://github.com/jxkxb/carnd_P4_advanced_lane_lines/blob/master/writeup/pic_original_vs_undistorted.png "Picture")
