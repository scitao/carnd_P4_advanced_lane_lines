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
7. Smoothing the lines and and plausibility check

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

## 5. Lane Curvature and Offset to Lane Center
Now, with the detected lines and the assumption that the camera is placed in the middle of the car, it was possible to calculate the lane curvature from the polinomial coeffcients and the offset of the detected lanes to the pictures middle (curvature: find_lanes.py, line 293; offset: find_lanes.py, line 234).
Of cause, the calculated curvature and offset have to be transformed to the normal-view and scaled to the real world. The coefficients for the linear transformation to the real world was done by comparing length in the picture (pixels) to standard values provided by the US Government (find_lanes.py, line 257 & 258).

## 6. Transformation and Visualiztion
To get to the final result, the detected lanes were unwarped (back to original-view) and plotted on the video. To visualize the curvature and offset to the center of the lane, the upper area of the video was darkened (like a sun visor in the car; find_lanes.py, line 272) and the values have been plotted in this area (find_lanes.py, line 278).
![alt text](https://github.com/jxkxb/carnd_P4_advanced_lane_lines/blob/master/writeup/pic_final_result.png "Final Result")

## 7. Smoothing and Plausibility checks
After the pipepline performed for most of the frames well there where still a few left where the left line was wobbling too much or even failed completely. That was especially the case where the color of the road changed. To improve the robustness of the lanes and avoid glitches two improvments were implemented in the end:
 1. Plausibility check (find_lanes.py, line 223):
  * Check of the sign of the quadratic argumnt of polyfit for the left and the right lane is the same
  * If not: skip frame
 2. Smoothing the lines by median over the last 10 frames (find_lanes.py, line 224 - 241):
  * Append the current np.polyfit to and array with the length of 10 and delete the first one
  * Draw the lines from the median of the 10 last frames

# Discussion
## Personal Experience
It was a tough challenge for me but considering my background I'm am happy and proud to submit my project. The most challenging thing was to implement all the functions and do the mathmatical transformations - not from the theoretical side but from the programming side. I've learned a lot again in Python and its libraries.

## Technical
My code is able to produce good results on the project video. There's almost no wobbling and the plotted green lane would keep the car safely on track.
I would love to and I will improve the code to make it more robust to other videos aswell. There are enough ideas I want to implement.
