## Advanced Lane Finding



**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Original Chessboard"
[image2]: ./test_images/test2.jpg "Original Road"

[image3]: ./output_images/undistorted_chessboard.jpg "Chessboard Undistorted"
[image4]: ./output_images/undistortion.jpg "Road Transformed"

[image5]: ./output_images/thresholded.png "Road Thresholded"

[image6]: ./output_images/warped.png "Road Warped"
[image7]: ./test_images/straight_lines1.jpg "Straight"
[image8]: ./output_images/straight_warped.png "Straight Warped"

[image9]: ./output_images/fit_polynomial.jpg "Fit Visual"
[image10]: ./output_images/test2.jpg "Output"
[video1]: ./output_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

Note: All of the related code are in `Line.py`. But that code does not produce resulting images as individual files in each stage of pipeline directly. There is a testing Jupyter Notebook named `Test With Outside functions.ipynb` that only uses the functions in `Line.py`. If there are no other files specified, the line numbers are refer to `Line.py`. There is a verbose version that prints the image directly in `project.ipynb` 

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image. (Step 1)

The code is from Line 33 to Line 71 in function `calculate_camera_distortion(images)` in `Line.py`.
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original:
![alt text][image1]
Transformed:
![alt text][image3]

Because the camera calibration matrix is the same throughout this project, I put the code related to calculation of the matrix into a separate code block in the notebooks. This can make the pipeline works in a consistent way and increases speed.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image. (Step 2)

This step is implemented in `cal_undistort(img)` from Line 81-84. I use `cv2.undistort()` to undistort the image using the camera and calibration matrix calculated in the previous step.
Here is the example:
Original:
![alt text][image2]
Transformed:
![alt text][image4]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result. ( Step 3 for gradient threshold, Step 4 for color threshold)

I used a combination of color and gradient thresholds to generate a binary image. This is in `gradient_threshold(img)`  (Line 145-160) for gradient threshold and `color_filters(img)` (Line 175- 178) for color threshold

For gradient threshold, I first transform the image to grayscale, and then use the combined result of absolute x/y sobel threshold and magitude/direction threshold. For x/y thresholding, I use (30,150). For magnitude, I use (40,250) because there is a clear distinction for color intensity between lanes and regular road pavement. I found using 40 as the lower threshold eliminates most of the noise pixels. For directional, I use (-1.3, 1.3). Because lane lines cannot be horizontal, only picking lines that are not horizontal can eliminate fake lanes caused by shadows or change of pavement. This threshold picks up most of the lane pixels for roads with darker pavement, but does not work for lane pixels with light pavement.

Therefore, it is nessesary to use color threshold to pick up lanes with light pavement. I first transformed the color image into HLS. I found the lane pixels differs most in S channel, so I use a threshold on S channel. This threshold picks up lane pixels in light pavement effectively, although it picks a lot of false positives like cars and sky. However, this is not a great deal because most of the false positives in this step will be eliminated by perspective transformation.

Then I found using S channel alone cannot solve the problem completely, and the algorithm still detected tree shades as lane pixels. Because tree shades are dark, I applied an L channel threshold to eliminate dark pixels, and problem solved for most cases.

It is also possible to apply H threshold so that only valid lane colors will be selected. However, this is very tricky to implement since the H value for white color is undefined.


Original:
![alt text][image4]

Binary:
![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.(Step 5)

 The code is in `calculate_perspective_transform():` (Line 73-79)
 
 I eyeballed the region of interest for perspective transformation, and I defined them as constant in the block right after Step 5. src are the coordinates before perspective transformation, dst are the corresponding coordinates after transformation.
 
 ```python
 src = np.float32([[220,700],[600,450],[675,450],[1050,700]])
dst = np.float32([[325,700],[325,0],[950,0],[950,700]])
 
 ```
 
 I use `cv2.getPerspectiveTransform(src, dst)` to get the transformation matrix, and `cv2.getPerspectiveTransform(dst, src)` to get the inverse transformation matrix. Then I use ` cv2.warpPerspective()` to transform the image

Original:
![alt text][image4]
Warped:
![alt text][image6]

I used a straight line image to check the correctness of the coordinates

Original:
![alt text][image7]
Warped:
![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code is from Line 183-Line 291. Because this is closely related to the calculation of curvature, I put this and next pipeline into a same function `find_lanes`. Most of the code is copied from Lesson 33 with a slightly change for the implementation of the Line class.

I first plotted a histogram for the color value for the bottom half of the image to detect the base of two lanes by choosing the maximum of the histogram on the left half of the image and on the right half of the image

```python
# Assuming you have created a warped binary image called "binary_warped"
# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
# Create an output image to draw on and  visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```
Then I used sliding-window search to find lane pixels from the bottom of the image to the top. If there are enough lane pixels found, the window will be slide to the mean of the found pixels so that it can keep finding the lanes.

![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 295 through 339 in my code in `find_lanes()` . Most of the code are copied from Lesson 35.

I used the pixel - meter conversion rate presented in the given code to calculate the real world distance

```python
# Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
```



I then used the equation that calculates the curvature from a polynomial in Lesson 35 to calculate the curvature in real world. Note `y_eval` is the maximum value in y axis.

```python
# Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
```

In order to calculate the offset of the car from the center of the road, I calculated the average of x of the bottom position of the the left lane and right lane , and compare that to the center of the image.

```python
offset = leftx_base - midpoint + rightx_base - midpoint
    if offset < 0:
        direction = "right"
    else:
        direction = "left"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Curvature = {}m'.format(int(min(left_curverad,right_curverad))),(10,50), font, 1,(255,255,255),2)
    cv2.putText(result,'Car is {}m {} of center'.format(int(abs(offset * xm_per_pix * 100)) / 100, direction),(10,150), font, 1,(255,255,255),2)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I used `cv2.warpPerspective` using the inverse matrix calculated before to transform the image back to front view.



![alt text][image10]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).



Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- The lane detected are sometimes shaky. 
- The calculated curvature varies a lot for consecutive video frames if the car is in straight lane, and it is very hard to get a good view of the curvature without pausing the video.
- Since the lane in the video are in a "dash-dot" pattern, if the dot part of the lane appears first from the car's view, the lane portion closer to the car will not be correctly detected. This is one of the reasons that causes the shaky lane artifact in the video. 
- All of the three problems above can be possibily fixed by storing recent lane detections into the class and introducing a smoothing algorithm.

- Other problems that happens for challenge videos:
- When there is a pavement change parallel to the lanes ( vertical pavement change). That pavement change will be classified as lanes. Perhaps adding a hue threshold could solve this problem because in most cases the lanes are in white or yellow. However, this threshold must be applied using other threshold mechanisms since hue value for black, white and gray are undefined, and the program could potentially assign any possible values to those colors
- When the road has great turns ( low curvature meters) , like the harder challenge video where the car is running on a mountain road, the current lane detection algorithms fails completely. Perhaps using a different region of interest for perspective transformation is required for this case. It is also helpful to determine the region of interest based on detected lane pixels from binary images. 
- When there is too much light caused by direct sunlight exposure, the threshold will not work properly.  
