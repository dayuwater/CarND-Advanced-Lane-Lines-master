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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

Note: All of my code are in `project.ipynb`. Finding the corresponding block in a IPython noteBook is troublesome just by "block number". Fortunately, I put titles of every important step in my notebook file, and I will mark the corresponding title for each rubric point.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image. (Step 1)

The code for this step is contained in the first code cell of the IPython notebook located in "./project.ipynb".  The title should be "First, I'll compute the camera calibration using chessboard images"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

Because the camera calibration matrix is the same throughout this project, I put the code related to calculation of the matrix into a separate code block. The second block is the testing block for camera calibration.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image. (Step 2)

I use `cv2.undistort()` to undistort the image using the camera and calibration matrix calculated in the previous step.
Here is the example:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result. ( Step 3 for gradient threshold, Step 4 for color threshold)

I used a combination of color and gradient thresholds to generate a binary image.

For gradient threshold, I first transform the image to grayscale, and then use the combined result of absolute x/y sobel threshold and magitude/direction threshold. The code is at 4th block below Step 3. The function name is `def gradient_threshold(img)`. For x/y thresholding, I use (30,150). For magnitude, I use (40,250) because there is a clear distinction for color intensity between lanes and regular road pavement. I found using 40 as the lower threshold eliminates most of the noise pixels. For directional, I use (-1.3, 1.3). Because lane lines cannot be horizontal, only picking lines that are not horizontal can eliminate fake lanes caused by shadows or change of pavement. This threshold picks up most of the lane pixels for roads with darker pavement, but does not work for lane pixels with light pavement.

Therefore, it is nessesary to use color threshold to pick up lanes with light pavement. The code is in the block just under "Step 4". (`hls_select()` and `color_filters()`) I first transformed the color image into HLS. I found the lane pixels differs most in S channel, so I use a threshold on S channel. This threshold picks up lane pixels in light pavement effectively, although it picks a lot of false positives like cars and sky. However, this is not a great deal because most of the false positives in this step will be eliminated by perspective transformation.

Then I found using S channel alone cannot solve the problem completely, and the algorithm still detected tree shades as lane pixels. Because tree shades are dark, I applied an L channel threshold to eliminate dark pixels, and problem solved for most cases.



![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.(Step 5)

 All the code related to this step are in the code block right below Step 5.
 
 I eyeballed the region of interest for perspective transformation, and I defined them as constant in the block right after Step 5. src are the coordinates before perspective transformation, dst are the corresponding coordinates after transformation.
 
 ```python
 src = np.float32([[220,700],[600,450],[675,450],[1050,700]])
dst = np.float32([[325,700],[325,0],[950,0],[950,700]])
 
 ```
 
 I use `cv2.getPerspectiveTransform(src, dst)` to get the transformation matrix, and `cv2.getPerspectiveTransform(dst, src)` to get the inverse transformation matrix. Then I use ` cv2.warpPerspective()` to transform the image

![alt text][image4]

I used a straight line image to check the correctness of the coordinates

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code are in 5th block below "Step 6". 

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

This part is also presented in line 11 - 90 in `find_lanes()` in the section "Try to process the lane finding in one block with concise code using the class"

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 113 through 129 in my code in `find_lanes()` in the section "Try to process the lane finding in one block with concise code using the class"

I used the pixel - meter conversion rate presented in the given code to calculate the real world distance

```python
# Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code are in `find_lanes()` in the section "Try to process the lane finding in one block with concise code using the class". It is the 2nd block from that title. 

The lines are from 92 to 109. I used `cv2.warpPerspective()` with the inverse transformation matrix calcuated before to transform the point detected back to original image.




![alt text][image6]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

In order to create a single function used for video pipeline, I combined important things in previous sections into a new function.
The code are in `find_lanes()` in the section "Try to process the lane finding in one block with concise code using the class". It is the 2nd block from that title. It detect lane pixels, fit into polynomials and calculate the curvature and offset of the car. It also draws all the information back to the image. In addition, it stores the detected lanes into provided class for consistency in video pipeline. All of the descriptions of that function are in the comments of that code

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- The lane detected are sometimes shaky. 
- The calculated curvature varies a lot for consecutive video frames, and it is very hard to get a good view of the curvature without pausing the video.
- Since the lane in the video are in a "dash-dot" pattern, if the dot part of the lane appears first from the car's view, the lane portion closer to the car will not be correctly detected. This is one of the reasons that causes the shaky lane artifact in the video. 
- All of the three problems above can be possibily fixed by storing recent lane detections into the class and introducing a smoothing algorithm.

- Other problems that happens for challenge videos:
- When there is a pavement change parallel to the lanes ( vertical pavement change). That pavement change will be classified as lanes. Perhaps adding a hue threshold could solve this problem because in most cases the lanes are in white or yellow. However, this threshold must be applied using other threshold mechanisms since hue value for black, white and gray are undefined, and the program could potentially assign any possible values to those colors
- When the road has great turns ( low curvature meters) , like the harder challenge video where the car is running on a mountain road, the current lane detection algorithms fails completely. Perhaps using a different region of interest for perspective transformation is required for this case. It is also helpful to determine the region of interest based on detected lane pixels from binary images. 
- When there is too much light caused by direct sunlight exposure, the threshold will not work properly.  