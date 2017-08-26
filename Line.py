import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from Line import *
import random

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

def calculate_camera_distortion(images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2]= np.mgrid[0:9,0:6].T.reshape(-1,2)



    # Arrays to store object points and image points from all the images.
    objpoints = []# 3d points in real world space
    imgpoints = []# 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(images)

    test_imgs = []

    # camera calibration results
    cameraMatrix = None
    calibrate = None


    # Step through the list and search for chessboard corners
    for i, fname in enumerate(tqdm(images)):
        img = cv2.imread(fname)
        test_imgs.append(img)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # calculate the camera matrix and calibration parameters using all the corners found in the 20 calibration images
    retval, cameraMatrix, calibrate, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return cameraMatrix, calibrate

def calculate_perspective_transform():
    # get the perspective transform matricies
    src = np.float32([[220,700],[600,450],[675,450],[1050,700]])
    dst = np.float32([[325,700],[325,0],[950,0],[950,700]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def cal_undistort(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    undist = cv2.undistort(img, cameraMatrix, calibrate, None, cameraMatrix)
    return undist

# filter the image using absolute sobel gradient in either x or y axis
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# filter the image using the magnitude of the sobel gradient
def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Calculate the magnitude 
    mag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(mag / np.max(mag) * 255)
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output

# filter the image using the direction of the sobel gradient
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(direction)
    binary_output[(direction > thresh[0]) & (direction < thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output
    
def gradient_threshold(img):
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(30, 150))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(30, 150))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(45, 255))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(-1.3, 1.3))

    combined = np.zeros_like(dir_binary)
    # Choosing the pixels that are either chosen by both x and y absolute sobel filter, 
    # or by both magnitude and directional sobel filters
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return combined

def hls_select(img, s_thresh=(0, 255), l_thresh=(0,255), h_thresh=(0,360)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s = hls[:,:,2]
    l = hls[:,:,1]
    h = hls[:,:,0]
    binary_output = np.zeros_like(s)
    binary_output[(s > s_thresh[0]) & (s <= s_thresh[1]) & (l > l_thresh[0]) & (l <= l_thresh[1]) & (h > h_thresh[0]) & (h <= h_thresh[1])] = 1
    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    return binary_output

def color_filters(img):
    img = hls_select(img, s_thresh=(90, 255), l_thresh=(40,255), h_thresh=(0,360))
    return img

def histogram_lanes(img):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    plt.plot(histogram)

def find_lanes(source_img, img, Minv, reuse = False, left_lane = None, right_lane = None):
    '''
    Find and calculate the lane curvature, and draw the lane area back to the source image
    '''
    # TODO: Store the line parameters into the Line class
    
    
    # Assuming you have created a warped binary image called "img"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # If we are reusing previous results
    if reuse:
        left_fit = left_lane.current_fit
        right_fit =  right_lane.current_fit
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    
    else:
        
        # If we are not reusing previous results, start from scratch
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
    

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # y_eval is the maximum y dimension
    y_eval = np.max(ploty)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero)) * 255
    color_warp[lefty, leftx] = [255, 0, 0]
    color_warp[righty, rightx] = [0, 0, 255]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))


    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (test_img.shape[1], test_img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(source_img, 1, newwarp, 0.3, 0)
    

    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    # Now our radius of curvature is in meters
#     print(left_curverad, 'm', right_curverad, 'm')
    # negative = right, positive = left
    offset = leftx_base - midpoint + rightx_base - midpoint
    if offset < 0:
        direction = "right"
    else:
        direction = "left"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Curvature = {}m'.format(int(min(left_curverad,right_curverad))),(10,50), font, 1,(255,255,255),2)
    cv2.putText(result,'Car is {}m {} of center'.format(int(abs(offset * xm_per_pix * 100)) / 100, direction),(10,150), font, 1,(255,255,255),2)
    
    # store the result into the Line class
    if not left_lane.detected:
        left_lane.detected = True
    if not right_lane.detected:
        right_lane.detected = True
        
    # store the left lane
    left_lane.current_fit = left_fit
    left_lane.radius_of_curvature = left_curverad
    left_lane.line_base_pos = (midpoint - leftx_base) * xm_per_pix
    left_lane.allx = leftx
    left_lane.ally = lefty
    

    # store the right lane
    right_lane.current_fit = right_fit
    right_lane.radius_of_curvature = right_curverad
    right_lane.line_base_pos = (rightx_base - midpoint) * xm_per_pix
    right_lane.allx = rightx
    right_lane.ally = righty
    
    
    return result, left_lane, right_lane

def preprocess(img, M):
    '''
    Convert the source image to perspective transformed binary lane maps
    '''
    test_img2 = cal_undistort(img.copy()) # need to apply step 2 here
    gradient_thresholded = gradient_threshold(test_img2)
    color_filtered = color_filters(test_img2)
    combined_binary = np.zeros_like(gradient_thresholded)
    combined_binary[(gradient_thresholded == 1) | (color_filtered == 1)] = 1
    binary_warped = cv2.warpPerspective(combined_binary, M, (img.shape[1], img.shape[0]))
    return binary_warped

def process_image(img, reuse = True):
    if img.shape != (720, 1280, 3):
        img = cv2.resize(img, (1280, 720))
    after_preprocess = preprocess(img,M)
    return find_lanes(img, after_preprocess, Minv, reuse = reuse, left_lane = left_lane, right_lane = right_lane)[0]



