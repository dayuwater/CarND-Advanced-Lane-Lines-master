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
