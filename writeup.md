## Writeup

---
<img src="https://github.com/BVG85/Project-4-Advanced-Lane-Finding/blob/master/output_images/warped_result.jpg">

## Advanced Lane Finding Project

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

[image1]: ./camera_cal/calibration1.jpg "Calibration - distorted"
[image2]: ./output_images/undistorted.jpg "Undistorted"
[image3]: ./test_images/test2.jpg
[image4]: ./output_images/calibrated.jpg "Test Image Calibrated"
[image5]: ./output_images/pipeline.jpg "Pipeline Result"
[image6]: ./output_images/calibratedPT.jpg "Calibrated Image for Perpective Transform"
[image7]: ./output_images/warp.jpg "Perspective Transform"
[image8]: ./output_images/histo.jpg "Histogram"
[image9]: ./output_images/swindow.jpg "Sliding Window"
[image10]: ./output_images/fit.jpg "Fitted drawing"
[image11]: ./output_images/histo.jpg "Histogram"
[image12]: ./output_images/warped_result.jpg "Final Result"
[video1]: ./project_video.mp4 "Video"
[video2]: ./output.mp4 "Video Output"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This file is the writeup file for the project.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step can be found in this notebook [P4 Script](https://github.com/BVG85/Project-4-Advanced-Lane-Finding/blob/master/P4%20Script.ipynb)

First "object points" were prepared, which will be the (x, y, z) coordinates of the chessboard corners in the world. It is assumed that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners are detected in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

The output `objpoints` and `imgpoints` were used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  This distortion correction was applied to to the test image using the `cv2.undistort()` function to obtain the following result.

##### Before Calibration
![alt text][image1]

##### After Calibration
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

As a first step, distortion correction was done with the `cal_and_undistort` function. The results can be seen below:
##### Before Calibration
![alt text][image3]
##### After Calibration
![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

A combination of color and gradient thresholds were used to generate a binary image in the `pipeline` function. A secondary function, named `pipeline2` was created for thresholding experiments.  The HLS color space was used with directional thresholding. An example of the pipeline output can be seen below. 

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

One of the straight lines test images was undistored with the `cal_and_undistort` function and then used to identify 4 source and destination coordinates within the `warp` function. The source and destination coordinates were hardcoded as below: 

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    # Four source coordinates
    src = np.float32(
        [[581, 462],
         [704, 462],
         [280, 665],
         [1025, 665]])
    offset = 150
    # Four desired coordinates
    dst = np.float32(
        [[offset, 0],
         [1280-offset, 0],
         [offset, 720],
         [1280- offset, 720]])
```

The perspective transform was tested and the results can be seen below: 
##### Calibrated Straight Lines Test Image
![alt text][image6]

##### Warped Image
![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

A histogram was then plotted by using the binary warped result (from the `warp` function and the `pipeline` functions). 

![alt text][image8]

Using the sliding window method to identify nonzero pixels and fitting this to a  2nd order polynomial lane lines were plotted:

![alt text][image9]
![alt text][image10]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of the curvature was calculated and the converted in x and y from pixel space to meters. It was assumed that the lane is about 30 meters long and 3.7 meters wide. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The following code was used to plot the result on the road with the result below.

```Python
# Create an image to draw the lines on
warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))


# Four source coordinates
nsrc = np.float32(
        [[540, 490],
         [750, 490],
         [280, 665],
         [1025, 665]])

offset = 150
    # Four desired coordinates
ndst = np.float32(
        [[offset, 0],
         [1280-offset, 0],
         [offset, 720],
         [1280- offset, 720]])


nMinv = cv2.getPerspectiveTransform(ndst, nsrc)

# Warp the blank back to original image space using inverse perspective matrix (Minv)
                                                    #binary_warped.shape
newwarp = cv2.warpPerspective(color_warp, nMinv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
# Combine the result with the original image
fresult = cv2.addWeighted(cal_cam_image, 1, newwarp, 0.3, 0)
plt.imshow(fresult)

mpimg.imsave('./output_images/warped_result2.jpg',fresult)
```

![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./outputT.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

High contrast lighting conditions remain challenging. The time of day and shadows from other cars can also prove problematic.
If other vehicles were to merge into the same lane will also prove to be problematic for the pipeline. The images can be processed to be of lower contrast and other color spaces can be explored further to make the pipeline more robust.
