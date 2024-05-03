#!/usr/bin/env python
# coding: utf-8

# Import Statements- "matplotlib" is used for backend plotting in Jupyter notebooks
from plantcv import plantcv as pcv
import matplotlib
import matplotlib.pyplot as plt
import argparse
#import imutils
import numpy as np
import cv2
import random as rng
from scipy import ndimage
import math
from scipy import stats

# Set figure size for the notebook
matplotlib.rcParams["figure.figsize"] = (8, 8)
pcv.params.debug = None


def options():
    """Parse command-line options
    The options function was converted from the class options in Jupyter.
    Rather than hardcoding the inputs, input arguments with the same
    variable names are used to retrieve inputs from the commandline.
    """
    parser = argparse.ArgumentParser(
        description="Citrus T50 Cut Down No Shadow Workflow")
    parser.add_argument("--image", help="Input image", required=True)
    parser.add_argument("--result", help="Results file", required=True)
    parser.add_argument("--outdir", help="Output directory", required=True)
    parser.add_argument(
        "--writeimg", help="Save output images", action="store_true")
    parser.add_argument("--debug", help="Set debug mode", default=None)
    parser.add_argument("--img_outdir", help="output directory", default=".")
    args = parser.parse_args()
    return args


def main():

    import numpy as np

    # get options
    args = options()

    # set debug to the global parameter
    pcv.params.debug = args.debug

    # increase text size in plots
    pcv.params.text_size = 5
    pcv.params.text_thickness = 20

    # Read the image
    img, path, filename = pcv.readimage(filename=args.image)

    x1 = np.shape(img)[1]
    y1 = np.shape(img)[0]

    # Step 1: Normalize the white color on a color card so you can later compare color between images.

    # Inputs:
    #   img = image object, RGB color space
    #   roi = coordinates where white reference is located, if none, it will use the whole image,
    #         otherwise (x position, y position, box width, box height)

    # white balance image based on white square from color card (in this example we focus on yellow on ruler)
    #img1 = pcv.white_balance(img, roi=((x1-45),(45),20,20));
    s = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    value = s[int(y1*0.2), int(x1*0.02)]
    if (20 < value < 105):
        img1 = pcv.white_balance(
            img, roi=((int(x1*0.02)), (int(y1*0.2)), 20, 20))
    else:
        if (s[int(y1*0.8), int(x1*0.02)] > 20):
            img1 = pcv.white_balance(
                img, roi=((int(x1*0.02)), (int(y1*0.8)), 20, 20))
        else:
            if (s[20, int(x1-40)] < 190):
                img1 = pcv.white_balance(img, roi=((int(x1-40)), (20), 20, 20))
            else:
                img1 = pcv.white_balance(
                    img, roi=((int(x1*0.02)), (int(y1*0.2)), 20, 20))
    #img1 = pcv.white_balance(img, roi=((x1-20),(20), 20, 20));
    #pcv.print_image(img1, "./results/color_correct/" + filename.rsplit('.', 1)[0] + "_1_ColorCorrect.jpg");

    # # Step 2: Isolate first ROI - outside border of fruit.

    # Pad image
    img1 = cv2.copyMakeBorder(img1, 20, 20, 20, 20,
                              cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # lets look closer at thresholding on HSV colorspace
    #s = pcv.rgb2gray_hsv(rgb_img=img1, channel="s");

    # now lets try to threshold the half showing the outer surface by the "s" channel
    #thresh_o = pcv.threshold.otsu(gray_img=s, max_value=255, object_type='light');
    #fill_o = pcv.fill(bin_img=thresh_o, size=10000);

    # get rid of noise
    #fill_o = pcv.fill_holes(fill_o);

    #wsum = np.sum(fill_o == 255);
    #bsum = np.sum(fill_o == 0);
    #wprop = wsum/(wsum + bsum);

    # if (wprop < 0.70):
    #    edge = pcv.canny_edge_detect(fill_o, sigma=3, thickness=3,low_thresh=0, high_thresh=100);
    #    edge2 = pcv.dilate(edge, 3, 3);
    #    edge3 = pcv.erode(edge2, 3, 3);
    #    fill_o = pcv.fill_holes(edge3);

    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 50, 0])
    upper = np.array([90, 255, 255])
    mask1 = cv2.inRange(hsv, lower, upper)
    #result = cv2.bitwise_and(img1, img1, mask=mask);
    #cv2.imwrite("./results/"+ filename.rsplit('.', 1)[0] + "_hsv_threshold.jpg",result);
    mask1_d = pcv.dilate(mask1, 3, 3)
    mask1_d = pcv.fill_holes(mask1_d)
    #mask = pcv.fill(mask, 500);
    mask1_e = pcv.erode(mask1_d, 3, 3)
    mask1_e = pcv.fill(mask1_e, 500)
    masked1 = pcv.apply_mask(img=img1, mask=mask1_e, mask_color='white')
    #pcv.print_image(masked1, "./results/" + filename.rsplit('.', 1)[0] + "_masked1.jpg");

    # another threshold that will help with shadowing in some of the images
    v = pcv.rgb2gray_hsv(rgb_img=masked1, channel="v")
    vt1 = pcv.threshold.binary(
        gray_img=v, threshold=70, max_value=255, object_type='dark')
    vt1 = pcv.fill(vt1, 300)
    vt1 = pcv.dilate(vt1, 2, 3)
    vt1 = pcv.erode(vt1, 2, 3)
    vt2 = pcv.logical_and(mask1_e, cv2.bitwise_not(vt1))
    vt2 = pcv.erode(vt2, 3, 2)
    vt2 = pcv.dilate(vt2, 3, 2)
    vt2 = pcv.fill_holes(vt2)
    vt2 = pcv.erode(vt2, 3, 2)
    vt2 = pcv.fill(vt2, 6000)

    # apply mask
    masked2 = pcv.apply_mask(img=masked1, mask=vt2, mask_color='white')
    masked2b = pcv.apply_mask(img=masked1, mask=vt2, mask_color='black')
    #pcv.print_image(masked2, "./results/" + filename.rsplit('.', 1)[0] + "_1_masked2.jpg")

    # Find contour of exterior of fruit to use later
    out_cntrs, out_hierarchy = cv2.findContours(
        vt2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Get dimensions of image to draw contour mask on
    h = np.shape(vt2)[0]
    w = np.shape(vt2)[1]

    # Create black image of same size to draw contours on
    out_mask = np.zeros((h, w), np.uint8)

    # Draw contour, it is thick so we can use this as a mask for pulp later
    look_cnts = cv2.drawContours(
        out_mask.copy(), out_cntrs, -1, (255, 255, 255), 1)
    look_cnts15 = cv2.drawContours(
        out_mask.copy(), out_cntrs, -1, (255, 255, 255), 15)
    #pcv.print_image(look_cnts, "./results/" + filename.rsplit('.', 1)[0] + "_0_out_cntrs.jpg")

    # Invert so we are masking the flavedo
    #look_cnts = pcv.invert(look_cnts);

    # Create mask by combining with the mask used for masked2 above
    out_mask = pcv.logical_and(vt2, look_cnts15)

    # Apply mask to original image
    #masked1 = pcv.apply_mask(img=img1, mask=fill_o, mask_color='white');

    #masked3 = pcv.gaussian_blur(img=masked2, ksize=(5, 5), sigma_x=0, sigma_y=None);
    pcv.print_image(masked2, "./results/mask2/" +
                    filename.rsplit('.', 1)[0] + "_1_masked.jpg")
    # run nb ml for identifying flavedo, pulp, albedo.
    mask = pcv.naive_bayes_classifier(rgb_img=masked2,
                                      pdf_file="./pulp_naive_bayes_pdfs.txt")

    #pulp_img = pcv.apply_mask(mask=(mask['pulp']), img=masked2, mask_color='black');
    #albedo_img = pcv.apply_mask(mask=(mask['albedo']), img=masked2, mask_color='black');
    #pcv.print_image(albedo_img, "./results/albedo/" + filename.rsplit('.', 1)[0] + "_1_albedo.jpg");
    #pcv.print_image(pulp_img, "./results/pulp/" + filename.rsplit('.', 1)[0] + "_1_pulp.jpg");

    # Albedo
    # Now albedo
    albedo = mask['albedo']
    ##
    # draw contour found for exterior of fruit to help reduce unwanted noise
    albedo2 = cv2.drawContours(albedo.copy(), out_cntrs, -1, (0, 0, 0), 10)
    #albedo2 = pcv.fill(albedo, 50);
    ##
    # Okay so some images missed parts of the albedo, because there really is not much difference in color between
    # pulp and flavedo (mostly in mandarin types)
    # To remedy this we are going to use some geometry to guess where the edges of the albedo should be
    # The assumption is that the edge where the albedo starts is the same shape as the outer edge of the fruit, but
    # will vary in where it starts
    # Step 1: define a minimum bounding rectangle around the outer contour and find center of contour (center of rect)
    # Step 2: Calculate distance from center to a sample of points along the contour to get an idea of shape
    # Step 3: Define a rectangle that encloses the points that were picked up from nb for albedo
    # Step 4: Calculate a scaling factor for the outer rectangle and inner rectangle which will be used to draw
    # inner interpolated edge of albedo
    # Step 5: Use the scaling factor, the distances, the angles that were used to define point for distance calculations,
    # and the center point to determine interpolated points that will define edges of albedo
    # Step 6: use interpolated points to draw convex hull for size comparisons == peel thickness

    # Step 1: define a minimum bounding rectangle around the outer contour and find center of contour (center of rect)

    # We need the points that make up the outer contour
    points_out = np.column_stack(np.where(look_cnts.transpose() > 0))

    # Rectangle around outer points -> returns ((center(y,x)), (h,w), (angle of rotation))
    #minRect_out = cv2.minAreaRect(points_out)
    minRect_out = cv2.boundingRect(points_out)
    # minRect_out

    # Step 2: Calculate distance from center to a sample of points along the contour to get an idea of shape
    # We need to define a couple of functions first
    center_o = (((minRect_out[2]/2)+minRect_out[0]),
                ((minRect_out[3]/2)+minRect_out[1]))

    # Use two functions to get distances
    # Define the points to use, smaller steps along circle mean more points
    angles2 = np.arange(0, 360, 4)

    # Create empty lists to store the coordinates and distances
    # These are the same length as the number of points we will use above
    xs = np.zeros_like(angles2)
    ys = np.zeros_like(angles2)
    dists = np.zeros_like(angles2)

    # A function to get the point locations that fall on outer contours when lines are drawn at a set of given angles
    def get_pt_at_angle(pts, pt, ang):
        angles = np.rad2deg(np.arctan2(*(pt - pts).T))
        angles = np.where(angles < -90, angles + 450, angles + 90)
        found = np.rint(angles) == ang
        if np.any(found):
            return pts[found][0]

    for i, angle in enumerate(angles2):
        pt1 = get_pt_at_angle(out_cntrs[0].squeeze(), center_o, angle)
        d = np.linalg.norm(pt1 - center_o)
        #xs[i] = pt1[0]
        #ys[i] = pt1[1]
        dists[i] = d
    # Actual distance calculation
    #get_distances(out_cntrs[0].squeeze(), center_o, angles2);

    # Step 3: Define a rectangle that encloses the points that were picked up from nb for albedo
    # We need to find the outer contour for those points first
    # Get those points
    points_a = np.column_stack(np.where(albedo2.transpose() > 0))

    # create hull
    hull_a = cv2.convexHull(points_a)

    # We want to draw the hull on masked rgb image to check
    result_a = masked2.copy()
    result_a = cv2.polylines(result_a, [hull_a], True, (0, 0, 255), 2)

    # Now create a mask from that hull so we can find the contour
    # create a black background that is the same size as the original mask
    albedo3 = np.zeros_like(albedo2)

    # Draw the hull from above on that mask so we can check it.
    albedo3 = cv2.polylines(albedo3, [hull_a], True, (255, 255, 255), 1)

    # Find contour of hull
    contours_a = cv2.findContours(
        albedo3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_a = contours_a[0] if len(contours_a) == 2 else contours_a[1]

    # We want the biggest one to enclose all pixels that make up mask.
    big_contour_a = max(contours_a, key=cv2.contourArea)

    # Draw contour and check what it looks like
    f_contr_a = albedo2.copy()
    f_contr_a = cv2.drawContours(
        f_contr_a, [big_contour_a], 0, (255, 255, 255), 5)

    check = np.zeros_like(albedo2)
    check = cv2.drawContours(check, [big_contour_a], 0, (255, 255, 255), -1)
    prop_a = int(np.sum(check == 255) /
                 (np.sum(check == 255) + np.sum(check == 0))*100)

    # Get points that are on that contour
    points_new = np.column_stack(np.where(f_contr_a.transpose() > 0))

    # Bounding rectangle
    minRect = cv2.boundingRect(points_new)

    # Draw rectangles to look at result
    fa2 = masked2.copy()
    fa2 = cv2.rectangle(fa2, (minRect[0], minRect[1]), (
        minRect[0]+minRect[2], minRect[1]+minRect[3]), (0, 0, 255), 2)
    fa2 = cv2.rectangle(fa2, (minRect_out[0], minRect_out[1]), (
        minRect_out[0]+minRect_out[2], minRect_out[1]+minRect_out[3]), (255, 0, 0), 2)

    # Step 4: Calculate a scaling factor for the outer rectangle and inner rectangle
    # Scale for the y coordinates
    scale_y = (
        1-(np.round((minRect_out[3] - minRect[3]) / (minRect_out[3]), 3)))

    # Scale for the x coordinates
    scale_x = (
        1-(np.round((minRect_out[2] - minRect[2]) / (minRect_out[2]), 3)))
    scale_x = 0.96 if scale_x < 0.96 else scale_x
    scale_y = 0.96 if scale_y < 0.96 else scale_y

    # Step 5: Use the scaling factor, the distances, the angles that were used to define point for distance calculations,
    # and the center point to determine interpolated points that will define edges of albedo

    # Empty lists to store new points
    x2 = np.zeros_like(angles2)
    y2 = np.zeros_like(angles2)

    # Iterate through the angle list and calculate locations of new points
    for i, angle in enumerate(angles2):
        angle = angle * np.pi / 180
        x2[i] = center_o[0] + (dists[i]*scale_x) * np.cos(angle)
        y2[i] = (np.shape(albedo2)[0]-(center_o[1] +
                 (dists[i]*scale_y) * np.sin(angle)))

    # Draw those points on the original mask
    test_albedo = albedo2.copy()
    test_albedo[y2, x2] = 255
    # pcv.plot_image(test_albedo)

    # Now for the final hull and contour
    # We are going to create a convex hull around the outermost points and interpolate the original albedo boundary
    # We need to transpose the points that make up the mask so we can use this in cv2.convexHull
    points_b = np.column_stack(np.where(test_albedo.transpose() > 0))

    # Get hull
    hull_b = cv2.convexHull(points_b)

    # Draw hull on image with other hull to compare
    result_a = cv2.polylines(result_a, [hull_b], True, (0, 255, 0), 2)
    #pcv.print_image(result_a, "./results/hulls/" + filename.rsplit('.', 1)[0] + "_hulls.jpg");

    # Now create a mask from that hull so we can find the contour
    # create a black background that is the same size as the original mask
    albedo3 = np.zeros_like(test_albedo)

    # Draw the hull from above on the black background
    albedo3 = cv2.fillPoly(albedo3, [hull_b], 255)

    # We have to find the contour of the hull to use it for anything later
    contours_b = cv2.findContours(
        albedo3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_b = contours_b[0] if len(contours_b) == 2 else contours_b[1]

    # We want the biggest one to enclose all pixels that make up mask.
    big_contour_b = max(contours_b, key=cv2.contourArea)

    # We have to pcik between big_a and big_b, come images have a great albedo mask straight from naiveBayes, so we want to keep those
    # We can determine that by looking at the proportion of the image is filled when we contour the nb mask
    # So if it is more than 50%, usually is good, so set final albedo contour to the "best" one.
    final_a_cntr = big_contour_b if prop_a < 50 else big_contour_a

    # Draw contour and check what it looks like
    f_contr_b = albedo2.copy()
    #f_contr_b = cv2.drawContours(f_contr_b, [big_contour_b], 0, (255,255,255), 5);
    f_contr_b = cv2.drawContours(
        f_contr_b, [final_a_cntr], 0, (255, 255, 255), 5)
    # pcv.plot_image(f_contr_b)

    # morphological closing, opening, noise control
    #f_contr_b2 = pcv.erode(f_contr_b, 2, 3)
    f_contr_b2 = pcv.dilate(f_contr_b, 2, 3)
    f_contr_b2 = pcv.erode(f_contr_b2, 2, 3)
    #f_contr_b2 = pcv.dilate(f_contr_b2, 3, 3);
    f_contr_b2 = pcv.fill(f_contr_b2, 300)
    f_contr_b2 = pcv.dilate(f_contr_b2, 2, 2)
    f_contr_b2 = pcv.fill(cv2.bitwise_not(f_contr_b2), 700)
    f_contr_b2 = pcv.fill(f_contr_b2, 300)
    f_contr_b2 = pcv.logical_or(f_contr_b2, cv2.bitwise_not(vt2))
    f_contr_b2 = pcv.erode(cv2.bitwise_not(f_contr_b2), 2, 3)
    f_contr_b2 = pcv.median_blur(f_contr_b2, 5)

    # Look at final mask on rgb image
    mask3 = pcv.apply_mask(masked2, f_contr_b2, "black")
    #pcv.print_image(mask3, "./results/" + filename.rsplit('.', 1)[0] + "_2_albedo.jpg");

    ########
    # PULP
    ########
    ##
    # Now for pulp.
    pulp = mask['pulp']

    # Use albedo mask to trim noise from pulp
    pulp2 = pcv.logical_and(pulp, albedo3)
    pulp2 = pcv.logical_and(pulp2, cv2.bitwise_not(f_contr_b2))

    pulp2 = pcv.dilate(pulp2, 2, 3)
    pulp2 = pcv.erode(pulp2, 2, 3)

    # Use contour from albedo to also trim a little
    pulp2 = cv2.drawContours(pulp2, [big_contour_b], 0, (0, 0, 0), 15)
    pulp2 = cv2.drawContours(pulp2, out_cntrs, -1, (0, 0, 0), 20)

    pulp2 = pcv.dilate(pulp2, 3, 2)
    pulp2 = pcv.erode(pulp2, 3, 2)

    points_p = np.column_stack(np.where(pulp2.transpose() > 0))

    # create hull
    hull_p = cv2.convexHull(points_p)

    # We want to draw the hull on masked rgb image to check
    result_p = masked2.copy()
    result_p = cv2.polylines(result_p, [hull_p], True, (0, 0, 255), 2)

    # Now create a mask from that hull
    # create a black background that is the same size as the original mask
    pulp_tmp = np.zeros_like(pulp2)

    # Draw the hull from above on that mask so we can check it.
    pulp_tmp = cv2.polylines(pulp_tmp, [hull_p], True, (255, 255, 255), 1)
    # pcv.plot_image(pulp_tmp)

    # Find contour of hull
    contours_p1 = cv2.findContours(
        pulp_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_p1 = contours_p1[0] if len(contours_p1) == 2 else contours_p1[1]

    # We want the biggest one to enclose all pixels that make up mask.
    big_contour_p1 = max(contours_p1, key=cv2.contourArea)

    # Draw contour and check what it looks like
    f_contr_p1 = pulp2.copy()
    f_contr_p1 = cv2.drawContours(
        f_contr_p1, [big_contour_p1], 0, (255, 255, 255), -1)
    # pcv.plot_image(f_contr_p1)

    pulp_tmp2 = pcv.logical_and(pulp, f_contr_p1)
    pulp_tmp2 = cv2.drawContours(
        pulp_tmp2, [big_contour_p1], 0, (255, 255, 255), 2)
    pulp2 = pcv.fill_holes(pulp_tmp2)
    pmask = pcv.apply_mask(masked2.copy(), pulp2, "white")

    l = pcv.rgb2gray_lab(rgb_img=pmask, channel="l")
    lt1 = pcv.threshold.otsu(l, 255, "dark")
    #pcv.print_image(lt_otsu, "./results/" + filename.rsplit('.', 1)[0] + "_3_otsumask.jpg");

    #lt1 = pcv.threshold.binary(gray_img=l, threshold=150, max_value=255, object_type='dark');
    #pcv.print_image(lt1, "./results/" + filename.rsplit('.', 1)[0] + "_4_regmask.jpg");

    lt2 = pcv.erode(lt1, 2, 2)
    lt2 = pcv.dilate(lt2, 2, 2)
    lt2 = pcv.erode(lt2, 2, 2)
    lt2 = pcv.fill(lt2, 300)
    lt2 = pcv.fill(pcv.invert(lt2), 500)
    pulp2 = pcv.invert(lt2)

    # Dilate a little to clean up edges.
    pulp2 = pcv.median_blur(pulp2, 3)
    pulp2 = pcv.logical_and(pulp2, cv2.bitwise_not(out_mask))

    ##
    # Try to trim more down noise from outer parts of pulp usually due to flavedo being included in masked image from nb
    #pulp2 = pcv.erode(pulp2, 2, 3);
    ##
    # Fill in noise in the larger part of the pulp
    #pulp2 = pcv.fill(pulp2, 200);
    #pulp2 = pcv.fill_holes(pulp2);
    #pulp2 = pcv.median_blur(pulp2, 3);
    ##
    ##
    # Combined masks for pulp and albedo to make sure albedo is properly masked (sometimes we end up including albedo when cleaning up noise)
    #pulp3 = pcv.logical_and(pulp2, f_contr_a2);

    # Apply mask
    pmask = pcv.apply_mask(masked2.copy(), pulp2, "black")
    #pcv.print_image(pmask, "./results/" + filename.rsplit('.', 1)[0] + "_3_pulp.jpg");

    pulp3 = pcv.logical_and(pulp2, cv2.bitwise_not(out_mask))
    ##
    # Now fro convex hull around pulp
    points_p = np.column_stack(np.where(pulp2.transpose() > 0))
    hull_p = cv2.convexHull(points_p)
    ##
    # Create mask to find contours
    mask_p = np.zeros_like(pulp2)
    mask_p = cv2.fillPoly(mask_p, [hull_p], 255)
    ##
    # Find contours
    contours_p = cv2.findContours(
        mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_p = contours_p[0] if len(contours_p) == 2 else contours_p[1]
    big_contour_p = max(contours_p, key=cv2.contourArea)

    ##
    # Now we are going to draw all three contours on masked image 2 to see what it looks like
    contrboth = masked2.copy()
    # outer contour
    contrboth = cv2.drawContours(contrboth, out_cntrs, 0, (0, 255, 0), 2)
    # albedo contour
    contrboth = cv2.drawContours(contrboth, [final_a_cntr], 0, (0, 0, 255), 2)
    # pulp contour
    contrboth = cv2.drawContours(contrboth, [big_contour_p], 0, (255, 0, 0), 2)

    #pcv.print_image(contrboth, "./results/" + filename.rsplit('.', 1)[0] + "_4_finalcntrs.jpg");

    # Flavedo
    # We are going to create the images we need to get measurements on internal size and color.
    # Start with flavedo
    flavedo = mask['flavedo']

    # Combine with original mask so white mask out background
    flavedo2 = pcv.logical_and(flavedo, vt2)
    flavedo3 = pcv.logical_and(flavedo2, cv2.bitwise_not(f_contr_p1))
    flavedo_img = pcv.apply_mask(
        mask=flavedo3, img=masked2, mask_color='black')
    #pcv.print_image(flavedo_img, "./results/" + filename.rsplit('.', 1)[0] + "_3_flavedo.jpg");

    # Some noise clean up
    #flavedo2 = pcv.median_blur(flavedo2, 5);
    #flavedo3 = pcv.fill(flavedo2, 500);
    ##

    # We are going to take each mask for each layer and identify which objects we want to analyze
    # So we will have a step for whole fruit, albedo, and pulp.

    # Steps
    # 1 ID objects (contours) from mask
    # 2 Create a ROI to define which objects we want
    # 3 Combine all obects we want to keep so they are treated as a single object and not analyzed separately
    # 4 Analysis of object(s)

    # Outer
    # Identify objects
    # Inputs:
    #   img - RGB or grayscale image data for plotting
    #   mask - Binary mask used for detecting contours
    out_obj, out_obj_hier = pcv.find_objects(img=masked2, mask=vt2)

    # Create a ROI for whole fruit
    # For this one I think we can just use the points we already extracted
    # from the outer contour to create a custom roi

    roi_out, roi_hier_out = pcv.roi.custom(masked2, vertices=points_out)

    # Decide which objects to keep
    # Simple for this because it is one object
    out_obj2, out_hier2, kept_mask_out, out_area = pcv.roi_objects(img=masked2, roi_contour=roi_out,
                                                                   roi_hierarchy=roi_hier_out,
                                                                   object_contour=out_obj,
                                                                   obj_hierarchy=out_obj_hier,
                                                                   roi_type='cutto')

    # Object combine kept objects - output from this is ready to be analyzed

    # Inputs:
    #   img - RGB or grayscale image data for plotting
    #   contours - Contour list
    #   hierarchy - Contour hierarchy array
    obj_out_fin, mask_out_fin = pcv.object_composition(
        img=masked2, contours=out_obj2, hierarchy=out_hier2)

    out1 = pcv.analyze_object(
        img=masked2b.copy(), obj=obj_out_fin, mask=mask_out_fin, label="outer")

    # Albedo

    # Identify objects

    # Inputs:
    #   img - RGB or grayscale image data for plotting
    #   mask - Binary mask used for detecting contours
    alb_obj, alb_obj_hier = pcv.find_objects(img=masked2, mask=f_contr_b2)

    # Create a ROI for albedo
    # For this one I think we will create a rectangular roi

    roi_alb, roi_hier_alb = pcv.roi.rectangle(masked2, 2, 2, int(
        np.shape(masked2)[0]*0.95), int(np.shape(masked2)[1]*0.75))

    # Decide which objects to keep
    # Simple for this because it is one object
    alb_obj2, alb_hier2, kept_mask_alb, alb_area = pcv.roi_objects(img=masked2, roi_contour=roi_alb,
                                                                   roi_hierarchy=roi_hier_alb,
                                                                   object_contour=alb_obj,
                                                                   obj_hierarchy=alb_obj_hier,
                                                                   roi_type='partial')

    # Object combine kept objects - output from this is ready to be analyzed

    # Inputs:
    #   img - RGB or grayscale image data for plotting
    #   contours - Contour list
    #   hierarchy - Contour hierarchy array
    obj_alb_fin, mask_alb_fin = pcv.object_composition(
        img=masked2, contours=alb_obj2, hierarchy=alb_hier2)

    alb1 = pcv.analyze_object(
        img=masked2b.copy(), obj=obj_alb_fin, mask=mask_alb_fin, label="albedo")

    # Pulp

    # Identify objects

    # Inputs:
    #   img - RGB or grayscale image data for plotting
    #   mask - Binary mask used for detecting contours
    pulp_obj, pulp_obj_hier = pcv.find_objects(img=masked2, mask=pulp2)

    # Create a ROI for albedo
    # For this one I think we will create a circular roi

    roi_pulp, roi_hier_pulp = pcv.roi.circle(img=masked2, x=int(
        center_o[0]), y=int(center_o[1]), r=int(np.shape(masked2)[1]/3))

    # Decide which objects to keep
    # Simple for this because it is one object
    pulp_obj2, pulp_hier2, kept_mask_pulp, pulp_area = pcv.roi_objects(img=masked2, roi_contour=roi_pulp,
                                                                       roi_hierarchy=roi_hier_pulp,
                                                                       object_contour=pulp_obj,
                                                                       obj_hierarchy=pulp_obj_hier,
                                                                       roi_type='partial')

    # Object combine kept objects - output from this is ready to be analyzed

    # Inputs:
    #   img - RGB or grayscale image data for plotting
    #   contours - Contour list
    #   hierarchy - Contour hierarchy array
    obj_pulp_fin, mask_pulp_fin = pcv.object_composition(
        img=masked2, contours=pulp_obj2, hierarchy=pulp_hier2)

    pulp1 = pcv.analyze_object(
        img=masked2b.copy(), obj=obj_pulp_fin, mask=mask_pulp_fin, label="pulp")

    pulp_color = pcv.analyze_color(img1, kept_mask_pulp, label="pulp")

    pcv.outputs.save_results(filename=args.result)

    # figt, axes = plt.subplot_mosaic("AAABBBCCCDDD;EEEEFFFFGGGG", figsize=(10,7))
    # axes["A"].imshow(cv2.cvtColor(masked2b, cv2.COLOR_BGR2RGB))
    # axes["A"].axis('off')
    # axes["A"].set_title("Whole")
    # axes["B"].imshow(cv2.cvtColor(flavedo_img, cv2.COLOR_BGR2RGB))
    # axes["B"].axis('off')
    # axes["B"].set_title("Flavedo")
    # axes["C"].imshow(cv2.cvtColor(mask3, cv2.COLOR_BGR2RGB))
    # axes["C"].axis('off')
    # axes["C"].set_title("Albedo")
    # axes["D"].imshow(cv2.cvtColor(pmask, cv2.COLOR_BGR2RGB))
    # axes["D"].axis('off')
    # axes["D"].set_title("Pulp")
    # axes["E"].imshow(cv2.cvtColor(out1, cv2.COLOR_BGR2RGB))
    # axes["E"].axis('off')
    # axes["E"].set_title("Whole Analyzed")
    # axes["F"].imshow(cv2.cvtColor(alb1, cv2.COLOR_BGR2RGB))
    # axes["F"].axis('off')
    # axes["F"].set_title("Albedo Analyzed")
    # axes["G"].imshow(cv2.cvtColor(pulp1, cv2.COLOR_BGR2RGB))
    # axes["G"].axis('off')
    # axes["G"].set_title("Pulp Analyzed")
    # plt.savefig("./results/" + filename.rsplit('.', 1)[0] + "_output.jpg")
    # plt.close()


if __name__ == '__main__':
    main()
