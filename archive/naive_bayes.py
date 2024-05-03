from plantcv import plantcv as pcv
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
import random as rng
from scipy import ndimage
import math
from scipy import stats

matplotlib.rcParams["figure.figsize"] = (8, 8)
pcv.params.debug = None

def naive_bayes(image, result):
    # increase text size in plots
    pcv.params.text_size = 5
    pcv.params.text_thickness = 20

    # Read the image
    img, path, filename = pcv.readimage(filename=image)

    x1 = np.shape(img)[1]
    y1 = np.shape(img)[0]

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
    
    img1 = cv2.copyMakeBorder(img1, 20, 20, 20, 20,
                              cv2.BORDER_CONSTANT, value=[255, 255, 255])


    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 50, 0])
    upper = np.array([90, 255, 255])
    mask1 = cv2.inRange(hsv, lower, upper)
    mask1_d = pcv.dilate(mask1, 3, 3)
    mask1_d = pcv.fill_holes(mask1_d)
    mask1_e = pcv.erode(mask1_d, 3, 3)
    mask1_e = pcv.fill(mask1_e, 500)
    masked1 = pcv.apply_mask(img=img1, mask=mask1_e, mask_color='white')

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

    # Create mask by combining with the mask used for masked2 above
    out_mask = pcv.logical_and(vt2, look_cnts15)

    # pcv.print_image(masked2, "./results/mask2/" +
    #                 filename.rsplit('.', 1)[0] + "_1_masked.jpg")
    # run nb ml for identifying flavedo, pulp, albedo.
    mask = pcv.naive_bayes_classifier(rgb_img=masked2,
                                      pdf_file="./pulp_naive_bayes_pdfs.txt")

    
    # Albedo
    # Now albedo
    albedo = mask['albedo']
    ##
    # draw contour found for exterior of fruit to help reduce unwanted noise
    albedo2 = cv2.drawContours(albedo.copy(), out_cntrs, -1, (0, 0, 0), 10)
   
    # We need the points that make up the outer contour
    points_out = np.column_stack(np.where(look_cnts.transpose() > 0))

    minRect_out = cv2.boundingRect(points_out)

    center_o = (((minRect_out[2]/2)+minRect_out[0]),
                ((minRect_out[3]/2)+minRect_out[1]))

    angles2 = np.arange(0, 360, 4)

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
        dists[i] = d

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

    points_b = np.column_stack(np.where(test_albedo.transpose() > 0))

    # Get hull
    hull_b = cv2.convexHull(points_b)

    # Draw hull on image with other hull to compare
    result_a = cv2.polylines(result_a, [hull_b], True, (0, 255, 0), 2)

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

    #   mask - Binary mask used for detecting contours
    out_obj, out_obj_hier = pcv.find_objects(img=masked2, mask=vt2)

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


    obj_pulp_fin, mask_pulp_fin = pcv.object_composition(
        img=masked2, contours=pulp_obj2, hierarchy=pulp_hier2)

    pulp1 = pcv.analyze_object(
        img=masked2b.copy(), obj=obj_pulp_fin, mask=mask_pulp_fin, label="pulp")

    pulp_color = pcv.analyze_color(img1, kept_mask_pulp, label="pulp")

    # out_img = cv2.copyMakeBorder(img, 20, 20, 20, 20,
    #                           cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # flavedo_out = pcv.apply_mask(
    #     mask=flavedo3, img=out_img, mask_color='black')                            
    # albedo_out = pcv.apply_mask(masked2, out_img, "black")
    # pulp_out = pcv.apply_mask(masked2.copy(), out_img, "white")

    return (filename, flavedo_img[20:-20,20:-20], mask3[20:-20,20:-20], pmask[20:-20,20:-20])

    # pcv.outputs.save_results(filename=result)

    # figt, axes = plt.subplot_mosaic(
    #     "AAABBBCCCDDD;EEEEFFFFGGGG", figsize=(10, 7))
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
    naive_bayes("Images\FTP.6.60.3_Fruit Quality Fruit1_1_2021-11-04-08-26-10.jpg", "out.json")
