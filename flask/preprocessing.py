import numpy as np
import cv2
import imutils


# gets 4 vertices as dictionary with x and y properties
def transform_vertices(vertices, ratio):
    output_vertices = []
    # get coordinates for fixed size picture and squash them between 0 and 1
    for i, vertice in enumerate(vertices):
        if i % 2 == 0:
            output_vertices.append(vertice * ratio[0])
        else:
            output_vertices.append(vertice * ratio[1])
    return output_vertices


# distance between two points with pythagoras
def calculate_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# do four point transformation (perspective transformation)
def four_point_transform(points, image):
    top_left = (points[0], points[1])
    bottom_left = (points[2], points[3])
    bottom_right = (points[4], points[5])
    top_right = (points[6], points[7])

    # Take largest width as final width of cutout image
    width_top = calculate_distance(top_left, top_right)
    width_bottom = calculate_distance(bottom_left, bottom_right)
    max_width = int(max((width_top, width_bottom)))

    height_left = calculate_distance(top_left, bottom_left)
    height_right = calculate_distance(top_right, bottom_right)
    max_height = int(max((height_left, height_right)))

    dst = np.array([
        [0, 0],
        [0, max_height],
        [max_width, max_height],
        [max_width, 0]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.array((top_left, bottom_left, bottom_right, top_right), dtype="float32"), dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


# https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[3] = pts[np.argmin(diff)]
    rect[1] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# Algorithm to skew image correctly
# Only works on well photographed pictures of receipts
def edge_detection(b_w_image, type='approx'):
    # find contours
    contours = cv2.findContours(b_w_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # calculate a min area for the bill
    res_y = len(b_w_image)
    res_x = len(b_w_image[0])
    min_area = res_x * res_y / 100

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break

        if type == 'approx':
            peri = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * peri, True)
        else:
            bbox = cv2.minAreaRect(contour)
            approximation = np.array(cv2.boxPoints(bbox))

        if len(approximation) == 4:
            # reshape approximation for better coding experience
            approximation = np.array(approximation).reshape((4, 2))

            # order points to tl, bl, br, tr
            points = order_points(approximation)
            # if bbox goes out of bound, limit to image dimension
            if type == "bbox":
                for point in points:
                    if point[0] < 0:
                        point[0] = 0
                    elif point[0] > res_x:
                        point[0] = res_x

                    if point[1] < 0:
                        point[1] = 0
                    elif point[1] > res_y:
                        point[1] = res_y
            return points
