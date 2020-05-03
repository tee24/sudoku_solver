import numpy as np
import cv2
import sys
from operator import itemgetter

np.set_printoptions(threshold=sys.maxsize)

corner_list = []
ordered_contours = []
interior_contours = []
corner_list_final = []

def image_show(image, window_name=""):
	"""
	Image shower, useful for tweaking the image preprocessing parameters.
	:param image: The image you want to show.
	:return: None
	"""
	cv2.namedWindow(window_name)  # Create a named window
	cv2.moveWindow(window_name, 1100, 60)  # Move it to (40,30)
	cv2.imshow(window_name,image)
	cv2.waitKey(0)

def image_resize(image, length = 500):
	"""
	An image re-sizer. Aspect ratio is maintained.
	:param image: open-cv loaded image.
	:param length: The re-sized images height in pixels.
	:return: The re-sized image.
	"""
	aspect_ratio = image.shape[1]/image.shape[0]
	width = length/aspect_ratio
	image = cv2.resize(image, (int(length),int(width)))
	return image


def contour_rect(contour, rect):
	"""
	This function loops through each corner of a open-cv rect object enclosing a contour and returns the closest point
	on the contour for each of the rect corners. It is used to find the four corners of the sudoku grid to assist with
	the perspective transformation.

	:param contour: An open-cv contour.
	:param rect: An open-cv rect.
	:return: A numpy array of shape (4,2)
	"""
	contour_corners = []

	for corner in rect:
		distances = []

		for point in contour:
			distances.append(np.linalg.norm(corner - point))

		contour_corners.append(contour[distances.index(min(distances))])
	contour_corners = np.array(contour_corners)
	contour_corners = np.resize(contour_corners, (4,2))
	return contour_corners

def distance(a, b):
	"""
	:param a: The x component of an element in R^2.
	:param b: The y component of an element in R^2.
	:return: A real number, the euclidean distance between (x,y) and the origin.
	"""
	point = np.array((a, b))
	origin = np.array((0, 0))
	return np.linalg.norm(point - origin)


def corners_finder(contour):
	"""
	Given an open-cv contour this function returns a list of 2-tuples, the first element is the top left hand corner of
	the contour the second element is the bottom right hand corner.
	:param contour: An open-cv contour, this will roughly correspond to one element of the sudoku grid.
	:return: List of two 2-tuples.
	"""

	num_points = contour.shape[0]
	moduli = []
	points = []
	for j in range(num_points):
		a = contour[j][0][0]
		b = contour[j][0][1]
		points.append((a, b))
		modulus = distance(a, b)
		moduli.append(modulus)
	bottom_right_vertex = points[moduli.index(max(moduli))]
	top_left_vertex = points[moduli.index(min(moduli))]
	return [top_left_vertex, bottom_right_vertex]


def cropper(image, corners):
	"""
	Crops an image to a rectangular shape. Used to split a grid up into its constituent parts so that they can be fed
	to tesseract-ocr.
	:param image: The image of the complete puzzle grid.
	:param corners: The corners of the subset of the image that we wish to crop to. This should correspond to one square
	in the grid.
	:return: The cropped image.
	"""
	roi = image[corners[0][1]: corners[1][1], corners[0][0]: corners[1][0]]
	y, x = roi.shape
	return roi[1: y - 1:, 1: x - 1]


def splitter(image_path):
	"""
	Decides if we need to apply morphological transformations, then calculates/orders contours and creates the list of
	numpy arrays to represent the board.
	:param image_path: String.
	:return: List of numpy arrays. Each element is a numpy representation of a grid square.
	"""
	array_pieces = []

	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	image = image_resize(image, 700)
	#################################################################################################################
	#################################################################################################################
	#################################################################################################################
	image_show(image)
	#################################################################################################################
	#################################################################################################################
	#################################################################################################################
	solution_output_image = image.copy()
	_, thresh = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	cells = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == 0]

	if len(cells) != 81:
		print("PROCESSING IMAGE")
		contours, hierarchy, thresh, solution_output_image = preprocessing_image(image_path)

	if len([contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == 0]) != 81:
		print("Image not clear enough, try with a different image!")
		print(len([contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == 0]))
		quit()

	solution_output_image = np.array(solution_output_image)
	solution_output_image = np.stack((solution_output_image,) * 3, axis=-1)  # change grayscale image to 3 channel rgb


	for i in range(len(contours)):  # this loop will go through each contour and find where it should be ordered based on the top left corner
		if hierarchy[0][i][3] == 0:
			corner_list.append(corners_finder(contours[i])[0])  # add top left corner to list
			interior_contours.append(contours[i])

	corner_list_unsorted = corner_list.copy()  # copy of the unordered list
	corner_list.sort(key=itemgetter(0))

	corner_list_sorted = []

	for i in range(9):
		sub_corners = sorted(corner_list[9 * i: 9 * (i + 1)], key=itemgetter(1))
		corner_list_sorted.append(sub_corners)

	corner_list_sorted = [item for sublist in corner_list_sorted for item in sublist]

	for _ in range(len(corner_list_sorted)):
		corner_list_final.append(corner_list_sorted[_])

	for i in range(len(corner_list)):
		index = corner_list_unsorted.index(corner_list_sorted[i])
		ordered_contours.append(interior_contours[index])

	for i in range(len(ordered_contours)):
		result_img = cropper(thresh, corners_finder(ordered_contours[i]))  # image is cropped to its "cells"
		cv2.imwrite(f"C:\\Users\\Tom\\PycharmProject\\sudokuSolver\\computerVision\\assets\\{i}.png", result_img)  # write image to hard drive
		array_pieces.append(np.invert(np.array(result_img)))
	#################################################################################################################
	#################################################################################################################
	#################################################################################################################
	###########################################################################################################################
	thresh2 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
	for i in range(len(interior_contours)):
		cv2.drawContours(thresh2, interior_contours, i, (0,255,0), 2)
	image_show(thresh2, "with contours internally drawn")
	###########################################################################################################################
	#################################################################################################################
	#################################################################################################################
	#################################################################################################################
	return array_pieces, solution_output_image

def preprocessing_image(image_path):
	"""
	Pre-processing function.
	Intended for use with photos of puzzles rather than screenshots.
	This will perform transformations to image to make ocr more accurate.
	The function will first perform threshholding and contour recognition to isolate the main puzzle grid.
	Then it will apply a perspective transformation to make the grid fit the screen.
	Finally it will apply adaptive threshholding and find the contours and hierarchy of the processed image.
	:param image_path: String. The location of the image.
	:return: contours: open-cv contour. hierarchy: open-cv hierarchy. new_img: image.
	"""
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	image = image_resize(image, 700)
	image = cv2.GaussianBlur(image, (1, 1), 0)
	im_copy = image.copy()

	result_img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 10)
	#################################################################################################################
	#################################################################################################################
	#################################################################################################################
	image_show(result_img, "looking for max contour")
	#################################################################################################################
	#################################################################################################################
	#################################################################################################################
	contours, _ = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	c = max(contours, key=cv2.contourArea)

	rect = cv2.minAreaRect(c)
	width, height = (int(x) for x in rect[1])
	box = contour_rect(c, cv2.boxPoints(rect))
	box = sorted(box, key=lambda x: x[1])
	box[:2] = sorted(box[:2], key=lambda x: x[0])
	box[2:] = sorted(box[2:], key=lambda x: x[0])
	pts1 = np.float32(box)
	pts2 = np.float32([[0, 0], [height, 0], [0, width], [height, width]])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	image_show(im_copy, "")
	dst = cv2.warpPerspective(im_copy, M, (height, width))


	#################################################################################################################
	#################################################################################################################
	#################################################################################################################
	delete = result_img.copy()
	delete = cv2.cvtColor(delete, cv2.COLOR_GRAY2RGB)
	cv2.drawContours(delete, [c], -1, (0, 255, 0), 2)
	cv2.circle(delete, (box[0][0], box[0][1]), 4, (0, 0, 255), 12)
	cv2.circle(delete, (box[1][0], box[1][1]), 4, (0, 0, 255), 12)
	cv2.circle(delete, (box[2][0], box[2][1]), 4, (0, 0, 255), 12)
	cv2.circle(delete, (box[3][0], box[3][1]), 4, (0, 0, 255), 12)
	image_show(delete, "drawn c")


	#################################################################################################################
	#################################################################################################################
	#################################################################################################################

	solution_output_image = dst.copy()
	image_show(dst,"")

	result_img = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 10)

	#################################################################################################################
	#################################################################################################################
	#################################################################################################################
	image_show(result_img, "thresholded warped image")
	#################################################################################################################
	#################################################################################################################
	#################################################################################################################

	kernel = np.ones((5, 5), np.uint8)
	result_img = cv2.blur(result_img, (2, 2), 0)
	result_img = cv2.dilate(result_img, kernel, iterations=1)
	result_img = cv2.erode(result_img, kernel, iterations=1)

	#################################################################################################################
	#################################################################################################################
	#################################################################################################################
	image_show(result_img, "thresholded warped image post erosion/dilasion etc")
	#################################################################################################################
	#################################################################################################################
	#################################################################################################################

	contours, hierarchy = cv2.findContours(result_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	return contours, hierarchy, result_img, solution_output_image