from PIL import Image
import pytesseract
import pandas as pd


def computer_predict_array(image_array):
	"""
	To be used to implement Tesseract OCR on grid elements.
	:param image_array: Numpy array representing the image.
	:return:
	"""
	pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
	new_image_array = Image.fromarray(image_array)
	data = pytesseract.image_to_data(new_image_array, lang='eng',
									 config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789',
									 output_type='data.frame')
	return data
	# data = pytesseract.image_to_data(new_image_array, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789', output_type='data.frame')
	# guess = data[data['conf'] != -1]['text'].item()
	# confidence = data[data['conf'] != -1]['conf'].item()
	#
	# if confidence > 90:
	# 	return guess
	# else:
	# 	return ''
