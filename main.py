from computerVision.gridImagesCV import splitter, corner_list_final, interior_contours, corners_finder
from digitClassifier.predict import computer_predict_array
from puzzleSolver.solver import solver
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def main_solver(image_path):
	"""
	The solver function. Pass this function the file path of the image that you want to solve.
	:param image_path: String
	:return: None
	"""
	grid_numbers = []

	grid_img_array, solution_image = splitter(image_path)

	empty_cell_index = []

	for x in range(len(grid_img_array)):
		try:
			grid_numbers.append(int(computer_predict_array(grid_img_array[x])))
		except:
			grid_numbers.append(0)
			empty_cell_index.append(x)

		print(f'[{x * "="}>{(80-x) * "-"}]')

	board = np.transpose(np.reshape(np.asarray(grid_numbers), (9, 9))).tolist()

	solver(board)


	#noinspection PyTypeChecker
	flatten_board = np.transpose(board)
	flatten_board = [item for sublist in flatten_board for item in sublist]

	cell = corners_finder(interior_contours[0])
	cell_width = cell[1][0] - cell[0][0]
	cell_height = cell[1][1] - cell[0][1]
	img_fraction = 0.5
	font_size = 1
	font = ImageFont.truetype('arial', font_size)
	img = Image.fromarray(solution_image)
	draw = ImageDraw.Draw(img)

	while font.getsize('0')[0] <  cell_width * img_fraction and font.getsize('0')[1] <  cell_height * img_fraction:
		font = ImageFont.truetype("arial.ttf", font_size)
		font_size += 1

	for i in empty_cell_index:
		draw.text((corner_list_final[i][0] + cell_width // 2 - font.getsize('0')[0] // 2, corner_list_final[i][1] + cell_height // 7), str(flatten_board[i]), (255,0,0), font=font)
	img.show()
	img.save('solution.png')

main_solver(r'C:\Users\Tom\PycharmProject\sudokuSolver\computerVision\assets\test10.jpg')