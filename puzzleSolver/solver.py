def feasible_move(board, coordinate, number):
	"""
	A function that determines if the placement of a number onto the grid is feasible or not. Returns a boolean value.
	:param board: List of lists. Each sublist is a representation of a row in the board.
	:param coordinate: The coordinate of the grid position.
	:param number: Int. The number to be placed on the grid.
	:return: Boolean.
	"""
	x, y = coordinate

	#  row check
	for i in range(9):
		if board[x][i] == number and y != i:
			return False

	#  column check
	for i in range(9):
		if board[i][y] == number and x != i:
			return False

	row = (x // 3) * 3
	col = (y // 3) * 3

	#  square check
	for i in range(row, row + 3):
		for j in range(col, col + 3):
			if board[i][j] == number and (i, j) != coordinate:
				return False

	return True


def solver(board):
	"""
	Solves the board in place using backtracking methods.
	:param board: List.
	:return: None.
	"""
	blank_cell = find_blank(board)
	if not blank_cell:
		return True
	else:
		row, col = blank_cell

	for i in range(1, 10):
		if feasible_move(board, (row, col), i):
			board[row][col] = i

			if solver(board):
				return True

			board[row][col] = 0

	return False


def find_blank(board):
	"""
	Finds the first blank position in the board.
	:param board:
	:return: Coordinate of the first blank cell if it exists, otherwise returns False.
	"""
	for i in range(9):
		for j in range(9):
			if board[i][j] == 0:
				return i, j
	return False
