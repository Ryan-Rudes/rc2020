class Coord:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __eq__(self, coord):
		if isinstance(coord, Coord):
			return self.x == coord.x and self.y == coord.y
		elif isinstance(coord, tuple):
			return self.x == coord[0] and self.y == coord[1]
		else:
			raise Exception("Can only compare a 2D tuple or another Coord with a Coord")
		
	def __add__(self, coord):
		if isinstance(coord, Coord):
			return Coord(self.x + coord.x, self.y + coord.y)
		elif isinstance(coord, tuple):
			return Coord(self.x + coord[0], self.y + coord[1])
		else:
			raise Exception("Can only add a 2D tuple or another Coord to a Coord")
			
	def __iadd__(self, coord):
		if isinstance(coord, Coord):
			self.x += coord.x
			self.y += coord.y
		elif isinstance(coord, tuple):
			self.x += coord[0]
			self.y += coord[1]
		else:
			raise Exception("Can only add a 2D tuple or another Coord to a Coord")
		
	def __sub__(self, coord):
		if isinstance(coord, Coord):
			return Coord(self.x - coord.x, self.y - coord.y)
		elif isinstance(coord, tuple):
			return Coord(self.x - coord[0], self.y - coord[1])
		else:
			raise Exception("Can only subtract a 2D tuple or another Coord from a Coord")
		
	def __isub__(self, coord):
		if isinstance(coord, Coord):
			self.x -= coord.x
			self.y -= coord.y
		elif isinstance(coord, tuple):
			self.x -= coord[0]
			self.y -= coord[1]
		else:
			raise Exception("Can only subtract a 2D tuple or another Coord from a Coord")
			
	def __str__(self):
		return '({x}, {y})'.format(x = self.x, y = self.y)