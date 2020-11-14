from objects import *
import numpy as np

class Map(object):
	def __init__(self, blockers, size, objects, max_steps):
		self.blockers = blockers
		self.size = size
		self.objects = objects
		self.max_steps = max_steps
		
class Dense(Map):
	def __init__(self):
		size = (11, 11)
		blockers = np.zeros(size)
		max_steps = 500
		objects = [*[BlueObject(1, 0, 0.05) for i in range(2)],
				   RedObject(-1, 0.5, 0.1),
				   YellowObject(-1, 0, 0.5)]
		
		super(Dense, self).__init__(blockers, size, objects, max_steps)
		
class Sparse(Map):
	def __init__(self):
		size = (13, 13)
		blockers = np.zeros(size)
		max_steps = 50
		objects = [BlueObject(1, 1, 0),
				   RedObject(-1, 1, 0)]
		
		super(Sparse, self).__init__(blockers, size, objects, max_steps)
		
class LongHorizon(Map):
	def __init__(self):
		size = (11, 11)
		blockers = np.zeros(size)
		max_steps = 1000
		objects = [*[BlueObject(1, 0, 0.01) for i in range(2)],
				   *[RedObject(-1, 0.5, 1) for i in range(2)]]
		
		super(LongHorizon, self).__init__(blockers, size, objects, max_steps)
		
class LongerHorizon(Map):
	def __init__(self):
		size = (7, 9)
		blockermap = """
		000010000
		000000000
		000010000
		000111000
		000010000
		000000000
		000010000
		""".strip()
		blockers = np.array([[int(tile) for tile in row.replace('\t', '')] for row in blockermap.split('\n')]).T
		max_steps = 1000
		objects = [*[BlueObject(1, 0.1, 0.01) for i in range(2)],
				   *[RedObject(-1, 0.8, 1) for i in range(5)]]
		
		super(LongerHorizon, self).__init__(blockers, size, objects, max_steps)
		
class LongDense(Map):
	def __init__(self):
		size = (11, 11)
		blockermap = """
		00000000000
		00000100000
		00000100000
		00000100000
		11011111011
		00000100000
		00000100000
		00000000000
		00000100000
		00000100000
		00000100000
		""".strip()
		blockers = np.array([[int(tile) for tile in row.replace('\t', '')] for row in blockermap.split('\n')]).T
		max_steps = 2000
		objects = [BlueObject(1, 0, 0.005) for i in range(4)]
		
		super(LongDense, self).__init__(blockers, size, objects, max_steps)