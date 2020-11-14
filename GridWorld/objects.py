from colors import blue, red, yellow
import numpy as np

class Object(object):
	def __init__(self, r, e_term, e_respawn):
		self.r = r
		self.e_term = e_term
		self.e_respawn = e_respawn
		
		self.hidden = False
		
	def collect(self):
		if self.hidden:
			return 0, False
		else:
			self.hidden = True
			return self.r, np.random.random() < self.e_term
		
	def update(self):
		if self.hidden and np.random.random() < self.e_respawn:
			self.hidden = False
			
class BlueObject(Object):
	def __init__(self, r, e_term, e_respawn):
		super(BlueObject, self).__init__(r, e_term, e_respawn)
		self.color = blue
		
class RedObject(Object):
	def __init__(self, r, e_term, e_respawn):
		super(RedObject, self).__init__(r, e_term, e_respawn)
		self.color = red
		
class YellowObject(Object):
	def __init__(self, r, e_term, e_respawn):
		super(YellowObject, self).__init__(r, e_term, e_respawn)
		self.color = yellow