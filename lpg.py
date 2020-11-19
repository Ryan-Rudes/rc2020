from tensorflow.keras.layers import LSTM, Dense, Layer
from tensorflow.keras.models import Model

class EmbeddingLayer(Layer):
  def __init__(self):
    super(EmbeddingLayer, self).__init__()
    self.projection_layer(16, activation = 'relu')
    self.output_layer = Dense(1, activation = 'relu')

  def call(self, inputs):
    x = self.projection_layer(inputs)
    out = self.output_layer(x)
    return out

class LPG(Model):
  def __init__(self):
    super(LPG, self).__init__()
    self.embedding = EmbeddingLayer()
    self.lstm = LSTM(256, activation = 'relu', go_backwards = True, input_shape = (None, 6))

  def call(self, r, d, gamma, pi, y0, y1):
    phi0 = self.embedding(y0)
    phi1 = self.embedding(y1)
    out = self.lstm([r, d, gamma, pi, phi0, phi1])
    return out
