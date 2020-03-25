import random
import math
class Matrix():
	def __init__(self,num_rows,num_cols):
		self.rows = num_rows
		self.cols = num_cols
		self.data = []
		for i in range(self.rows):
			self.data.append([])
			for j in range(self.cols):
				self.data[i].append(0)

	def randomize(self): 
		for i in range(self.rows):
			for j in range(self.cols):
				self.data[i][j] = random.uniform(-1,1)

	@staticmethod 
	def subtract(a,b):
		result = Matrix(a.rows, b.cols)
		for i in range(result.rows):
			for j in range(result.cols):
				result.data[i][j] = a.data[i][j] - b.data[i][j]
		return result

	def plus(self, x): 
		if (type(x)==Matrix):
			for i in range(self.rows):
				for j in range(self.cols):
					self.data[i][j] += x.data[i][j]
		else:
			for i in range(self.rows):
				for j in range(self.cols):
					self.data[i][j] += x
	@staticmethod
	def transpose(mat):
		result = Matrix(mat.cols, mat.rows)
		for i in range(mat.rows):
			for j in range(mat.cols):
				result.data[j][i] += mat.data[i][j]
		return result

	@staticmethod 
	def multiply(n,m):
		if (type(m)==Matrix):
				result = Matrix(n.rows, m.cols)
				for i in range(result.rows):
					for j in range(result.cols):
						sum = 0
						for k in range(n.cols):
							sum += n.data[i][k] * m.data[k][j]
						result.data[i][j] = sum;
				return result

	def scaler(self,n):
		if (type(n) !=Matrix):
			for i in range(self.rows):
				for j in range(self.cols):
					self.data[i][j] *= n
			
	@staticmethod
	def toArray(mat):
		output_arr = []
		for i in range(mat.rows):
			for j in range(mat.cols):
				output_arr.append(mat.data[i][j])
		return output_arr
		
	@staticmethod
	def toMatrix(arr): 
		mat = Matrix(len(arr), 1)
		for i in range(len(arr)):
			mat.data[i][0] = arr[i]
		return mat 

	@staticmethod 
	def sigmoid(x):
		for i in range(x.rows):
			for j in range(x.cols):
				val = x.data[i][j]
				x.data[i][j] = 1 / ( 1 + math.exp(-val));
		return x

	@staticmethod
	def disgmoid(y): 
		for i in range(y.rows):
			for j in range(y.cols):
				val = y.data[i][j]
				y.data[i][j] = val * ( 1 - val)
		return y

class NueralNetwork:
	def __init__(self, inp, hid, out):
		self.input_nodes = inp
		self.hidden_nodes = hid
		self.output_nodes = out
		self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
		self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
		self.bias_h = Matrix(self.hidden_nodes, 1)
		self.bias_o = Matrix(self.output_nodes, 1) 
		self.weights_ih.randomize()
		self.weights_ho.randomize()
		self.bias_h.randomize()
		self.bias_o.randomize()
		self.learning_rate = 0.1

	def feedForward(self, arr):
		inputs = Matrix.toMatrix(arr)
		hidden = Matrix.multiply(self.weights_ih, inputs)
		hidden.plus(self.bias_h)
		hidden = Matrix.sigmoid(hidden)
		output = Matrix.multiply(self.weights_ho,hidden)
		output.plus(self.bias_o)
		output = Matrix.sigmoid(output)
		output = Matrix.toArray(output)
		return output

	def train(self, input_array,target_array):
		inputs = Matrix.toMatrix(input_array)
		hidden = Matrix.multiply(self.weights_ih, inputs)
		hidden.plus(self.bias_h)
		hidden = Matrix.sigmoid(hidden)
		outputs = Matrix.multiply(self.weights_ho,hidden)
		outputs.plus(self.bias_o)
		outputs = Matrix.sigmoid(outputs)     
		targets = Matrix.toMatrix(target_array)
		output_errors = Matrix.subtract(targets, outputs) # changed here not true 
		gradients = Matrix.disgmoid(outputs)
		gradients = Matrix.multiply(gradients, output_errors)
		gradients.scaler(self.learning_rate)
		hidden_T = Matrix.transpose(hidden)
		weight_ho_deltas = Matrix.multiply(gradients, hidden_T)
		self.weights_ho.plus(weight_ho_deltas)
		self.bias_o.plus(gradients)
		who_t = Matrix.transpose(self.weights_ho)
		hidden_errors = Matrix.multiply(who_t, output_errors)
		hidden_gradient = Matrix.disgmoid(hidden)
		hidden_gradient = Matrix.multiply(hidden_gradient, hidden_errors)
		hidden_gradient.scaler(self.learning_rate)
		inputs_T = Matrix.transpose(inputs)
		weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T)
		self.weights_ih.plus(weight_ih_deltas)
		self.bias_h.plus(hidden_gradient)

def main():
	input_data = [[0,0],[0,1],[1,0],[1,1]]
	target_data = [[0],[1],[1],[0]]
	nn = NueralNetwork(2,4,1)
	for i in range(30000):
			for i in range(4):
			#j = random.randint(0, 3)
				nn.train(input_data[i],target_data[i])
			#Adjust the learning Rate
			#Change to Random choose
			#More loops 
	print(nn.feedForward([0,0]))
	print(nn.feedForward([0,1]))
	print(nn.feedForward([1,0]))
	print(nn.feedForward([1,1]))

if __name__ == '__main__':
	main()
