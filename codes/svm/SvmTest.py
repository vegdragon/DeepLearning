import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np



style.use('ggplot')

class Support_Vector_Machine:
	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = { 1:'r', -1:'b' }
		if self.visualization:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1,1,1)

	# train
	def fit(self, data):
		self.data = data
		# { ||w||: [w,b] }
		opt_dict = {}

		transforms = [[1,1],
					  [-1,1],
					  [-1,-1],
					  [1,-1]]

		# cal max/min feature value, so we can figure out step_size
		# for gradient descend
		all_data = []
		for yi in self.data:
			for featureset in self.data[yi]:
				for feature in featureset:
					all_data.append(feature)

		self.max_feature_value = max(all_data)
		self.min_feature_value = min(all_data)
		all_data = None

		# support vectors yi(xi.w+b) = 1
		# if we close enough to the sv, we can stop, ex. 1.01
		step_sizes = [self.max_feature_value * 0.1,
					  self.max_feature_value * 0.01,
					  # point of expense:
					  self.max_feature_value * 0.001]

		# extremely expensive
		b_range_multiple = 2

		# we don't need to take as small of steps
		# with b as we do w
		b_multiple = 5

		# cal w size
		latest_optimum = self.max_feature_value*10

		for step in step_sizes:
			w = np.array([latest_optimum, latest_optimum])
			
			# we can do this because convex
			optimized = False
			while not optimized:
				for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
								   self.max_feature_value*b_range_multiple, 
								   step*b_multiple):
					for transformation in transforms:
						w_t = w*transformation
						found_option = True
						# weakest link in the SVM fundamentally
						# SMO attempts to fix this a bit
						# yi(xi.w+b) >= 1
						# 
						# ### add a break here later...
						for i in self.data:
							for xi in self.data[i]:
								yi=i
								# iterate all 
								if not yi*(np.dot(w_t,xi)+b) >= 1:
									found_option = False

						if found_option:
							opt_dict[np.linalg.norm(w_t)] = [w_t, b]

				if w[0] < 0:
					# we just passed the local minimum
					optimized = True
					print('optimized a step.')
				else:
					# move on a step further
					# w = [5, 5]
					# step = 1
					# w - step = [4, 4]
					w = w - step

			norms = sorted([n for n in opt_dict])
			# opt_dict is ||w|| : [w,b]
			# SVM target is to find minimum ||w||, so we take the first item as our choice
			opt_choice = opt_dict[norms[0]]
			self.w = opt_choice[0]
			self.b = opt_choice[1]
			# switch to a smaller step, to be more precise to the optimum
			latest_optimum = opt_choice[0][0]+step*2

		for i in self.data:
			for xi in self.data[i]:
				yi = i
				print(xi, ":", yi*(np.dot(self.w, xi)+self.b))




	def predict(self, features):
		# sign( x.w+b )
		classification = np.sign(np.dot(np.array(features), self.w) + self.b)
		if classification!=0 and self.visualization:
			self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])

		return classification

	def visualize(self):
		[[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

		# hyperplane = x.w+b
		# v = x.w+b
		# psv = 1  (positive SV)
		# nsv = -1 (negative SV)
		# dec = 0  (decision boundary)
		def hyperplane(x,w,b,v):
			return (-w[0]*x-b+v) / w[1]

		datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
		hyp_x_min = datarange[0]
		hyp_x_max = datarange[1]

		# (w.x+b) = 1
		# positive support vector hyperplane
		psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
		psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
		self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

		# (w.x+b) = -1
		# negative support vector hyperplane
		psv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
		psv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
		self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])

		# (w.x+b) = 0
		# negative support vector hyperplane
		psv1 = hyperplane(hyp_x_min, self.w, self.b, 0)
		psv2 = hyperplane(hyp_x_max, self.w, self.b, 0)
		self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])
		plt.show()

data_dict = { 
	-1:np.array([
		[1,7],
		[2,8],
		[3,8],
		]),
	1:np.array([
		[5,1],
		[6,-1],
		[7,3],
		])
	}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)
svm.visualize()

