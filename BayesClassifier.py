import pandas
import math
import numbers
from math import e as exp
from math import pi as pi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class BayesClassifier:
	def __init__(self, train_set) :
		self.train = pandas.read_csv(train_set)
		self.train_rows = len(self.train.index)
		self.attribute_name = list(self.train)
		self.pA = []

	def normalDistribution(self, attribute_name, testdata):
		fmean = self.train[self.train['quality']==0][attribute_name].mean()
		fvar  = self.train[self.train['quality']==0][attribute_name].var()

		fp = (exp**(-(testdata.get(attribute_name)-fmean)**2/(2*fvar)))/(2*pi*fvar)**(1/2)
		
		pmean = self.train[self.train['quality']==1][attribute_name].mean()
		pvar  = self.train[self.train['quality']==1][attribute_name].var()

		tp = (exp**(-(testdata.get(attribute_name)-pmean)**2/(2*pvar)))/(2*pi*pvar)**(1/2)		

		return fp, tp		

	def discreteDistribution(self, attribute_name, testdata, m_estimate = None):
		ftotal = self.train[self.train['quality']==0]
		fcount = ftotal[ftotal[attribute_name]==testdata.get(attribute_name)]

		fp = len(fcount.index)/len(ftotal.index)
		
		ttotal = self.train[self.train['quality']==1]
		tcount = ttotal[ttotal[attribute_name]==testdata.get(attribute_name)]

		tp = len(tcount.index)/len(ttotal.index)
		
		return fp, tp

	def predict(self, test_set, m_estimate = None):
		test = pandas.read_csv(test_set)
		confusion_matrix = { 'TP' : 0, 'TN' : 0, 'FP' : 0, 'FN' : 0 }
		self.pA = []
		
		for idx, row in test.iterrows():
			positive = len(self.train[self.train['quality']==1].index)/self.train_rows
			negative = len(self.train[self.train['quality']==0].index)/self.train_rows

			for attr in test:
				if test[attr].dtype == np.float64:
					fp, tp = self.normalDistribution(attr, row)
				else:
					fp, tp = self.discreteDistribution(attr, row, m_estimate)

				positive *= tp
				negative *= fp 

			if positive > negative:
				if row.get('quality') == 1:
					confusion_matrix['TP'] += 1
				else:
					confusion_matrix['FP'] += 1
			else:
				if row.get('quality') == 1:
					confusion_matrix['FN'] += 1
				else:
					confusion_matrix['TN'] += 1

			self.pA.append(positive)
			

	def plotROC(self):
		pass





bf = BayesClassifier('wineQuality_Class2_Train.csv')
bf.predict('wineQuality_Class2_Test.csv', m_estimate = None)




