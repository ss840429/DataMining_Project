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
		# self.train = self.train[800:]
		self.train_rows = len(self.train.index)
		self.attribute_name = list(self.train)
		self.attribute_name.remove('quality')
		self.pA = []
		self.confusion_matrix = {}

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
		# test = test[:800]
		self.confusion_matrix = { 'TP' : 0, 'TN' : 0, 'FP' : 0, 'FN' : 0 }
		self.pA = []
		print(self.attribute_name)
		for idx, row in test.iterrows():
			positive = len(self.train[self.train['quality']==1].index)/self.train_rows			## P(X|NO )P(NO )
			negative = len(self.train[self.train['quality']==0].index)/self.train_rows			## P(X|YES)P(YES)

			for attr in self.attribute_name:
				if test[attr].dtype == np.float64:
					fp, tp = self.normalDistribution(attr, row)
				else:
					fp, tp = self.discreteDistribution(attr, row, m_estimate)

				positive *= tp
				negative *= fp 

			if positive > negative:
				if row.get('quality') == 1:
					self.confusion_matrix['TP'] += 1
				else:
					self.confusion_matrix['FP'] += 1
			else:
				if row.get('quality') == 1:
					self.confusion_matrix['FN'] += 1
				else:
					self.confusion_matrix['TN'] += 1

			self.pA.append((positive, row.get('quality')))
		
		self.pA = sorted(self.pA, key = lambda x : x[0]) 	# Sort by positive 	
		print(self.pA)

	def showConfusionMatrix(self):
		print( '--------------------------------' )
		print( '|R \ P|     +     |     -      |' )
		print( '--------------------------------' )
		print( '|  +  |    {:-3d}    |    {:-3d}     |'.format(self.confusion_matrix['TP'], self.confusion_matrix['FN']) )
		print( '--------------------------------' )
		print( '|  -  |    {:-3d}    |    {:-3d}     |'.format(self.confusion_matrix['FP'], self.confusion_matrix['TN']) )
		print( '--------------------------------' )

		precision = self.confusion_matrix['TP'] / (self.confusion_matrix['TP']+self.confusion_matrix['FP'])
		recall    = self.confusion_matrix['TP'] / (self.confusion_matrix['TP']+self.confusion_matrix['FN'])
		fmeasure = 2*precision*recall / (precision+recall)

		print()
		print( 'precision : {}'.format(precision))
		print( 'recall    : {}'.format(recall))
		print( 'F-measure : {}'.format(fmeasure))

	def plotROC(self):
		
		matrix = { 'TP' : self.confusion_matrix['TP']+self.confusion_matrix['FN'],'TN' : 0,'FP' : self.confusion_matrix['FP']+self.confusion_matrix['TN'], 'FN' : 0 }
		TPR = [matrix['TP']/(matrix['TP']+matrix['FN'])]
		FPR = [matrix['FP']/(matrix['FP']+matrix['TN'])]

		for idx, p in enumerate(self.pA[1:]):
			
			if self.pA[idx][1] == 0:
				matrix['FP'] -= 1
				matrix['TN'] += 1
			else :
				matrix['TP'] -= 1
				matrix['FN'] += 1

			tpr = matrix['TP']/(matrix['TP']+matrix['FN'])
			fpr = matrix['FP']/(matrix['FP']+matrix['TN'])
			TPR.append(tpr)
			FPR.append(fpr)

		plt.title('ROC Curve')
		plt.xlabel( 'FPR' )
		plt.ylabel( 'TPR' )
		plt.plot(FPR, TPR, color='red', label='Real')
		plt.plot([0,1], [0,1], label='Random')
		plt.show()

		# pass





bf = BayesClassifier('wineQuality_Class1.csv')
bf.predict('wineQuality_Class1_Train.csv', m_estimate = None)
bf.showConfusionMatrix()
bf.plotROC()

# bg.predict('wineQuality_Class1_Test.csv', m_estimate = True)
# bf.showConfusionMatrix()
# bf.plotROC()


