
# coding: utf-8

# In[6]:

import pandas as pd 
import collections as c 
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls 
from plotly.graph_objs import * 
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm 
import sklearn.cross_validation import KFold


#loads the data 
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')


#creates new column from the loansdata dataframe where we strip out the % sign at the end 
clean_interest_rate = loansData['Interest.Rate'].map(lambda x: x.rstrip('%'))


#converts data into a float 
clean_interest_rate = clean_interest_rate.map(lambda x: float(x))

#turns data into a decimal by dividing by 100 
clean_interest_rate = clean_interest_rate.map(lambda x: x / 100)

#ensures the decimals only round to 4 digits
clean_interest_rate = clean_interest_rate.map(lambda x: round(x, 4))

#defines the loansdata interest rate dataframe as the cleaned up version of it
loansData['Interest.Rate'] = clean_interest_rate


#calls a new column to clean up the loan length dataframe and removes the word months 
clean_loan_length = loansData['Loan.Length'].map(lambda x: x.rstrip(' months'))

#converts the data to an integer
clean_loan_length = clean_loan_length.map(lambda x: int(x))

#creates a new dataframe and sets them equal 
loansData['Loan.Length'] = clean_loan_length

#creates a new fico score column
loansData['FICO.Score'] = [int(x.split('-')[0]) for x in loansData['FICO.Range']]

#create a histogram of the fico score 
plt.figure()
loansData['FICO.Score'].hist()
plt.show()


#defining the variables 
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

#dependent variable 
y = np.matrix(intrate).transpose()

#independent variables shaped as columns 
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

x = np.column_stack([x1, x2])

