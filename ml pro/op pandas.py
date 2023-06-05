# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:15:00 2022

@author: sethy
"""

'''pandas is a lib used for working in the data set
it has a function for analysiing , cleaning , exploring  and manippulatiob of the data .


pandas give answers like * is there any correlation between two or more cols
*the avg value
*max and min value

pandas are also able to delete rows that aee not relevant , or contain wrong data , like empty or null values
 this  is called cleaning of the data
'''
import pandas as pd
from pandas import DataFrame
mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}

myvar = pd.DataFrame(mydataset)

print(myvar)

