# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:52:58 2022

@author: sethy
"""
import random
#dice roll
print("dice is rolled")
cnt=0
cnt1=0
l=[1,2,3,4,5,6]
for x in range(100):
    y= random.choice(l)
    if (y%2==0):
        cnt= cnt+1
        print(y,"=even")
        
    else:
        cnt1 = cnt1+1
        print(y,"=odd")
        
print("total even",cnt)
print("total odd", cnt1)