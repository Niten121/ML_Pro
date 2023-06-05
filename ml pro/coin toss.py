# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:03:57 2022

@author: sethy
"""
import random
import matplotlib.pyplot as plt
cnt=0
cnt1=0
toss=['h','t']

for y in range(100):
    y = random.choice(toss)
    if (y =='h'):
        cnt= cnt+1
        # print(y,"head")
        
    else:
        cnt1 = cnt1+1
        # print(y,"tail")
        
print("total head",cnt)
print("total tail", cnt1)

x=toss
y=[cnt,cnt1]
plt.bar(x,y,width=0.2)
plt.show()
