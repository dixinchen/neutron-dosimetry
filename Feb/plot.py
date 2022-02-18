#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 21:28:01 2022

@author: dchen
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

file = open('Book1.csv')
csvreader = csv.reader(file)
data = []
for i in csvreader:
    data.append(i)
data = np.array(data).T

x = np.linspace(0, 60, 60)
y = []
for i in range(len(data[1])):
    y.append(float(data[1][i]))  
    y[i] = y[i]*1e12 

f = plt.figure()
plt.plot(x, y, '.--')
plt.xlabel('Distance [cm]')
plt.ylabel('Dose [pGy]')
plt.title('50 protons at 169.23 MeV (physics: opt0)')