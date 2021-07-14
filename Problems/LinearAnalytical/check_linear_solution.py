# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 17:42:29 2021

@author: aa
"""

Lx=1.0
Ly=1.5
hy=-0.06
lam=3.0
mu=1.0

b = hy*(lam+2*mu)/(4*mu*(lam+mu))
a = -(lam/(lam+2*mu))*b

print(f'strain x = {a},strain y = {b}')
print(f'disp x = {a*Lx},disp y = {b*Ly}')