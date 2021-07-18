hx=-0.06
lam=3.0
mu=1.0
a = hx*(lam+2.0*mu)/(4.0*mu*(lam+mu))
b = -lam*a/(lam+2.0*mu)

print('ux_max = ',a*1.0)
print('uy_max = ',b*1.5)
