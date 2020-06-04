import numpy as np

f = 1
b = 0
f1 = lambda x:3*np.sign(2*np.pi*f*x + b)
data = np.linspace(-5,5,1000)
np.fft.fft(data)