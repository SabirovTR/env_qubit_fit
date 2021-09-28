import cmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle


with open('STS_FROM_POWER, 30.11.20_raw_data.pkl', 'rb') as f:
    raw_data = pickle.load(f)

#распаковываем подмассивы
pow = raw_data['power [dBm]']
data = raw_data['data']
freq = (raw_data['Frequency [Hz]'])/1e+6


slice =9
abs_data = abs(data[slice])
#arg_data = [cmath.phase(data[2][i]) for i in range(len(freq))]
real_data = [data[slice][i].real for i in range(len(freq))]
imag_data = [data[slice][i].imag for i in range(len(freq))]
arg_data = np.unwrap([cmath.phase(data[slice][i]) for i in range(len(freq))])


def fit_abs(freq, freq_r, phi, gamma1, gamma2, abs_a, rabi_freq):
    t = (freq - freq_r)/gamma2
    r = ((gamma1/(2*gamma2))*1/((1 + t**2 + rabi_freq**2/(gamma1*gamma2))))
    return abs_a*np.sqrt(1 - 2*r*(np.cos(phi) - t*np.sin(phi)) + (r**2)*(1 + t**2))


def fit_arg(freq, freq_r, phi, gamma1, gamma2, arg_a, rabi_freq, tau):
    t = (freq - freq_r) / gamma2
    r = ((gamma1 / (2 * gamma2)) * 1 / ((1 + t ** 2 + rabi_freq ** 2 / (gamma1 * gamma2))))
    return arg_a + (-2*np.pi*freq*tau) + np.arctan(-(t*np.cos(phi) + np.sin(phi))/(1/r + t*np.sin(phi) - np.cos(phi)))

'''
tau_bound = -2*np.pi*(arg_data[-1] - arg_data[0])/(freq[-1] - freq[0])
print(tau_bound)
popt, pcov = curve_fit(fit_arg, freq, arg_data, bounds=((freq[0], 0.0, 0.0, 0.0, 0.0, 0.0, -tau_bound), (freq[-1], 2*np.pi, np.inf, np.inf, np.inf, np.inf, tau_bound)), maxfev=1000)
#popt, pcov = curve_fit(fit_arg, freq, abs_data, maxfev=10000)
print(popt)
perr = np.sqrt(np.diag(pcov))
print(-2*np.pi*arg_data[0]/freq[0])

arg_data_opt = fit_arg(freq, *popt)

plt.plot(freq, arg_data)
plt.plot(freq, fit_arg(freq, *popt))
plt.show()

plt.plot(real_data, imag_data)
plt.show()

phs_data = arg_data + 2*np.pi*freq*popt[-1] - popt[-3]

plt.plot(freq, phs_data)
plt.show()

plt.scatter(abs_data*np.sin(phs_data), abs_data*np.cos(phs_data), 2)
plt.show()

plt.scatter(phs_data, abs_data, 2)
plt.show()
'''
popt, pcov = curve_fit(fit_abs, freq, abs_data, bounds=((freq[0], 0.0, 0.0, 0.0, min(abs_data), 0.0), (freq[-1], 2*np.pi, np.inf, np.inf, 1.2*max(abs_data), np.inf)), maxfev=1000)
#popt, pcov = curve_fit(fit_arg, freq, abs_data, maxfev=10000)
print(popt)
perr = np.sqrt(np.diag(pcov))
print(-2*np.pi*arg_data[0]/freq[0])

abs_data_opt = fit_abs(freq, *popt)

plt.plot(freq, abs_data)
plt.plot(freq, fit_abs(freq, *popt))
plt.show()

plt.subplot(1, 2, 2)
plt.plot(freq, abs_data)
plt.plot(freq, abs_data_opt)

plt.subplot(1, 2, 1)
plt.plot(freq, arg_data)
plt.plot(freq, arg_data_opt)

plt.show()

plt.plot(arg_data, abs_data)
plt.plot(arg_data_opt, abs_data_opt)
plt.show()
