import numpy as np
import scipy
import matplotlib.pyplot as plt 
import pandas as pd

number = 3

#--------------------------------------------------- Importing Data, defining constants

# Column Names
names = ['Time / s', 'Gauge Angle', 'Angle', 'Torque / Nm']

torsion1 = pd.read_csv('{} - Torsion.csv'.format(3), header=None, names=names, skiprows=4, skipfooter=81)
torsion2 = pd.read_csv('{} - Torsion.csv'.format(3), header=None, names=names, skiprows=16)

# Structural Constants
l = 0.076 # length in m
D = 0.006 # diameter in m

# Derived Structural constants
r =  D / 2
J = np.pi * r**4 / 2

def tau_func(T):
    return T * r / J

def gamma_func(theta):
    return theta * r / l

#--------------------------------------------------- First Dataset
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

#------------------------------- Raw Data Plot
ax1.plot(torsion1['Gauge Angle'], torsion1['Torque / Nm'])
ax1.set_title('a) Raw Data')
ax1.set_xlabel('Angle / Radians')
ax1.set_ylabel('Torque / Nm')
ax1.grid()

#------------------------------- Regression Plot
tau = tau_func(torsion1['Torque / Nm'] )
gamma = gamma_func(torsion1['Gauge Angle'])

# Plotting Stress/Strain, MPa v Radians
ax2.plot(gamma, tau*1E-6, label='Data', marker='x', linestyle='')

# Linear Regression
mod = scipy.stats.linregress(gamma, tau)
ax2.plot(gamma, (mod.intercept + mod.slope*gamma)*1E-6, label='Linear Fit')

# Getting G
G = mod.slope*1E-9
dG = 1.96*mod.stderr*1E-9
print('G = ', G, ' +/- ', dG, ' GPa')

# Formatting Plot
ax2.set_title('b) Stress over Strain Graph')
ax2.set_xlabel('Strain $\gamma$ / Radians')
ax2.set_ylabel('Stress $\\tau$ / MPa')
ax2.grid()
ax2.legend()
plt.tight_layout()
plt.show()

#--------------------------------------------------- Second Dataset

fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 10))

#------------------------------- Raw Data Plot
ax1.plot(torsion2['Angle'],  torsion2['Torque / Nm'] )

# formatting
ax1.set_title('a) Raw Data')
ax1.set_ylabel('Torque / Nm')
ax1.set_xlabel('Angle / Radians')
ax1.grid()

#------------------------------- Regression Plots
tau = tau_func(torsion2['Torque / Nm'] )
gamma = gamma_func(torsion2['Angle'])

# Skips this many datapoints at the start of the plots: cuts out zeroed datapoints.
start_index = 5
ax2.plot(gamma[start_index:], tau[start_index:]*1E-6, label='Stress/Strain Data')

#Initial Regression - Only regresses over the range [start_index:regress_index] - i.e., select the linear region here
regress_index = 20
res = scipy.stats.linregress(gamma[start_index:regress_index], tau[start_index:regress_index])
ax2.plot(gamma[start_index:regress_index], (gamma[start_index:regress_index]*res.slope + res.intercept)*1E-6, label='Linear Regression')

#Offset Regression
offset_float = 0.002
ax2.plot(gamma[start_index:regress_index+3]+offset_float, (gamma[start_index:regress_index+3]*res.slope + res.intercept)*1E-6, label='Offset Linear Regression')

#Yield Point - You have to input this and find the best index to use
yield_index = 22
ax2.plot(gamma[yield_index], tau[yield_index]*1E-6, label='Yield Stress', marker='o', linestyle='')

#Ultimate Point
ultimate_index = np.argmax(tau)
ax2.plot(gamma[ultimate_index], tau[ultimate_index]*1E-6, label='Ultimate Stress', marker='o', linestyle='')

print('Yield Stress = ', tau[yield_index]*1E-6,' MPa,  Ultimate Stress = ', tau[ultimate_index]*1E-6, ' MPa')

# Formatting
ax2.set_title('b) Annotated Stress/Strain Diagram')
ax2.set_xlabel('Strain $\gamma$ / Radians')
ax2.set_ylabel('Stress $\\tau$ / MPa')
ax2.grid()
ax2.legend()
plt.show()
