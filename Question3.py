"""
Author: Joshua Issa
SID: 20783023
Course: PHYS 375
PS1 - Question 3

Collaboration Declaration:
I discussed the assignment with Andy Heremez.
"""


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from matplotlib import rc
rc('text', usetex=True)

def read_data(file):
    """Reads in the W22_ps1_orbit.dat file.
    INPUTS:
        file - Expected to be W22_ps1_orbit.dat.

    OUTPUTS:
        df - A DataFrame of the information in the txt file.
    """

    df = pd.read_csv(file, sep=' ', header=None)

    df = df.rename({0: 'Orbital Phase', 1: 'Star 1 Vel', 2: 'Star 2 Vel', 3: 'App Mag'}, axis=1)

    return df


def part_a(orbit_data, savepath=''):
    """Plot the radial velocities over time for a full orbit.
    INPUTS:
        orbit_data - The DataFrame containing the orbit data.
        savepath - The path to save the plot to. If no path is given, the plot is displayed.
    OUTPUTS:
        The plot for PS1 Question 3a.
    """

    plt.plot(orbit_data['Orbital Phase'], orbit_data['Star 1 Vel'], label = 'Star 1 Velocity')
    plt.plot(orbit_data['Orbital Phase'], orbit_data['Star 2 Vel'], label = 'Star 2 Velocity')

    plt.title('Radial Velocities over a Full Orbit')

    plt.ylabel(r'Radial Velocity [$km/s$]')
    plt.xlabel('Orbital Phase (over 50 days)')

    plt.legend()
    if savepath == '': plt.show()
    else: plt.savefig(savepath)

    plt.close('all')

def part_b(orbit_data, savepath=''):
    """Calculate msin^3i based on the radial velocity plot.
    INPUTS:
        orbit_data - The DataFrame containing the orbit data.
        savepath - The path to save the plot to. If no path is given, the plot is displayed.
    OUTPUTS:
        (1) Reconstructed velocities plot to verify correct amplitude was determined.
        (2) Apparent magnitude over time plot.
    """

    #period of 50 days
    P, G = 50*24*60*60, 6.67e-11

    v_1 = orbit_data['Star 1 Vel'].to_numpy()
    v_2 = orbit_data['Star 2 Vel'].to_numpy()
    t = orbit_data['Orbital Phase'].to_numpy()

    #assuming the wave is some generic sine wave of form v(t) = Asin(kt) + b

    #index 0 is t=0 --> v(0) = b
    b_1 = v_1[0]
    b_2 = v_2[0]

    #it follows that v(t) = A + b when sin(kt) = 1, which occus at the maximum of the wave
    A_1 = np.max(v_1) - b_1
    A_2 = np.max(v_2) - b_2

    #since the period of the wave is from 0 to 1 for both, k is the same for both.
    k = 2*np.pi/(t[-1]-t[0])

    #reconstruction for the sake of verification
    re_v_1 = -A_1 * np.sin(k*t) + b_1
    re_v_2 = A_2 * np.sin(k*t) + b_2

    fig, axs = plt.subplots(1,2, figsize=(10,5))

    #if the plot is linear, then its right
    axs[0].plot(v_1, re_v_1)
    axs[0].set_title('Star 1 Velocity')

    axs[1].plot(v_2, re_v_2, color='orange')
    axs[1].set_title('Star 2 Velocity')

    left = np.min(v_1) if np.min(v_1) < np.min(v_2) else np.min(v_2)
    right = np.max(v_1) if np.max(v_1) > np.max(v_2) else np.max(v_2)

    left -= 5
    right += 5

    axs[0].set_xlim(left,right)
    axs[1].set_xlim(left,right)
    axs[0].set_ylim(left,right)
    axs[1].set_ylim(left,right)

    fig.suptitle('Comparing Reconstructed Velocity Curves')

    fig.supylabel(r'Reconstructed Velocity [$km/s$]')
    fig.supxlabel(r'Original Velocity [$km/s$]')

    if savepath == '': plt.show()
    else: plt.savefig(savepath.replace('.png', '_reconstruction.png'))

    plt.close('all')


    plt.plot(t, orbit_data['App Mag'])
    plt.title('Apparent Magnitude over a Full Orbit')
    plt.ylabel(r'Apparent Magnitude ($m$)')
    plt.xlabel('Orbital Phase (over 50 days)')

    plt.gca().invert_yaxis()

    if savepath == '': plt.show()
    else: plt.savefig(savepath.replace('.png', '_magnitude.png'))

    plt.close('all')


    #Now that it's verified that the velocities were successfully extracted, we can figure out msin^3i

    #According to p. 326 of Ryden, "the amplitudes of the two radial velocity curves yield va sini and vb sini"

    #from equation 13.67 from Ryden : the amplitudes are the velocities, divided by 1000 to get m/s
    ma_mb_sin3i = P*np.power(A_1*1000 + A_2*1000, 3)/2/np.pi/G

    #from Equation 13.62 we know that the ratio of velocities gives the ratio of the masses
    ma_mb = A_2/A_1

    #from Equation 13.74 we find mb given ma/mb
    mbsin3i = P*np.power(A_1*1000, 3)*np.power(1 + ma_mb, 2)/2/np.pi/G

    #Equation 13.62 implies that if you know mbsin3i you can know masin3i
    masin3i = ma_mb * mbsin3i

    print(f'masin3i {masin3i}kg')
    print(f'mbsin3i {mbsin3i}kg')
    #verification: I'm right
    print(f'Verification {ma_mb_sin3i}kg {masin3i + mbsin3i}kg')

    return masin3i, mbsin3i

def part_c(orbit_data, savepath=''):
    """Plot the logarithm of L/Lo where Lo is when both stars are visible.
    INPUTS:
        orbit_data - The DataFrame containing the orbit data.
        savepath - The path to save the plot to. If no path is given, the plot is displayed.
    OUTPUTS:
        The plot for PS1 Question 3c.
    """

    #looking at the magnitude over time plot, it's pretty clear to understand that when the stars overlap
    #they diminish in their magnitude, and so anytime they aren't (the minimum magnitude) is Lo

    app_mag = orbit_data['App Mag'].to_numpy()

    min_mag = np.min(app_mag)

    #based on Equation 13.39 from Ryden
    L_L0 = np.power(10, 0.4*(min_mag - app_mag))

    plt.plot(orbit_data['Orbital Phase'], np.log10(L_L0))
    plt.title(r'log($L/L_0$) over a Full Orbit')
    plt.xlabel('Orbital Phase (over 50 days)')
    plt.ylabel(r'log($L/L_0$)')

    if savepath == '': plt.show()
    else: plt.savefig(savepath)

    plt.close('all')

def part_d_e(Ma, Mb):

    #from the back cover of Ryden
    sb = 5.67e-8

    #from Ryden p. 576
    Mo = 1.989e30
    Lo = 3.839e26
    Ro = 6.955e8

    a_ratio = Ma/Mo
    b_ratio = Mb/Mo
    print(f'Solar Mass {Mo}kg')
    print('M_a/Mo', a_ratio)
    print('M_b/Mo', b_ratio)

    #therefore, the masses are less than 1.66Mo

    #we use Equation 13.78 to get the luminosity
    L_a = Lo*0.35*np.power(a_ratio, 2.62)
    L_b = Lo*0.35*np.power(b_ratio, 2.62)

    #we use Equation 13.77 to get the radius
    R_a = Ro*1.06*np.power(a_ratio, 0.945)
    R_b = Ro*1.06*np.power(b_ratio, 0.945)

    #we can use Equation 13.52 to get the temperature
    T_a = np.power(L_a/4/np.pi/np.power(R_a, 2)/sb, 0.25)
    T_b = np.power(L_b/4/np.pi/np.power(R_b, 2)/sb, 0.25)

    print(f'T_a {T_a}K')
    print(f'T_b {T_b}K')

    print(f'R_a {R_a}m')
    print(f'R_b {R_b}m')

def main():

    here = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(here, 'W22_ps1_orbit.dat')

    orbit_data = read_data(file)

    a_plot = os.path.join(here, 'PS1-Q3a.png')
    part_a(orbit_data, a_plot)

    b_plot = os.path.join(here, 'PS1-Q3b.png')
    Ma, Mb = part_b(orbit_data, b_plot)

    c_plot = os.path.join(here, 'PS1-Q3c.png')
    part_c(orbit_data, c_plot)

    #note: since the stars are eclipsing --> i must be near 90 (Ryden p. 329)
    #therefore, we can say that sin3i approximately == 1
    #might not be doing this correctly
    part_d_e(Ma, Mb)


if __name__ == '__main__':
    main()
