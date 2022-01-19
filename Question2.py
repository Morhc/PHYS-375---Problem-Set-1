"""
Author: Joshua Issa
SID: 20783023
Course: PHYS 375
PS1 - Question 2

Collaboration Declaration:
I discussed the assignment with Andy Heremez. We considered in-depth how to interpret Question 2b.
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from matplotlib import rc
rc('text', usetex=True)

def read_data(file):
    """Reads in the hipparcos.txt file.
    INPUTS:
        file - Expected to be hipparcos.txt.

    OUTPUTS:
        df - A DataFrame of the information in the txt file.
    """

    df = pd.read_csv(file, sep=' ', header=None)

    df = df.rename({0: 'Parallax', 1: 'V Mag', 2: 'B Mag', 3: 'I Mag'}, axis=1)

    #Since the parallax is given in milliarcseconds, I'm converting it to arcseconds.
    df['Parallax'] = df['Parallax'] / 1000

    return df

def absolute_magnitude(parallax, app_mag):
    """Calculates the absolute magnitude given the parallax and apparent magnitude.
    INPUTS:
        parallax - The parallax of the star.
        app_mag - The apparent magnitude of the star.

    OUTPUTS:
        abs_mag - The absolute magnitude of the star.
    """

    #Ryden equation 13.3 [gives distance in parsecs!]
    d = 1 / parallax

    #Ryden equation 13.23
    abs_mag = app_mag - 5*np.log10(d/10)

    return abs_mag

def empirical_temperature(B_V):
    """Calculates the temperature of the star according to the empirical temperature relation.
    INPUTS:
        B_V - The B-V colour index of the star.

    OUTPUTS:
        temp - The temperature of the star
    """

    #Ryden equation 13.36
    temp = 9000/(B_V + 0.93)

    return temp

def calc_lum_ratio(V, parallax):
    """Calculate Lv/Lo from the magnitude data.
    INPUTS:
        V - The V magnitude data.
        parallax - The parallax data.

    OUTPUTS:
        lum_ratio - The calculated Lv/Lo.
    """

    #get the absolute V magnitude
    Mv = absolute_magnitude(parallax, V)

    """
    My justification of using Mv rather than Mbol:
    PS1 - Q2b specifically says that we wamt to plot Lv/Lo not L/Lo. The implication is that we want
    specifically the luminosity from the V filter. Additionally it seems that obtaining the bolometric
    constant is outside the scope of this part. Technically, you could assume that since we are given
    B, V, and I they serve as a pseudo-bolometric range, but the specific mention of Lv makes me doubt
    the inclusion of B and I. Finally, I chose to use Mv rather than V (or absolute vs apparent) because
    the given bolometric magnitude of the Sun of 4.74 is absolute not apparent.
    """

    #based on Equation 13.39 from Ryden
    lum_ratio = np.power(10, 0.4*(4.74 - Mv))

    return lum_ratio

def calc_lum(R, T):
    """Calculate the luminosity according to Equation 13.51.
    INPUTS:
        R - The radius of the star.
        T - The temperature of the star.
    OUTPUTS:
        lum - The luminosity of the star.
    """

    #the Stefan-Boltzmann constant from Ryden p. 144
    sb = 5.67e-8

    #Equation 13.51 from Ryden p. 321
    lum = 4*np.pi*np.power(R, 2)*sb*np.power(T, 4)

    return lum

def part_a(hipparcos, savepath=''):
    """Plots absolute V magnitude (Mv) against B-V with brighter stars at the top and bluer stars on the left.
    INPUTS:
        hipparcos - The DataFrame containing the magnitudes.
        savepath - The path to save the plot to. If no path is given, the plot is displayed.

    OUTPUTS:
        The plot for PS1 Question 2a.
    """

    V = hipparcos['V Mag'].to_numpy()
    B = hipparcos['B Mag'].to_numpy()
    parallax = hipparcos['Parallax'].to_numpy()

    #Get the absolute V magnitude
    Mv = absolute_magnitude(parallax, V)

    #Get the colour index with Ryden equation 13.35
    B_V = B - V

    plt.scatter(B_V, Mv, edgecolors='black')

    #According to p. 316-317 of Ryden, if a star has B-V < 0 it is bluer than Vega and
    #if it has B-V > 0 it is redder than Vega. This means that it is justified to leave the
    #x-axis running from 0 --> max(B-V)

    #The magnitude scale is backwards to how we feel (p. 311 of Ryden), so the brighter a star is
    #the lower its magnitude is (p. 310 of Ryden). Therefore, the y-axis needs to be flipped.
    plt.gca().invert_yaxis()

    plt.xlabel(r'$B-V$')
    plt.ylabel(r'$M_V$')
    plt.title(r'$M_V$ vs $B-V$')


    if savepath == '': plt.show()
    else: plt.savefig(savepath)

    plt.close('all')

def part_b(hipparcos, savepath=''):
    """Plots log(Lv/Lo) as a function of log(T).
    INPUTS:
        hipparcos - The DataFrame containing the magnitudes.
        savepath - The path to save the plot to. If no path is given, the plot is displayed.

    OUTPUTS:
        The plot for PS1 Question 2b.
    """

    V = hipparcos['V Mag'].to_numpy()
    B = hipparcos['B Mag'].to_numpy()
    parallax = hipparcos['Parallax'].to_numpy()

    #Get the colour index
    B_V = B - V

    #Get the temperatures
    temp = empirical_temperature(B_V)

    #Get Lv/Lo
    lum_ratio = calc_lum_ratio(V, parallax)

    plt.scatter(np.log10(temp), np.log10(lum_ratio), edgecolors='black')

    plt.xlabel(r'$log$($T [K]$)')
    plt.ylabel(r'$log$($L_V$/$L_\odot\/$)')

    plt.title(r'$log$($L_V$/$L_\odot\/$) vs $log$($T$)')

    if savepath == '': plt.show()
    else: plt.savefig(savepath)

    plt.close('all')

def part_c(hipparcos, savepath=''):
    """Compares the Stefan-Boltzmann law to Part B.
    INPUTS:
        hipparcos - The DataFrame containing the magnitudes.
        savepath - The path to save the plot to. If no path is given, the plot is displayed.

    OUTPUTS:
        The plot for PS1 Question 2c.
    """

    #the luminosity of the Sun from Ryden p. 310
    Lo = 3.86e26

    #the radius of the Sun from Ryden p. 319
    Ro = 696e6


    V = hipparcos['V Mag'].to_numpy()
    B = hipparcos['B Mag'].to_numpy()
    parallax = hipparcos['Parallax'].to_numpy()

    #Get the colour index
    B_V = B - V

    #Get the temperatures
    temp = empirical_temperature(B_V)

    #Get Lv/Lo
    lum_ratio = calc_lum_ratio(V, parallax)

    #get the three sets of luminosities and scale them by Lo
    LRo_1 = calc_lum(Ro, temp)/Lo
    LRo_2 = calc_lum(0.2*Ro, temp)/Lo
    LRo_3 = calc_lum(5*Ro, temp)/Lo

    plt.scatter(np.log10(temp), np.log10(lum_ratio), edgecolors='black', label=None)

    plt.plot(np.log10(temp), np.log10(LRo_2), color='orange', label=r'$R = 0.2R_\odot\/$')
    plt.plot(np.log10(temp), np.log10(LRo_1), color='red', label=r'$R = R_\odot\/$')
    plt.plot(np.log10(temp), np.log10(LRo_3), color='green', label=r'$R = 5R_\odot\/$')

    plt.xlabel(r'$log$($T [K]$)')
    plt.ylabel(r'$log$($L_V$/$L_\odot\/$)')

    plt.title(r'$log$($L_V$/$L_\odot\/$) vs $log$($T$)')

    plt.legend()

    if savepath == '': plt.show()
    else: plt.savefig(savepath)

    plt.close('all')


def main():

    here = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(here, 'hipparcos.txt')

    hipparcos = read_data(file)

    a_plot = os.path.join(here, 'PS1-Q2a.png')
    part_a(hipparcos, a_plot)

    #Email Broderick about the approach to calculating Lv/Lo
    b_plot = os.path.join(here, 'PS1-Q2b.png')
    part_b(hipparcos, b_plot)

    c_plot = os.path.join(here, 'PS1-Q2c.png')
    part_c(hipparcos, c_plot)

    """
    Looking at the plot I generated for Part C, there does not seem to be any relation between radius
    of a star and its temperature. The luminosity definitely depends on the temperature + radius, but
    you can have three stars at three different radiuses be the same temperature. Additionally, you
    can have stars at the same radius and different temperatures. This is very surprising because
    equation 13.51 would seem to imply that R is proportional to T^-2.
    """

if __name__ == '__main__':
    main()
