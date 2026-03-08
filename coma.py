import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

mpl.rcParams.update({

    # --- Figure ---
    "figure.figsize": (7.5, 6.0),   # inches
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    # --- Fonts ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "mathtext.fontset": "stix",

    # --- Axes ---
    "axes.linewidth": 1.2,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,

    # --- Ticks ---
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    # --- Lines ---
    "lines.linewidth": 3,
    "lines.markersize": 6,

    # --- Legend ---
    "legend.frameon": False,
    "legend.loc": "best",
    "legend.handlelength": 2.5,

    # --- Errorbars ---
    "errorbar.capsize": 3,

    # --- LaTeX (optional) ---
    # "text.usetex": True,
    # "text.latex.preamble": r"\usepackage{amsmath}",

})

def delta_coma(rhoR,phi_0,nu,theta_c,Fs):

    sq   = np.sqrt(1 - (rhoR*np.sin(phi_0*np.pi/180.))**2)
    rhoc = rhoR * np.cos(phi_0*np.pi/180.)

    return 0.125/Fs**2 * (
         (4*rhoc**2 + 1 + 2*rhoc*sq)
        - 3*nu/theta_c*(rhoc**2 + rhoc * sq)
    )

theta_c = 1.*np.pi/180.
Fs = 1.2    # m 

plt.figure()
rhoR = np.arange(0.,1.1,0.1)
nu = theta_c

# plate-scale correction
ps = nu/theta_c/8/Fs**2
plt.axhline(y=ps,color='k',linestyle=":",label='plate scale corr.')
phi0 = 0
plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d}'+r'$^{\circ}$')
phi0 = 45
plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d}'+r'$^{\circ}$ or 315'+r'$^{\circ}$')
phi0 = 90
plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d}'+r'$^{\circ}$ or 270'+r'$^{\circ}$')
phi0 = 135
plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d}'+r'$^{\circ}$ or 225'+r'$^{\circ}$')
phi0 = 180
plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d} deg.')
#phi0 = 225
#plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d} deg.')
#phi0 = 270
#plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d} deg.')
#phi0 = 315
#plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d} deg.')
plt.legend(loc='best', fontsize=15)
plt.xlabel(r'$\rho_R$')
plt.ylabel(r'$(\overline{\Delta x_{cam}}(\phi=0)-\overline{\Delta x_{cam}}(\phi=\pi))/(F\cdot \theta_c)$-$\nu/(8F_{\#}^{2}\theta_c)$')
plt.savefig('Coma_withplatescale.pdf', bbox_inches='tight')
plt.show()


# plate-scale correction
ps = nu/theta_c/8/Fs**2
plt.axhline(y=ps,color='k',linestyle=":",label='plate scale corr.')
ps = 0
phi0 = 0
plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d}'+r'$^{\circ}$')
phi0 = 45
plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d}'+r'$^{\circ}$ or 315'+r'$^{\circ}$')
phi0 = 90
plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d}'+r'$^{\circ}$ or 270'+r'$^{\circ}$')
phi0 = 135
plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d}'+r'$^{\circ}$ or 225'+r'$^{\circ}$')
phi0 = 180
plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d} deg.')
#phi0 = 225
#plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d} deg.')
#phi0 = 270
#plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d} deg.')
#phi0 = 315
#plt.plot(rhoR, delta_coma(rhoR,phi0,nu,theta_c,Fs)-ps,label=r'$\phi_{0}=$'+f'{phi0:d} deg.')
plt.legend(loc='best', fontsize=15)
plt.xlabel(r'$\rho_R$')
plt.ylabel(r'$(\overline{\Delta x_{cam}}(\phi=0)-\overline{\Delta x_{cam}}(\phi=\pi))/(F\cdot \theta_c)$')
plt.savefig('Coma_noplatescale.pdf',bbox_inches='tight' )
plt.show()

