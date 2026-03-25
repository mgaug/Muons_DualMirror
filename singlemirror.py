import matplotlib.pyplot as plt
import numpy as np
from scipy import special

from telescope import *
from setup import *
from plotting import *

SetUp()

tel = MST()

mp.dps = 20  # we use augmented precision in some functions, which may have denominators close to zero


theta_c_deg = 1.3
theta_c = theta_c_deg*np.pi/180.

telescope = tel.name

print ('Quadratic camera with: ')
tel.print()

psi = 180
shadow_conditions_quadratic_camera_nucycle(psi, theta_c, tel.R1,tel.Acam,tel.Dpcam,tel.FOV)    
plt.savefig(f'ShadowConditions_psi{psi:.0f}_nucycle_{telescope:s}.pdf')
plt.show()

psi = 0
shadow_conditions_quadratic_camera_nucycle(psi, theta_c, tel.R1,tel.Acam,tel.Dpcam,tel.FOV)
plt.savefig(f'ShadowConditions_psi{psi:.0f}_nucycle_{telescope:s}.pdf')
plt.show()

nu_deg = np.ceil(tel.FOV/2-theta_c_deg)
shadow_conditions_quadratic_camera_psicycle(nu_deg, theta_c, tel.R1,tel.Acam,tel.Dpcam,tel.FOV)   # without effect of baffles
plt.savefig(f'ShadowConditions_nu{nu_deg:.0f}_psicycle_{telescope:s}.pdf')
plt.show()



nu_deg = 0
psi = 0
Lmax2_shadow_quadratic_camera(nu_deg, psi, theta_c, tel.R1,tel.Acam, tel.Dpcam)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rhole/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

nu_deg = np.ceil(tel.FOV/2-theta_c_deg)
psi = 0
Lmax2_shadow_quadratic_camera(nu_deg, psi, theta_c, tel.R1,tel.Acam, tel.Dpcam)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rhole/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

psi = 180

Lmax2_shadow_quadratic_camera(nu_deg, psi, theta_c, tel.R1,tel.Acam, tel.Dpcam)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rhole/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()



nu_deg = 0
psi = 0
Lmin2_shadow_quadratic_camera(nu_deg, psi, theta_c, tel.R1,tel.Acam, tel.Dpcam)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rhole/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmin2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

nu_deg = np.ceil(tel.FOV/2-theta_c_deg)
psi = 0
Lmin2_shadow_quadratic_camera(nu_deg, psi, theta_c, tel.R1,tel.Acam, tel.Dpcam)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rhole/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmin2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

psi = 180

Lmin2_shadow_quadratic_camera(nu_deg, psi, theta_c, tel.R1,tel.Acam, tel.Dpcam)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rhole/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmin2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()



#nu_deg = 0
#psi = 0
#shadow_conditions_nucycle(psi, theta_c, R1,Rsb,Des,Dpb,tit='')
#nu_deg = 4
#shadow_conditions_nucycle(psi, theta_c, R1,Rsb,Des,Dpb,tit='')
#nu_deg = 4
#psi = 180
#shadow_conditions_nucycle(psi, theta_c, R1,Rsb,Des,Dpb,tit='')


nu_deg = 0
psi = 0
Lmaxmin2_shadow_quadratic_camera(nu_deg, psi, theta_c, tel.R1,tel.Acam, tel.Dpcam)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rhole/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmaxmin2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

nu_deg = np.ceil(tel.FOV/2-theta_c_deg)
psi = 0
Lmaxmin2_shadow_quadratic_camera(nu_deg, psi, theta_c, tel.R1,tel.Acam, tel.Dpcam)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rhole/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmaxmin2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

psi = 180

Lmaxmin2_shadow_quadratic_camera(nu_deg, psi, theta_c, tel.R1,tel.Acam, tel.Dpcam)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rhole/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmaxmin2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()





nu_deg = 0
psi = 0
Lmaxmin2_vs_LVacanti_shadow_relative(nu_deg, psi,theta_c,tel.R1,tel.Rhole,tel.Des,tel.Dpb,1./4./tel.Fp,tel.Dps,tel.Rsb)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rsb/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax1max2min2_relVacanti_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}_{telescope:s}.png')
plt.show()

nu_deg = np.ceil(tel.FOV/2-theta_c_deg)
psi = 0
Lmaxmin2_vs_LVacanti_shadow_relative(nu_deg, psi,theta_c,tel.R1,tel.Rhole,tel.Des,tel.Dpb,1./4./tel.Fp,tel.Dps,tel.Rsb)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rsb/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax1max2min2_relVacanti_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

psi = 180
Lmaxmin2_vs_LVacanti_shadow_relative(nu_deg, psi,theta_c,tel.R1,tel.Rhole,tel.Des,tel.Dpb,1./4./tel.Fp,tel.Dps,tel.Rsb)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rsb/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax1max2min2_relVacanti_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()



nu_deg = 0
psi = 0
Lmaxmin2_vs_Lmaxmin_shadow_relative(nu_deg, psi, theta_c, tel.R1,tel.Rhole,tel.Des,tel.Dpb, 1./4./tel.Fp,tel.Dps,tel.Rsb)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rsb/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax1max2min2_rel_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

nu_deg = np.ceil(tel.FOV/2-theta_c_deg)
psi = 0
Lmaxmin2_vs_Lmaxmin_shadow_relative(nu_deg, psi, theta_c, tel.R1,tel.Rhole,tel.Des,tel.Dpb, 1./4./tel.Fp,tel.Dps,tel.Rsb)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rsb/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax1max2min2_rel_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

psi = 180
Lmaxmin2_vs_Lmaxmin_shadow_relative(nu_deg, psi, theta_c, tel.R1,tel.Rhole,tel.Des,tel.Dpb, 1./4./tel.Fp,tel.Dps,tel.Rsb)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rsb/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax1max2min2_rel_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

nu_deg = 0
psi = 0
Lmaxmin2_vs_Lmaxmin_shadow_absolute(nu_deg, psi, theta_c, tel.R1,tel.Rhole,tel.Des,tel.Dpb, 1./4./tel.Fp,tel.Dps,tel.Rsb)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rsb/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax1max2min2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

nu_deg = np.ceil(tel.FOV/2-theta_c_deg)
psi = 0
Lmaxmin2_vs_Lmaxmin_shadow_absolute(nu_deg, psi, theta_c, tel.R1,tel.Rhole,tel.Des,tel.Dpb, 1./4./tel.Fp,tel.Dps,tel.Rsb)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rsb/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax1max2min2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

psi = 180
Lmaxmin2_vs_Lmaxmin_shadow_absolute(nu_deg, psi, theta_c, tel.R1,tel.Rhole,tel.Des,tel.Dpb, 1./4./tel.Fp,tel.Dps,tel.Rsb)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rsb/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax1max2min2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()



nu_deg = 0
psi = 0
Lmax2_shadow_absolute(nu_deg, psi, theta_c, tel.R1,tel.Rhole,tel.Des,tel.Dpb, 1./4./tel.Fp,tel.Rsb)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rsb/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

nu_deg = np.ceil(tel.FOV/2-theta_c_deg)

psi = 0

Lmax2_shadow_absolute(nu_deg, psi, theta_c, tel.R1,tel.Rhole,tel.Des,tel.Dpb, 1./4./tel.Fp,tel.Rsb)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rsb/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

psi = 180

Lmax2_shadow_absolute(nu_deg, psi, theta_c, tel.R1,tel.Rhole,tel.Des,tel.Dpb, 1./4./tel.Fp,tel.Rsb)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rsb/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmax2_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()


nu_deg = 0

psi = 0

Lmin_hole_absolute(nu_deg, psi, theta_c, tel.R1,tel.Rhole,tel.Des,tel.Dpb, 1./4./tel.Fp,tel.Rsb, tel.Dps)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rhole/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmin_HoleConditions_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

nu_deg = np.ceil(tel.FOV/2-theta_c_deg)

psi = 0

Lmin_hole_absolute(nu_deg, psi, theta_c, tel.R1,tel.Rhole,tel.Des,tel.Dpb, 1./4./tel.Fp,tel.Rsb, tel.Dps)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rhole/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmin_HoleConditions_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

psi = 180

Lmin_hole_absolute(nu_deg, psi, theta_c, tel.R1,tel.Rhole,tel.Des,tel.Dpb, 1./4./tel.Fp,tel.Rsb, tel.Dps)
ax = plt.gca()
circle = plt.Circle((0., 0.), tel.Rhole/tel.R1, transform=ax.transData._b, color="k", alpha=0.2)
ax.add_artist(circle)
plt.savefig(f'Lmin_HoleConditions_psi{psi:.0f}_nu{nu_deg:.1f}_{telescope:s}.png')
plt.show()

#psi = 0
#hole_conditions_nucycle(psi, theta_c, R1,Rhole,Des,1./4./Fp)
#ax = plt.gca()
#circle = plt.Circle((0., 0.), Rhole/R1, transform=ax.transData._b, color="k", alpha=0.2)
#ax.add_artist(circle)
#plt.savefig(f'HoleConditions_psi{psi:.0f}_nucycle_{telescope:s}.pdf')
#plt.show()

#psi = 180
#hole_conditions_nucycle(psi, theta_c, R1,Rhole,Des,1./4./Fp)
#ax = plt.gca()
#circle = plt.Circle((0., 0.), Rhole/R1, transform=ax.transData._b, color="k", alpha=0.2)
#ax.add_artist(circle)
#plt.savefig(f'HoleConditions_psi{psi:.0f}_nucycle_{telescope:s}.pdf')
#plt.show()

#nu_deg = np.floor(tel.FOV/2-theta_c)
#hole_conditions_psicycle(nu_deg, theta_c, R1,Rhole,Des,1./4./Fp)
#ax = plt.gca()
#circle = plt.Circle((0., 0.), Rhole/R1, transform=ax.transData._b, color="k", alpha=0.2)
#ax.add_artist(circle)
#plt.savefig(f'HoleConditions_nu{nu_deg:.0f}_psicycle_{telescope:s}.pdf')
#plt.show()


fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)

rhoR = np.arange(0.001,1.05,0.0995, dtype=np.float64)

psi = 90

# total
tot = 4 * special.ellipe(rhoR**2) * tel.R1**2 / theta_c
totcorr = 2*np.pi/4./tel.Fp * tel.R1**2 * (1-rhoR**2) * (1 - 2*np.cos(psi) / rhoR * (special.ellipk(rhoR**2) - special.ellipe(rhoR**2)))
phi = 0
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 45
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 90
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 135
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 180
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 225
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 270
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 315
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
ax.plot(rhoR, totcorr/tot,'--',color='k', label='integrated \n over all'+r' $\phi$')
ax.legend(loc='center left', fontsize=15,bbox_to_anchor=(0.95, 0.5))
ax.set_title(rf'SST for $\nu$={nu*180/np.pi:.0f}$^\circ$, $\theta_c$={theta_c*180/np.pi:.1f}$^\circ$, $\psi$={psi:.0f}') 
ax.set_xlabel(r'$\rho_R$')
ax.set_ylabel(r'$\Delta L_{max}^{(1)}/L_{max}^{(0)}$')
plt.savefig(f'LMax2Correction_{telescope:s}_{psi:d}.pdf', bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(right=0.99)

psi = 0
totcorr = 2*np.pi/4./tel.Fp * tel.R1**2 * (1-rhoR**2) * (1 - 2*np.cos(psi) / rhoR * (special.ellipk(rhoR**2) - special.ellipe(rhoR**2)))
phi = 0
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 45
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 90
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 135
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 180
ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
#phi = 225
#ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
#phi = 270
#ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
#phi = 315
#ax.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp,nu,psi)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
#ax.legend(loc='center left', fontsize=15,bbox_to_anchor=(0.95, 0.5))

ax.plot(rhoR, totcorr/tot,'--',color='k', label='integrated \n over all'+r' $\phi$')
ax.set_title(rf'SST for $\nu$={nu*180/np.pi:.0f}$^\circ$, $\theta_c$={theta_c*180/np.pi:.1f}$^\circ$, $\psi$={psi:.0f}') 
ax.set_xlabel(r'$\rho_R$')
ax.set_ylabel(r'$\Delta L_{max}^{(1)}/L_{max}^{(0)}$')
plt.savefig(f'LMax2Correction_{telescope:s}_{psi:d}.pdf', bbox_inches='tight')
plt.show()

phi = 0
nu = 0
#plt.plot(rhoR, Lmax(rhoR,phi,R1,theta_c,1./4./Fp), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
Lmax_s = Lmax(rhoR,phi,tel.R1,theta_c)
Lmax_w = Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp)

dff = (Lmax_w - Lmax_s)/Lmax_s
print ('Lmax_s: ', Lmax_s, ' Lmax_w: ',Lmax_w, ' dff: ', dff)
plt.plot(rhoR,dff, label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 45
plt.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 90
plt.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 135
plt.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 180
plt.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 225
plt.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 270
plt.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
phi = 315
plt.plot(rhoR, (Lmax(rhoR,phi,tel.R1,theta_c,1./4./tel.Fp)-Lmax(rhoR,phi,tel.R1,theta_c))/Lmax(rhoR,phi,tel.R1,theta_c), label=r'$\phi-\phi_{0}=$'+f'{phi:d}'+r'$^{\circ}$')
plt.legend(loc='best', fontsize=15)
plt.xlabel(r'$\rho_R$')
plt.ylabel(r'$L_{max} (c-corr.)/L_{max}$ (Vacanti)')
plt.savefig('LMaxCorrection_c_{telescope:s}.pdf', bbox_inches='tight')
plt.show()

