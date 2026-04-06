from atmosphere_helper_v import AtmosphereHelper
import numpy as np
import matplotlib.pyplot as plt
from mumodel_helper_v import ev2nm, nm2ev

photon_energies = np.linspace(1.5, 5.0, 200)

atm = AtmosphereHelper.from_lst()

theta_c = 1. # deg
rhoR_min = 0.
rhoR_max = 1.0
rhoRs = np.arange(rhoR_min,rhoR_max,0.19999)

zenith_angle = 0  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

vaod = 0.03
AE = 1.45    # non-dusty periods day-time, see Fig. 3 of https://www.aanda.org/articles/aa/full_html/2023/05/aa45787-22/F3.html
Haer = 500.
HPBL = 800. * costheta ** 0.77
HElterman = 1200

scale_h = 9700.

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_av_transmission_rho_mol(photon_energies,np.deg2rad(theta_c),costheta,
                                 rhoR_min=rhoR_min, rhoR_max=rhoR_max,
                                 scale_h=9700., label='CTAO-N Av. Winter',ax=ax,
                                 n_rho=10,lw=1)
atm.plot_av_transmission_rho_mol(photon_energies,np.deg2rad(theta_c),costheta,
                                 rhoR_min=rhoR_min, rhoR_max=rhoR_max,
                                 scale_h=10300., label='CTAO-N Av. Summer',ax=ax,
                                 n_rho=10,lw=1)

atm.plot_av_transmission_rho_aer(photon_energies,np.deg2rad(theta_c),costheta,
                                 rhoR_min=rhoR_min, rhoR_max=rhoR_max,
                                 vaod=vaod, AE=AE,Haer=Haer,HPBL=HPBL,
                                 HElterman=HElterman, ax=ax,
                                 n_rho=10,lw=2, label='CTAO-N, Winter model')

HPBL = 100.     # extremely shallow nocturnal turbulent surface layer PBL at Paranal, see
                # https://academic.oup.com/mnras/article/492/1/934/5674124, afterwards exponential decay
Haer = 500.
HElterman = 9000

atm.plot_av_transmission_rho_aer(photon_energies,np.deg2rad(theta_c),costheta,
                                 rhoR_min=rhoR_min, rhoR_max=rhoR_max,
                                 vaod=vaod, AE=AE,Haer=Haer,HPBL=HPBL,
                                 HElterman=HElterman, ax=ax,
                                 n_rho=10,lw=2, label='CTAO-S, best guess')
plt.show()


Haer = 500.
HPBL = 800. * costheta ** 0.77
HElterman = 1200


fig, ax = plt.subplots(constrained_layout=True)
atm.plot_av_phi_aer_vs_rhoR(photon_energies,np.deg2rad(theta_c),rhoRs,
                            costheta,
                            vaod, AE,
                            Haer=Haer,
                            HPBL=HPBL,
                            HElterman=HElterman, n_path=256, ax=ax)
atm.plot_av_transmission_rho_aer(photon_energies,np.deg2rad(theta_c),costheta,
                                 rhoR_min=rhoR_min, rhoR_max=rhoR_max,
                                 vaod=vaod, AE=AE,Haer=Haer,HPBL=HPBL,
                                 HElterman=HElterman, ax=ax,
                                 n_rho=10,lw=3, color='k')
plt.show()


zenith_angle = 60  # deg
costheta=np.cos(np.deg2rad(zenith_angle))
HPBL = 800. * costheta ** 0.77

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_av_phi_aer_vs_rhoR(photon_energies,np.deg2rad(theta_c),rhoRs,
                            costheta,
                            vaod, AE,
                            Haer=Haer,
                            HPBL=HPBL,
                            HElterman=HElterman, ax=ax)
atm.plot_av_transmission_rho_aer(photon_energies,np.deg2rad(theta_c),costheta,
                                 rhoR_min=rhoR_min, rhoR_max=rhoR_max,
                                 vaod=vaod, AE=AE,Haer=Haer,HPBL=HPBL,
                                 HElterman=HElterman, ax=ax,
                                 n_rho=10,lw=3, color='k')
plt.show()


zenith_angle = 0  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_av_phi_mol_vs_rhoR(photon_energies,np.deg2rad(theta_c),rhoRs,
                            costheta=costheta,ax=ax)
atm.plot_av_transmission_rho_mol(photon_energies,np.deg2rad(theta_c),costheta,
                                 rhoR_min=rhoR_min, rhoR_max=rhoR_max, ax=ax,
                                 n_rho=10,lw=3, color='k')
plt.show()

zenith_angle = 60  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_av_phi_mol_vs_rhoR(photon_energies,np.deg2rad(theta_c),rhoRs,
                            costheta=costheta,ax=ax)
atm.plot_av_transmission_rho_mol(photon_energies,np.deg2rad(theta_c),costheta,
                                 rhoR_min=rhoR_min, rhoR_max=rhoR_max, ax=ax,
                                 n_rho=10,lw=3, color='k')
plt.show()




zenith_angle = 0  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

scale_h = 9700.

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_av_transmission_mol(photon_energies,rmax=5000.,costheta=costheta, scale_h=scale_h, rmin=None, ax=ax, debug=True, n_path=10)
atm.plot_av_transmission_mol(photon_energies,rmax=5000.,costheta=costheta, scale_h=scale_h,rmin=1000., ax=ax, debug=True, n_path=10)
plt.show()

zenith_angle = 0  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

Haer = 500.
HPBL = 800. * costheta ** 0.77
HElterman = 1200
AE = 1.2

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, Haer=500, label="Haer=500",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, Haer=1000, label="Haer=1000",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, Haer=1500, label="Haer=1500",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, Haer=2000, label="Haer=2000",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, Haer=500, label="Haer=500",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, Haer=1000, label="Haer=1000",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, Haer=1500, label="Haer=1500",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, Haer=2000, label="Haer=2000",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
plt.hlines(np.exp(-vaod/costheta),nm2ev(532.)-0.1,nm2ev(532.)+0.1,colors='k',lw=4)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0],1.)
plt.show()


zenith_angle = 60  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

Haer = 500.
HPBL = 800. * costheta ** 0.77
HElterman = 1200
AE = 1.2

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, Haer=500, label="Haer=500",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, Haer=1000, label="Haer=1000",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, Haer=1500, label="Haer=1500",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, Haer=2000, label="Haer=2000",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, Haer=500, label="Haer=500",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, Haer=1000, label="Haer=1000",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, Haer=1500, label="Haer=1500",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, Haer=2000, label="Haer=2000",
                          vaod=vaod, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
plt.hlines(np.exp(-vaod/costheta),nm2ev(532.)-0.1,nm2ev(532.)+0.1,colors='k',lw=4)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0],1.)
plt.show()


zenith_angle = 60  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

Haer = 500.
HPBL = 800. * costheta ** 0.77
HElterman = 1200
AE = 1.2

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HPBL=500, label="HPBL=500",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HPBL=1000, label="HPBL=1000",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HPBL=1500, label="HPBL=1500",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HPBL=3500, label="HPBL=3500",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, HPBL=500, label="HPBL=500",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, HPBL=1000, label="HPBL=1000",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, HPBL=1500, label="HPBL=1500",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, HPBL=3500, label="HPBL=3500",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
plt.hlines(np.exp(-vaod/costheta),nm2ev(532.)-0.1,nm2ev(532.)+0.1,colors='k',lw=4)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0],1.)
plt.show()

zenith_angle = 0  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

Haer = 500.
HPBL = 800. * costheta ** 0.77
HElterman = 1200
AE = 1.2

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HPBL=500, label="HPBL=500",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HPBL=1000, label="HPBL=1000",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HPBL=1500, label="HPBL=1500",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HPBL=3500, label="HPBL=3500",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, HPBL=500, label="HPBL=500",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, HPBL=1000, label="HPBL=1000",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, HPBL=1500, label="HPBL=1500",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, HPBL=3500, label="HPBL=3500",
                          vaod=vaod, Haer=Haer, HElterman=HElterman, ax=ax, show=False)
plt.hlines(np.exp(-vaod/costheta),nm2ev(532.)-0.1,nm2ev(532.)+0.1,colors='k',lw=4)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0],1.)
plt.show()

zenith_angle = 0  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

Haer = 500.
HPBL = 800. * costheta ** 0.77
HElterman = 1200
AE = 1.2

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HElterman=500, label="HElterman= 500",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HElterman=800, label="HElterman= 800",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HElterman=1000, label="HElterman= 1000",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HElterman=1500, label="HElterman= 1500",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, HElterman=500, label="HElterman= 500, rmax=200",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)###, debug=True)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, HElterman=800, label="HElterman= 800, rmax=200",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)###, debug=True)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, HElterman=1000, label="HElterman= 1000, rmax=200",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)#, debug=True)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE, HElterman=1500, label="HElterman= 1500, rmax=200",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)#, debug=True)
plt.hlines(np.exp(-vaod/costheta),nm2ev(532.)-0.1,nm2ev(532.)+0.1,colors='k',lw=4)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0],1.)
plt.show()

zenith_angle = 60  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HElterman=500, label="HElterman= 500",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HElterman=800, label="HElterman= 800",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HElterman=1000, label="HElterman= 1000",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, HElterman=1500, label="HElterman= 1500",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=200.0, costheta=costheta, AE=AE, HElterman=500, label="HElterman= 500, rmax=200",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)###, debug=True)
atm.plot_transmission_aer(photon_energies,rmax=200.0, costheta=costheta, AE=AE, HElterman=800, label="HElterman= 800, rmax=200",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)###, debug=True)
atm.plot_transmission_aer(photon_energies,rmax=200.0, costheta=costheta, AE=AE, HElterman=1000, label="HElterman= 1000, rmax=200",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)#, debug=True)
atm.plot_transmission_aer(photon_energies,rmax=200.0, costheta=costheta, AE=AE, HElterman=1500, label="HElterman= 1500, rmax=200",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, ax=ax, show=False)#, debug=True)
plt.hlines(np.exp(-vaod/costheta),nm2ev(532.)-0.1,nm2ev(532.)+0.1,colors='k',lw=4)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0],1.)
plt.show()


zenith_angle = 60  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

vaod = 0.03
Haer = 500.
HPBL = 800. * costheta ** 0.77
HElterman = 1200

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=-0.5, label="AE= -0.5",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=0.0, label="AE= 0.0",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=0.5, label="AE= 0.5",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=1.5, label="AE= 1.5",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=2.5, label="AE= 2.5",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
plt.hlines(np.exp(-vaod/costheta),nm2ev(532.)-0.1,nm2ev(532.)+0.1,colors='k',lw=4)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0],1.)
plt.show()

zenith_angle = 0  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=-0.5, label="AE= -0.5",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=0.0, label="AE= 0.0",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=0.5, label="AE= 0.5",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=1.5, label="AE= 1.5",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=2.5, label="AE= 2.5",
                          vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
plt.hlines(np.exp(-vaod/costheta),nm2ev(532.)-0.1,nm2ev(532.)+0.1,colors='k',lw=4)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0],1.)
plt.show()


zenith_angle = 60  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

vaod = 0.03
Haer = 500.
HPBL = 800. * costheta ** 0.77
HElterman = 1200
AE = 1.2

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=4000.0, costheta=costheta, AE=AE, vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=3000.0, costheta=costheta, AE=AE,vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=2000.0, costheta=costheta, AE=AE,vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=1000.0, costheta=costheta, AE=AE,vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE,vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=250.0, costheta=costheta, AE=AE,vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
plt.hlines(np.exp(-vaod/costheta),nm2ev(532.)-0.1,nm2ev(532.)+0.1,colors='k',lw=4)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0],1.)
plt.show()

zenith_angle = 0  # deg
costheta=np.cos(np.deg2rad(zenith_angle))

fig, ax = plt.subplots(constrained_layout=True)
atm.plot_transmission_aer(photon_energies,rmax=5000.0, costheta=costheta, AE=AE, vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=4000.0, costheta=costheta, AE=AE, vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=3000.0, costheta=costheta, AE=AE,vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=2000.0, costheta=costheta, AE=AE,vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=1000.0, costheta=costheta, AE=AE,vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=500.0, costheta=costheta, AE=AE,vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
atm.plot_transmission_aer(photon_energies,rmax=250.0, costheta=costheta, AE=AE,vaod=vaod, Haer=Haer, HPBL=HPBL, HElterman=HElterman, ax=ax, show=False)
plt.hlines(np.exp(-vaod/costheta),nm2ev(532.)-0.1,nm2ev(532.)+0.1,colors='k',lw=4)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0],1.)
plt.show()

