import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib as mpl
from mpmath import mp
from dual_mirror_lib import *
from single_mirror_lib import *

'''
A small library to produce the plots in the paper Section 4
'''

def Lmax_sing(rhoR,phi,R1, theta_c,c,nu,psi, phi0=0):

    cospp0 = mp.cos((phi-phi0)*mp.pi/180.)
    sinpp0 = mp.sin((phi-phi0)*mp.pi/180.)

    cospps = mp.cos((phi-psi)*mp.pi/180.)
    sinpps = mp.sin((phi-psi)*mp.pi/180.)
    
    sqsinp = mp.sqrt(1-rhoR**2*sinpp0**2)
    LmaxVac = R1 * (rhoR*cospp0 + sqsinp ) / theta_c   # Vacanti solution 
    
    FOcorr = c*R1**2 * (1-rhoR**2)   # 0th order correction, forgotten by Vacanti 

    F1corr = c*R1**2 * (1-rhoR**2) * (nu/theta_c) * cospps # 1st order correction for nu != 0
    F2corr = c*R1**2 * (1-rhoR**2) * (nu/theta_c) * rhoR * sinpp0 * sinpps /2./sqsinp  # 2nd order correction for nu != 0
    
    return np.float64(LmaxVac + FOcorr + F1corr - F2corr)

def Lmax(rhoR,phi,R1, theta_c,c=0,nu=0,psi=0, phi0=0):
    
    return np.array([Lmax_sing(v,phi,R1,theta_c,c,nu,psi, phi0) for v in np.atleast_1d(rhoR)], dtype=np.float64)

def L_noshadow(rhoR,phi,R1, theta_c,c=0,nu=0,psi=0, phi0=0):

    cospps = np.cos((phi-psi)*np.pi/180.) 
    sinpps = np.sin((phi-psi)*np.pi/180.) 

    sinpp0 = np.sin((phi-phi0)*np.pi/180.)
    cospp0 = np.cos((phi-phi0)*np.pi/180.)

    if (isinstance(sinpp0, (list, tuple, np.ndarray)) and isinstance(rhoR, (list, tuple, np.ndarray))):

        # helper to bring the psi-only arrays into shape
        rhoshaper = np.ones((rhoR.size,1),dtype=np.float64)
        phishaper = np.ones((1,phi.size),dtype=np.float64)          

        dphi = (phi[None,:] - phi0 + 180.) % 360. - 180.   # now in range (-180, 180]
        sqsinp = np.sqrt(1-rhoR[:,None]**2*sinpp0**2)    
        
        Lmax   = np.where((rhoR[:,None]*phishaper < 1.) | ((np.abs(dphi) < 90.) & (rhoR[:,None]*sinpp0 < 1.)),
                          R1*(rhoR[:,None]*cospp0 + np.sqrt(1. - (rhoR[:,None]*sinpp0)**2))/theta_c, # Vacanti solution for rhoR < 1, other Lmin needs to be subtracted
                          0.)                          
                          
        F1corr = np.where((rhoR[:,None]*phishaper < 1.) | ((np.abs(dphi) < 90.) & (rhoR[:,None]*sinpp0 < 1.)),
                          R1**2 * c* (1-(rhoR[:,None]*phishaper)**2) * (1. + nu/theta_c*(cospps)),
                          0.)  # first order correction
        F2corr = np.where((rhoR[:,None]*phishaper < 1.) | ((np.abs(dphi) < 90.) & (rhoR[:,None]*sinpp0 < 1.)),
                          R1**2 * c* (1-(rhoR[:,None]*phishaper)**2) * (nu/theta_c) * rhoR[:,None] * sinpp0 * sinpps /2./sqsinp,  # 2nd order correction for nu != 0
                          0.)

        Lmin   = np.where((rhoR[:,None]*phishaper < 1.),
                          0.,
                          np.where((np.abs(dphi) < 90.) & (rhoR[:,None]*sinpp0 < 1.),
                                   R1*(rhoR[:,None]*cospp0 - np.sqrt(1. - (rhoR[:,None]*sinpp0)**2))/theta_c + R1**2 * c * (1-(rhoR[:,None]*phishaper)**2)*(1. + nu/theta_c*(cospps)),
                                   0.)
                          ) # Vacanti solution for rhoR < 1, other Lmin needs to be subtracted

        #rho_test = np.where(((rhoR[:,None]*phishaper < 1.) | ((np.abs(dphi) < 90.) & (rhoR[:,None]*sinpp0 < 1.))),1000.,rhoR[:,None]*phishaper)
        #print ('rho_test: ', rho_test[rho_test < 1000.])
        #phi_test = np.where((rhoR[:,None]*phishaper < 1.) | ((np.abs(dphi) < 90.) & (rhoR[:,None]*sinpp0 < 1.)),1000.,rhoshaper*phi)
        #print ('phi_test: ', phi_test[phi_test < 1000.])
        #print ('Lmin_test: ', Lmin[rho_test < 1000.])
        
        return Lmax + F1corr + F2corr - Lmin

    else:

        dphi = (phi - phi0 + 180.) % 360. - 180.   # now in range (-180, 180]

        if (rhoR < 1.) or ((np.abs(dphi) < 90.) and (rhoR*sinpp0 < 1.)):
            Lmax   = R1*(rhoR*cospp0 + np.sqrt(1, - (rhoR*sinpp0)**2))/theta_c, # Vacanti solution for rhoR < 1, other Lmin needs to be subtracted
            
            F1corr = R1**2 * c* (1-rhoR**2) * (1. + nu/theta_c*(cospps)),
            F2corr = R1**2 * c* (1-rhoR**2) * (nu/theta_c) * rhoR * sinpps * sinpps /2./sqsinp,  # 2nd order correction for nu != 0

            return Lmax + F1corr + F2corr

        return 0.

    
def Lmax2_sing(rhoR,phi,R2,D2,theta_c,nu,psi):

    cosp  = R2*rhoR*mp.cos(phi*mp.pi/180.)
    cospp = D2*nu*mp.cos((psi-phi)*mp.pi/180.)
    sinp = R2**2*(1-rhoR**2*mp.sin(phi*mp.pi/180.)**2)
    sinpp = 2*D2*R2*nu*rhoR*mp.sin((phi-psi)*mp.pi/180)*mp.sin(phi*mp.pi/180)
    cosdp = D2**2*(theta_c + nu*mp.cos((psi-phi)*mp.pi/180.))**2

    return D2 + cosp/theta_c + cospp/theta_c + mp.sqrt(sinp-sinpp+cosdp)/theta_c

def Lmax2(rhoR,phi,R2,D2,theta_c,nu,psi):
    
    return np.array([Lmax2_sing(v,phi,R2,D2,theta_c,nu,psi) for v in np.atleast_1d(rhoR)], dtype=np.float64)

def realign_polar_xticks(ax):
    for x, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        #print ('x, cos(x), label', x,np.cos(x), label)
        if np.abs(np.cos(x)) < 1e-8: # numerical 0
            label.set_horizontalalignment('center')
        elif np.cos(x) > 0.:
            label.set_horizontalalignment('left')
        elif np.cos(x) < 0:
            label.set_horizontalalignment('right')


def Lmin_hole_absolute(nu_deg, psi, theta_c, R1,Rhole,Des,Dpb, c, Rsb, Dps, phi_0=0):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(22.)
    
    rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    phi = np.arange(-180.,180.1,0.2, dtype= np.float64)

    print ('phi: ',phi.shape)
    print ('rho: ',rhoR.shape)

    #rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    #phi = np.arange(-180.,180,0.2, dtype= np.float64)

    rhoR_mean = 0.5*(rhoR[:-1] + rhoR[1:])     # centers, (Nr,)    
    phi_mean = 0.5*(phi[:-1] + phi[1:])        # centers, (Nθ,)    
    
    nu = np.deg2rad(nu_deg)

    cond_baffle = muon_traversing_baffle_condition(rhoR_mean,phi_0,nu,psi,R1,Rsb,Des,Dpb)  # has shape of rhoR_mean x 1 x 1
    #cond_M2     = M2_traverse_condition(rhoR_mean,phi_0,nu,psi,R1,Rsb,Des,Dpb)      # has shape of rhoR_mean
    #cond_hole   = muon_traversing_hole_condition(rhoR_mean,R1,Rhole)                # has shape of rhoR_mean
    cond_shadow = global_shadow_condition_from_M2(rhoR_mean,phi_0,nu,psi,
                                                  phi_mean,theta_c,
                                                  R1,Rsb,Dpb,Des)                   # has shape of rhoR_mean x phi_mean x 1

    #print ('cond_shadow: ', cond_shadow.shape)
    #print ('cond_baffle: ', cond_baffle.shape)
    #print ('cond_M2: ', cond_M2[:,None,None].shape)

    #Lmin_hole = Lmin_from_hole(rhoR_mean,phi_0,nu,psi,phi_mean,theta_c,R1,Rhole,c,Des,Dps)
    
    #Lmin_hole = np.where(cond_hole[:,None,None],
    #                     Lmin_from_muon_passing_hole(rhoR_mean,phi_0,nu,psi,phi_mean,theta_c,R1,Rhole,c,Des),
    #                     Lmin_hole_other_light_losses(rhoR_mean,phi_0,nu,psi,phi_mean,theta_c,R1,Rhole,c,Dps)
    #)

    #print ('Lmin_hole: ',Lmin_hole.shape)

    #Lmin_no_hole = Lmin_light_losses_no_M2_traverse(rhoR_mean,phi_0,nu,psi,phi_mean,theta_c,R1,Rsb,Des)

    #print ('Lmin_no_hole: ',Lmin_no_hole.shape)    

    Dc = muon_baffle_crossing_vertical_distance(rhoR_mean,phi_0,nu,psi,R1,Rsb,Des,Dpb) # has shape rhoR_mean x 1 x phi0
    print ('Dc: ',Dc.shape)
    #print ('nu: ',nu)
    #Dctest = np.where(cond_baffle,Dc,np.nan)
    
    print ('Dc: ',Dc[cond_baffle])

    rho_test = rhoR_mean[:,None,None]
    #print ('rho: ',rho_test[cond_baffle]*R1/Rsb)

    #Lmin_M2_no_baffle = np.where(cond_M2[:,None,None],
    #                             Lmin_hole,
    #                             Lmin_no_hole)

    Lmin_M2_no_baffle = Lmin_from_nobaffle(rhoR_mean,phi_0,nu,psi,phi_mean,theta_c,R1,Rhole,c,Rsb,Des,Dpb,Dps)
    
    #print ('Lmin_M2_nobaffle: ',Lmin_M2_no_baffle.shape)

    #Z_min=np.nanmin(Lmin_M2_no_baffle)
    #Z_max=np.nanmax(Lmin_M2_no_baffle)

    #print ('Zmin: ',Z_min, ' Zmax: ',Z_max)

    Lmin_shadow = Lmin_for_M2_shadows(rhoR_mean,phi_0,nu,psi,phi_mean,theta_c,R1,Rhole,c,Rsb,Dpb,Des,Dps)

    #print ('Lmin_shadow: ',np.squeeze(Lmin_shadow, axis=2).shape)    
    
    Z_min=np.nanmin(Lmin_shadow)
    Z_max=np.nanmax(Lmin_shadow)

    print ('Zmin: ',Z_min, ' Zmax: ',Z_max)
    
    Lmin = np.where(cond_shadow,
                    Lmin_shadow, 0.)

    #np.where(cond_baffle,
    #                         Dc,
    #                         Lmin_M2_no_baffle
    #                ),
    #                0.
    #)

    Lmin_test = np.where(cond_baffle,
                         Dc,
                         Lmin_M2_no_baffle)
        
    
    #print ('Lmin_M2: ',np.squeeze(Lmin, axis=2).shape)    
                             
    #L = L_noshadow(rhoR_mean,phi_mean,R1, theta_c,c,nu,psi) * cond
    #Lmin_h = hole_condition_with_secondary(rhoR_mean,phi_mean,R1,Rhole,Des,theta_c,nu,psi,c) * cond

    Z_min=np.nanmin(Lmin_test)
    Z_max=np.nanmax(Lmin_test)

    print ('Zmin: ',Z_min, ' Zmax: ',Z_max)
    # ------------------------------------------------------------
    norm = mcolors.LogNorm(vmin=1e-1,vmax=Z_max,clip=False)

    #print ('rho: ',rhoR.shape)
    
    cm = ax.pcolormesh(np.deg2rad(phi),rhoR,
                       np.squeeze(Lmin,axis=2),
                       #np.squeeze(Lmin,axis=2),
                       #, # / L_noshadow(rhoR_mean,phi_mean,R1, theta_c,c,nu,psi),
                       shading='auto', norm=norm,
                       cmap='viridis')
    fig.colorbar(cm, ax=ax, shrink=0.8,pad=0.1,label=r'$L_{min}$ (m)')    
    
    cm.set_label(r'$\nu=$'+f'{nu_deg:.1f}'+r'$^{\circ}$')
    ax.set_title(rf'$\nu={nu_deg:.1f}^\circ, \psi = {psi:.0f}^\circ, \phi_0=0^\circ$')
    ax.set_xlabel(r'$\phi-\phi_0$')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position),ax.get_rmax()*1.25,r'$\rho_R$',fontsize=21,
            rotation=label_position,ha='center',va='center')
    realign_polar_xticks(ax)
    ax.grid(alpha=0.2)
    #ax.legend(handles=handles, loc="upper left",
    #          bbox_to_anchor=(-0.4,1.15),
    #          fontsize=18)

def Lmax2_shadow_absolute(nu_deg, psi, theta_c,R1,Rhole,Des,Dpb, c,Rsb, phi_0=0):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(22.)
    
    rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    phi = np.arange(-180.,180.1,0.2, dtype= np.float64)

    rhoR_mean = 0.5*(rhoR[:-1] + rhoR[1:])     # centers, (Nr,)    
    phi_mean = 0.5*(phi[:-1] + phi[1:])        # centers, (Nθ,)    
    
    nu = np.deg2rad(nu_deg)

    cond_shadow = global_shadow_condition_from_M2(rhoR_mean,phi_0,nu,psi,
                                                  phi_mean,theta_c,
                                                  R1,Rsb,Dpb,Des)                   # has shape of rhoR_mean x phi_mean x 1

    #print ('cond_shadow: ',cond_shadow.shape)

    Lmax2 = Lmax_shadow_condition_from_M2(rhoR_mean,phi_0,nu,psi, 
                                         phi_mean,theta_c, 
                                         R1,Rsb,Dpb,Des)
    
    #print ('Lmax2: ', Lmax2.shape)
                         
    Z_min=np.nanmin(Lmax2[cond_shadow])
    Z_max=np.nanmax(Lmax2)
    Lmax2_bad = np.where(cond_shadow & (Lmax2 < Dpb), Lmax2, 11.)
                              
    #print (Lmax2_bad.shape)

    print ('Zmin: ',Z_min, ' Zmax: ',Z_max)
    # ------------------------------------------------------------
    norm = mcolors.LogNorm(vmin=1.,vmax=Z_max,clip=False)
    #norm = mcolors.Normalize(vmin=Z_min,vmax=Z_max,clip=False)

    # Get color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #fig.subplots_adjust(wspace=-0.5)
    handles = []
    
    cm = ax.pcolormesh(np.deg2rad(phi),rhoR,
                       np.squeeze(Lmax2, axis=2),
                       #, # / L_noshadow(rhoR_mean,phi_mean,R1, theta_c,c,nu,psi),
                       shading='auto', norm=norm,
                       cmap='viridis')
    fig.colorbar(cm, ax=ax, shrink=0.8,pad=0.1,label=r'$L_{2,max}$ (m)')    
    
    #cm.set_label(r'$\nu=$'+f'{nu_deg:.1f}'+r'$^{\circ}$')
    ax.set_title(rf'$\nu={nu_deg:.1f}^\circ, \psi = {psi:.0f}^\circ, \phi_0=0^\circ$')
    ax.set_xlabel(r'$\phi-\phi_0$')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position),ax.get_rmax()*1.25,r'$\rho_R$',fontsize=21,
            rotation=label_position,ha='center',va='center')
    realign_polar_xticks(ax)
    ax.grid(alpha=0.2)
            
def Lmaxmin2_shadow_absolute(nu_deg, psi,theta_c,R1,Rhole,Des,Dpb,c,Dps,Rsb,phi_0=0):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(22.)
    
    rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    phi = np.arange(-180.,180.1,0.2, dtype= np.float64)

    rhoR_mean = 0.5*(rhoR[:-1] + rhoR[1:])     # centers, (Nr,)    
    phi_mean = 0.5*(phi[:-1] + phi[1:])        # centers, (Nθ,)    
    
    nu = np.deg2rad(nu_deg)

    cond_shadow = global_shadow_condition_from_M2(rhoR_mean,phi_0,nu,psi,
                                                  phi_mean,theta_c,
                                                  R1,Rsb,Dpb,Des)                   # has shape of rhoR_mean x phi_mean x 1

    #print ('cond_shadow: ',cond_shadow.shape)

    Lmax2 = Lmax_shadow_condition_from_M2(rhoR_mean,phi_0,nu,psi, 
                                         phi_mean,theta_c, 
                                         R1,Rsb,Dpb,Des)
    
    #print ('Lmax2: ', Lmax2.shape)
                         
    Lmin2 = Lmin_for_M2_shadows(rhoR_mean,phi_0,nu,psi, 
                                phi_mean,theta_c, 
                                R1,Rhole,c,
                                Rsb,Dpb,Des,Dps)
    
    #print ('Lmin2: ', Lmin2.shape)

    Lmaxmin2 = Lmax2 - Lmin2

    
    Z_min=np.nanmin(Lmaxmin2[cond_shadow])
    Z_max=np.nanmax(Lmaxmin2)
    #Lmax2_bad = np.where(cond_shadow & (Lmax2 < Dpb), Lmax2, 11.)                              
    #print (Lmax2_bad.shape)

    print ('Zmin: ',Z_min, ' Zmax: ',Z_max)
    # ------------------------------------------------------------
    norm = mcolors.LogNorm(vmin=1.,vmax=Z_max,clip=False)
    #norm = mcolors.Normalize(vmin=Z_min,vmax=Z_max,clip=False)

    # Get color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #fig.subplots_adjust(wspace=-0.5)
    handles = []
    
    cm = ax.pcolormesh(np.deg2rad(phi),rhoR,
                       np.squeeze(Lmaxmin2, axis=2),
                       #, # / L_noshadow(rhoR_mean,phi_mean,R1, theta_c,c,nu,psi),
                       shading='auto', norm=norm,
                       cmap='viridis')
    fig.colorbar(cm, ax=ax, shrink=0.8,pad=0.1,label=r'$L_{2,max}-L_{2,min}$ (m)')    
    
    ax.set_title(rf'$\nu={nu_deg:.1f}^\circ, \psi = {psi:.0f}^\circ, \phi_0=0^\circ$')
    ax.set_xlabel(r'$\phi-\phi_0$')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position),ax.get_rmax()*1.25,r'$\rho_R$',fontsize=21,
            rotation=label_position,ha='center',va='center')
    realign_polar_xticks(ax)
    ax.grid(alpha=0.2)

def Lmaxmin2_vs_Lmaxmin_shadow_absolute(nu_deg, psi,theta_c,R1,Rhole,Des,Dpb,c,Dps,Rsb,phi_0=0):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(22.)
    
    rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    phi = np.arange(-180.,180.1,0.2, dtype= np.float64)

    rhoR_mean = 0.5*(rhoR[:-1] + rhoR[1:])     # centers, (Nr,)    
    phi_mean = 0.5*(phi[:-1] + phi[1:])        # centers, (Nθ,)    
    
    nu = np.deg2rad(nu_deg)

    Lmax1 = Lmax_M1(rhoR_mean,phi_0,nu,psi,phi_mean,theta_c,R1,c)
    
    cond_shadow = global_shadow_condition_from_M2(rhoR_mean,phi_0,nu,psi,
                                                  phi_mean,theta_c,
                                                  R1,Rsb,Dpb,Des)                   # has shape of rhoR_mean x phi_mean x 1

    #print ('cond_shadow: ',cond_shadow.shape)

    Lmax2 = Lmax_shadow_condition_from_M2(rhoR_mean,phi_0,nu,psi, 
                                         phi_mean,theta_c, 
                                         R1,Rsb,Dpb,Des)
    
    #print ('Lmax2: ', Lmax2.shape)
                         
    Lmin2 = Lmin_for_M2_shadows(rhoR_mean,phi_0,nu,psi, 
                                phi_mean,theta_c, 
                                R1,Rhole,c,
                                Rsb,Dpb,Des,Dps)
    
    #print ('Lmin2: ', Lmin2.shape)

    #Lmaxmin2 = Lmax2 - Lmin2
    Lmaxmin2 = Lmax1 - Lmax2 + Lmin2
    #Lmaxmin2 = Lmax1 - Lmax2

    
    Z_min=np.nanmin(Lmaxmin2[cond_shadow])
    Z_max=np.nanmax(Lmaxmin2)

    print ('Zmin: ',Z_min, ' Zmax: ',Z_max)
    # ------------------------------------------------------------
    norm = mcolors.LogNorm(vmin=10.,vmax=Z_max,clip=False)
    #norm = mcolors.Normalize(vmin=Z_min,vmax=Z_max,clip=False)

    # Get color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #fig.subplots_adjust(wspace=-0.5)
    handles = []

    cm = ax.pcolormesh(np.deg2rad(phi),rhoR,
                       np.squeeze(Lmaxmin2, axis=2),
                       #, # / L_noshadow(rhoR_mean,phi_mean,R1, theta_c,c,nu,psi),
                       shading='auto', norm=norm,
                       cmap='viridis')
    fig.colorbar(cm, ax=ax, shrink=0.8,pad=0.1,label=r'$L_{max}-L_{2,max}+L_{2,min}$ (m)')
    #fig.colorbar(cm, ax=ax, shrink=0.8,pad=0.1,label=r'$L_{max}-L_{2,max}$ (m)')        
    
    #cm.set_label(r'$\nu=$'+f'{nu_deg:.1f}'+r'$^{\circ}$')
    ax.set_title(rf'$\nu={nu_deg:.1f}^\circ, \psi = {psi:.0f}^\circ, \phi_0=0^\circ$')
    ax.set_xlabel(r'$\phi-\phi_0$')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position),ax.get_rmax()*1.25,r'$\rho_R$',fontsize=21,
            rotation=label_position,ha='center',va='center')
    realign_polar_xticks(ax)
    ax.grid(alpha=0.2)
            
def Lmaxmin2_vs_Lmaxmin_shadow_relative(nu_deg, psi,theta_c,R1,Rhole,Des,Dpb,c,Dps,Rsb,phi_0=0):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(22.)
    
    rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    phi = np.arange(-180.,180.1,0.2, dtype= np.float64)

    rhoR_mean = 0.5*(rhoR[:-1] + rhoR[1:])     # centers, (Nr,)    
    phi_mean = 0.5*(phi[:-1] + phi[1:])        # centers, (Nθ,)    
    
    nu = np.deg2rad(nu_deg)

    Lmax1 = Lmax_M1(rhoR_mean,phi_0,nu,psi,phi_mean,theta_c,R1,c)
    
    cond_shadow = global_shadow_condition_from_M2(rhoR_mean,phi_0,nu,psi,
                                                  phi_mean,theta_c,
                                                  R1,Rsb,Dpb,Des)                   # has shape of rhoR_mean x phi_mean x 1

    #print ('cond_shadow: ',cond_shadow.shape)

    Lmax2 = Lmax_shadow_condition_from_M2(rhoR_mean,phi_0,nu,psi, 
                                         phi_mean,theta_c, 
                                         R1,Rsb,Dpb,Des)
    
    #print ('Lmax2: ', Lmax2.shape)
    Lmax2 = np.where(cond_shadow, Lmax2, 0.)
    #print ('Lmax2: ', Lmax2.shape)
                         
    Lmin2 = Lmin_for_M2_shadows(rhoR_mean,phi_0,nu,psi, 
                                phi_mean,theta_c, 
                                R1,Rhole,c,
                                Rsb,Dpb,Des,Dps)    
    
    #print ('Lmin2: ', Lmin2.shape)
    Lmin2 = np.where(cond_shadow, Lmin2, 0.)
    #print ('Lmin2: ', Lmin2.shape)    

    #Lmaxmin2 = Lmax2 - Lmin2
    Lmaxmin2 = (Lmax1 - Lmax2 + Lmin2)/Lmax1
    #Lmaxmin2 = (Lmax1 - Lmax2)/Lmax1    
    #Lmaxmin2 = Lmax1 - Lmax2

    #print ('Lmaxmin2: ', Lmaxmin2.shape)    
    
    Z_min=np.nanmin(Lmaxmin2)
    Z_max=np.nanmax(Lmaxmin2)

    print ('Zmin: ',Z_min, ' Zmax: ',Z_max)
    # ------------------------------------------------------------
    #norm = mcolors.LogNorm(vmin=Z_min,vmax=Z_max,clip=False)
    norm = mcolors.Normalize(vmin=0.,vmax=1.,clip=False)

    # Get color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #fig.subplots_adjust(wspace=-0.5)
    handles = []
    
    cm = ax.pcolormesh(np.deg2rad(phi),rhoR,
                       np.squeeze(Lmaxmin2, axis=2),
                       #, # / L_noshadow(rhoR_mean,phi_mean,R1, theta_c,c,nu,psi),
                       shading='auto', norm=norm,
                       cmap='viridis')
    fig.colorbar(cm, ax=ax, shrink=0.8,pad=0.1,label=r'$(L_{max}-L_{2,max}+L_{2,min})/L_{max}$')
    #fig.colorbar(cm, ax=ax, shrink=0.8,pad=0.1,label=r'$L_{max}-L_{2,max}$ (m)')        
    
    #cm.set_label(r'$\nu=$'+f'{nu_deg:.1f}'+r'$^{\circ}$')
    ax.set_title(rf'$\nu={nu_deg:.1f}^\circ, \psi = {psi:.0f}^\circ, \phi_0=0^\circ$')
    ax.set_xlabel(r'$\phi-\phi_0$')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position),ax.get_rmax()*1.25,r'$\rho_R$',fontsize=21,
            rotation=label_position,ha='center',va='center')
    realign_polar_xticks(ax)
    ax.grid(alpha=0.2)
    
def Lmaxmin2_vs_LVacanti_shadow_relative(nu_deg, psi,theta_c,R1,Rhole,Des,Dpb,c,Dps,Rsb,phi_0=0):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(22.)
    
    rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    phi = np.arange(-180.,180.1,0.2, dtype= np.float64)

    rhoR_mean = 0.5*(rhoR[:-1] + rhoR[1:])     # centers, (Nr,)    
    phi_mean = 0.5*(phi[:-1] + phi[1:])        # centers, (Nθ,)    
    
    nu = np.deg2rad(nu_deg)

    Lmax1 = Lmax_M1(rhoR_mean,phi_0,nu,psi,phi_mean,theta_c,R1,c)
    
    cond_shadow = global_shadow_condition_from_M2(rhoR_mean,phi_0,nu,psi,
                                                  phi_mean,theta_c,
                                                  R1,Rsb,Dpb,Des)                   # has shape of rhoR_mean x phi_mean x 1

    #print ('cond_shadow: ',cond_shadow.shape)

    Lmax2 = Lmax_shadow_condition_from_M2(rhoR_mean,phi_0,nu,psi, 
                                         phi_mean,theta_c, 
                                         R1,Rsb,Dpb,Des)
    
    #print ('Lmax2: ', Lmax2.shape)
    Lmax2 = np.where(cond_shadow, Lmax2, 0.)
    #print ('Lmax2: ', Lmax2.shape)
                         
    Lmin2 = Lmin_for_M2_shadows(rhoR_mean,phi_0,nu,psi, 
                                phi_mean,theta_c, 
                                R1,Rhole,c,
                                Rsb,Dpb,Des,Dps)    
    
    #print ('Lmin2: ', Lmin2.shape)
    Lmin2 = np.where(cond_shadow, Lmin2, 0.)
    #print ('Lmin2: ', Lmin2.shape)    

    L_Vacanti = L_Vacanti_shadow_from_M2(rhoR_mean, phi_0,
                                         phi_mean, theta_c,
                                         R1, Rsb)
    
    #Lmaxmin2 = Lmax2 - Lmin2
    Lmaxmin2 = (Lmax2 - Lmin2 - L_Vacanti)/Lmax1
    #Lmaxmin2 = (Lmax1 - Lmax2)/Lmax1    
    #Lmaxmin2 = Lmax1 - Lmax2

    #print ('Lmaxmin2: ', Lmaxmin2.shape)    
    
    Z_min=np.nanmin(Lmaxmin2)
    Z_max=np.nanmax(Lmaxmin2)

    print ('Zmin: ',Z_min, ' Zmax: ',Z_max)
    # ------------------------------------------------------------
    #norm = mcolors.LogNorm(vmin=Z_min,vmax=Z_max,clip=False)
    norm = mcolors.Normalize(vmin=-0.5,vmax=0.5,clip=False)

    # Get color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #fig.subplots_adjust(wspace=-0.5)
    handles = []
    
    cm = ax.pcolormesh(np.deg2rad(phi),rhoR,
                       np.squeeze(Lmaxmin2, axis=2),
                       #, # / L_noshadow(rhoR_mean,phi_mean,R1, theta_c,c,nu,psi),
                       shading='auto', norm=norm,
                       cmap='viridis')
    fig.colorbar(cm, ax=ax, shrink=0.8,pad=0.1,label=r'$(L_{2,max}-L_{2,min}-L_{2,Vacanti})/L_{max}$')
    #fig.colorbar(cm, ax=ax, shrink=0.8,pad=0.1,label=r'$L_{max}-L_{2,max}$ (m)')        
    
    #cm.set_label(r'$\nu=$'+f'{nu_deg:.1f}'+r'$^{\circ}$')
    ax.set_title(rf'$\nu={nu_deg:.1f}^\circ, \psi = {psi:.0f}^\circ, \phi_0=0^\circ$')
    ax.set_xlabel(r'$\phi-\phi_0$')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position),ax.get_rmax()*1.25,r'$\rho_R$',fontsize=21,
            rotation=label_position,ha='center',va='center')
    realign_polar_xticks(ax)
    ax.grid(alpha=0.2)
            
def hole_conditions_nucycle(psi, theta_c, R1,Rhole,Des,c):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(22.)
    
    rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    phi = np.arange(-180.,180,0.2, dtype= np.float64)

    handles = []

    # Get color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #fig.subplots_adjust(wspace=-0.5)
    
    cmap = ListedColormap(["blue", "red"])
    for k, nu_deg in enumerate(np.arange(0.,(FOV/2-theta_c)+1,1.5)):
        nu = np.deg2rad(nu_deg)
        color = colors[k % len(colors)]  # cycle through colors

        idx = np.argsort(phi)
        phi_s = np.deg2rad(phi[idx])
        
        cmap = ListedColormap([(1,1,1,0), color])  # outside: transparent RGBA, inside: your color                
        cf = ax.contour(phi_s,rhoR, hole_condition_with_secondary(rhoR,phi,R1,Rhole,Des,theta_c,nu,psi,c)[:,idx],levels=[-0.5,0.5,1.5],shading='auto',colors=[color],linewidths=1) # cmap=cmap)
        # Build a legend proxy using the contour line color
        color = cf.get_edgecolor()[0]
        handles.append(Line2D([0], [0], color=color, lw=1,
                              label=rf"$\nu={nu_deg:.1f}^\circ$"))
        
        cf.set_label(r'$\nu=$'+f'{nu_deg:.1f}'+r'$^{\circ}$')
    ax.set_title(rf'$\psi = {psi:.0f}^\circ, \phi_0=0^\circ$')
    ax.set_xlabel(r'$\phi-\phi_0$')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position),ax.get_rmax()*1.25,r'$\rho_R$',fontsize=21,
            rotation=label_position,ha='center',va='center')
    realign_polar_xticks(ax)
    ax.grid(alpha=0.2)
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(-0.4,1.15),
              fontsize=18)
            
def hole_conditions_psicycle(nu_deg, theta_c, R1,Rhole,Des,c):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(22.)
    
    rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    phi = np.arange(-180.,180,0.2, dtype= np.float64)

    handles = []

    # Get color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #fig.subplots_adjust(wspace=-0.5)
    
    cmap = ListedColormap(["blue", "red"])
    for k, psi in enumerate(np.arange(0.,360,60.)):
        nu = np.deg2rad(nu_deg)
        color = colors[k % len(colors)]  # cycle through colors

        idx = np.argsort(phi)
        phi_s = np.deg2rad(phi[idx])
        
        cf = ax.contour(phi_s,rhoR, hole_condition_with_secondary(rhoR,phi,R1,Rhole,Des,theta_c,nu,psi,c)[:,idx],levels=[-0.5,0.5,1.5],shading='auto',colors=[color],linewidths=1)
        # Build a legend proxy using the contour line color
        color = cf.get_edgecolor()[0]
        handles.append(Line2D([0], [0], color=color, lw=2,
                              label=rf"$\psi={psi:.0f}^\circ$"))
        cf.set_label(r'$\psi=$'+f'{psi:.0f}'+r'$^{\circ}$')
    ax.set_title(rf"$\nu = {nu_deg:.1f}^\circ, \phi_0=0^\circ$")
    ax.set_xlabel(r'$\phi-\phi_0$')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position),ax.get_rmax()*1.25,r'$\rho_R$',fontsize=21,
            rotation=label_position,ha='center',va='center')
    realign_polar_xticks(ax)
    ax.grid(alpha=0.2)
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(-0.4,1.15),
              fontsize=18)

def shadow_conditions_nucycle(psi, theta_c, R1,Rsb,Des,Dpb,FOV):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(22.)
    
    rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    phi = np.arange(-180.,180,0.2, dtype= np.float64)

    handles = []

    # Get color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #fig.subplots_adjust(wspace=-0.5)
    
    cmap = ListedColormap(["blue", "red"])
    for k, nu_deg in enumerate(np.arange(0.,(FOV/2-theta_c)+1,1.5)):
        nu = np.deg2rad(nu_deg)
        color = colors[k % len(colors)]  # cycle through colors        
        cf = ax.contour(np.deg2rad(phi),rhoR,
                        np.squeeze(global_shadow_condition_from_M2(rhoR,0.,nu,psi,phi,theta_c,R1,Rsb,Dpb,Des),axis=2),
                        levels=[-0.5,0.5,1.5],colors=[color])
        # Build a legend proxy using the contour line color
        color = cf.get_edgecolor()[0]
        handles.append(Line2D([0], [0], color=color, lw=2,
                              label=rf"$\nu={nu_deg:.1f}^\circ$"))
        
        cf.set_label(r'$\nu=$'+f'{nu_deg:.1f}'+r'$^{\circ}$')
    ax.set_title(rf'$\psi = {psi:.0f}^\circ, \phi_0=0^\circ$')
    ax.set_xlabel(r'$\phi-\phi_0$')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position),ax.get_rmax()*1.25,r'$\rho_R$',fontsize=21,
            rotation=label_position,ha='center',va='center')
    realign_polar_xticks(ax)
    ax.grid(alpha=0.2)
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(-0.4,1.15),
              fontsize=18)
            
def shadow_conditions_psicycle(nu_deg, theta_c, R1,Rsb,Des,Dpb,FOV):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(22.)
    
    rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    phi = np.arange(-180.,180,0.2, dtype= np.float64)

    handles = []

    # Get color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #fig.subplots_adjust(wspace=-0.5)
    
    cmap = ListedColormap(["blue", "red"])
    for k, psi in enumerate(np.arange(0.,360,60.)):
        nu = np.deg2rad(nu_deg)
        color = colors[k % len(colors)]  # cycle through colors        
        cf = ax.contour(np.deg2rad(phi),rhoR,
                        np.squeeze(global_shadow_condition_from_M2(rhoR,0.,nu,psi,phi,theta_c,R1,Rsb,Dpb,Des),axis=2),
                        levels=[-0.5,0.5,1.5],colors=[color])
        # Build a legend proxy using the contour line color
        color = cf.get_edgecolor()[0]
        handles.append(Line2D([0], [0], color=color, lw=2,
                              label=rf"$\psi={psi:.0f}^\circ$"))
        cf.set_label(r'$\psi=$'+f'{psi:.0f}'+r'$^{\circ}$')
    ax.set_title(rf'$\nu = {nu_deg:.1f}^\circ, \phi_0=0^\circ$')
    ax.set_xlabel(r'$\phi-\phi_0$')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position),ax.get_rmax()*1.25,r'$\rho_R$',fontsize=21,
            rotation=label_position,ha='center',va='center')
    realign_polar_xticks(ax)
    ax.grid(alpha=0.2)
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(-0.4,1.15),
              fontsize=18)

def baffle_conditions_nucycle(psi,R1,Rsb,Des,Dpb,theta_c):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(22.)
    
    rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    phi_0 = np.arange(-180.,180,0.2, dtype= np.float64)

    handles = []

    # Get color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #fig.subplots_adjust(wspace=-0.5)
    
    #cmap = ListedColormap(["blue", "red"])
    for k, nu_deg in enumerate([1.0, 2.0, 3.0 ]): #enumerate(np.arange(1.5,(FOV/2-theta_c)+1,1.5)):
        nu = np.deg2rad(nu_deg)
        color = colors[k % len(colors)]  # cycle through colors

        idx = np.argsort(phi_0)
        phi_0s = np.deg2rad(phi_0[idx])

        cond = np.squeeze(muon_traversing_baffle_condition(rhoR,phi_0,nu,psi,R1,Rsb,Des,Dpb),axis=1)
        cmap = ListedColormap([(1,1,1,0), color])  # outside: transparent RGBA, inside: your color        

        cf = ax.contourf(phi_0s,rhoR, cond[:,idx],levels=[-0.5,0.5,1.5],cmap=cmap) # colors=["white",color] ) #, antialiased=False)
        # Build a legend proxy using the contour line color
        #color = cf.get_facecolor()[0]
        handles.append(Line2D([0], [0], color=color, lw=2,
                              label=rf"$\nu={nu_deg:.1f}^\circ$"))
        
        cf.set_label(r'$\nu=$'+f'{nu_deg:.1f}'+r'$^{\circ}$')
    ax.set_title(rf'$\psi = {psi:.0f}^\circ$')
    ax.set_xlabel(r'$\phi_0$')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position),ax.get_rmax()*1.25,r'$\rho_R$',fontsize=21,
            rotation=label_position,ha='center',va='center')
    realign_polar_xticks(ax)
    ax.grid(alpha=0.2)
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(-0.4,1.15),
              fontsize=18)

def baffle_conditions_psicycle(nu_deg,R1,Rsb,Des,Dpb,theta_c):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(22.)
    
    rhoR = np.arange(0.001,1.01,0.000995, dtype=np.float64)
    phi_0 = np.arange(-180.,180,0.2, dtype= np.float64)

    handles = []

    # Get color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #fig.subplots_adjust(wspace=-0.5)
    
    #cmap = ListedColormap(["blue", "red"])
    for k, psi in enumerate(np.arange(0.,360.,60.)): #enumerate(np.arange(1.5,(FOV/2-theta_c)+1,1.5)):
        nu = np.deg2rad(nu_deg)
        color = colors[k % len(colors)]  # cycle through colors

        idx = np.argsort(phi_0)
        phi_0s = np.deg2rad(phi_0[idx])

        cond = np.squeeze(muon_traversing_baffle_condition(rhoR,phi_0,nu,psi,R1,Rsb,Des,Dpb),axis=1)
        
        cmap = ListedColormap([(1,1,1,0), color])  # outside: transparent RGBA, inside: your color        

        cf = ax.contourf(phi_0s,rhoR, cond[:,idx],levels=[-0.5,0.5,1.5],cmap=cmap) # colors=["white",color] ) #, antialiased=False)
        # Build a legend proxy using the contour line color
        #color = cf.get_facecolor()[0]
        handles.append(Line2D([0], [0], color=color, lw=2,
                              label=rf"$\psi={psi:.0f}^\circ$"))
        
        cf.set_label(r'$\psi=$'+f'{psi:.0f}'+r'$^{\circ}$')
    ax.set_title(rf'$\nu = {nu_deg:.1f}^\circ$')
    ax.set_xlabel(r'$\phi_0$')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position),ax.get_rmax()*1.25,r'$\rho_R$',fontsize=21,
            rotation=label_position,ha='center',va='center')
    realign_polar_xticks(ax)
    ax.grid(alpha=0.2)
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(-0.4,1.15),
              fontsize=18)
