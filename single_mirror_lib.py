import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def Lmax_M1(rhoR, phi0,nu,psi,phi,theta_c,R1,c):
    ''' 
    Implements the single mirror maximum length Eqs. 30 to Eq. 32
    
    Input: 
    ======
    
    Muon parameters: 
    
    - rhoR: radial distance of the muon impact point in relative units, 
            rhoR = rho/R1
    - phi0: azimquthal projection of muon impact point on primary mirror plane, 
            in deg 
    - nu:   inclination angle of muon, 
            in rad  
    - psi:  azimuthal projection of muon inclination on primary mirror plane, 
            in deg

    Photon parameters: 

    - phi:     azimuthal projection of photon emission angle on primary mirror plane, 
               in deg
    - theta_c: Cherenkov angle of emission, 
               in rad

    M1 parameters: 

    - R1:      radius of primary mirror M1  
               in m
    - c:       curvature of the mirror M1, =1/(4F) for a parabolic mirror
               in m^-1
'''

    rhoR = np.atleast_1d(rhoR).astype(np.float64)  # (Nr,)
    phi  = np.atleast_1d(phi).astype(np.float64)   # (Nphi,)
    phi0 = np.atleast_1d(phi0).astype(np.float64)  # (Nphi0,)

    rho = rhoR[:, None, None]                      # (Nr,1,1)
    ph  = phi[None, :, None]                       # (1,Nphi,1)
    ph0 = phi0[None, None, :]                      # (1,1,Nphi0)

    # Angles in radians
    ang_pps = np.deg2rad(ph - psi)                 # (1,Nphi,1) if psi scalar
    ang_pp0 = np.deg2rad(ph - ph0)                 # (1,Nphi,Nphi0)

    cospps = nu * np.cos(ang_pps)                  # (1,Nphi,1)
    sinpps = nu * np.sin(ang_pps)                  # (1,Nphi,1)
    rsinpp0 = rho * np.sin(ang_pp0)                # (Nr,Nphi,Nphi0)
    rcospp0 = rho * np.cos(ang_pp0)                # (Nr,Nphi,Nphi0)

    dphi = (ph - ph0 + 180.0) % 360.0 - 180.0      # (1,Nphi,Nphi0)    
    extra = (np.abs(dphi) < 90.0)

    sqrtarg = 1-rsinpp0**2
    sqrtcond = (sqrtarg  > 0)
    sqsinp = np.sqrt(np.maximum(sqrtarg, 1.0e-8))
    LmaxVac = np.where(sqrtcond,R1 * (rcospp0 + sqsinp) / theta_c, 0.) # Vacanti solution, np.maximum condition avoids unnecessary warnings
    
    FOcorr = c*R1**2 * (1-rho**2)                  # 0th order correction, forgotten by Vacanti 
    F1corr = FOcorr * cospps / theta_c             # 1st order correction for nu != 0
    F2corr = FOcorr * sinpps * rsinpp0 /2./sqsinp/theta_c  # 2nd order correction for nu != 0
    
    return np.where(((rho < 1.0) | extra) & sqrtcond,
                    LmaxVac + FOcorr + F1corr - F2corr,
                    0.
    )

def Lmin_M1(rhoR, phi0,nu,psi,phi,theta_c,R1,c):
    ''' 
    Implements the single mirror minimum length Eq. 33
    
    Input: 
    ======
    
    Muon parameters: 
    
    - rhoR: radial distance of the muon impact point in relative units, 
            rhoR = rho/R1
    - phi0: azimquthal projection of muon impact point on primary mirror plane, 
            in deg 
    - nu:   inclination angle of muon, 
            in rad  
    - psi:  azimuthal projection of muon inclination on primary mirror plane, 
            in deg

    Photon parameters: 

    - phi:     azimuthal projection of photon emission angle on primary mirror plane, 
               in deg
    - theta_c: Cherenkov angle of emission, 
               in rad

    M1 parameters: 

    - R1:      radius of primary mirror M1  
               in m
    - c:       curvature of the mirror M1, =1/(4F) for a parabolic mirror
               in m^-1
'''

    rhoR = np.atleast_1d(rhoR).astype(np.float64)  # (Nr,)
    phi  = np.atleast_1d(phi).astype(np.float64)   # (Nphi,)
    phi0 = np.atleast_1d(phi0).astype(np.float64)  # (Nphi0,)

    rho = rhoR[:, None, None]                      # (Nr,1,1)
    ph  = phi[None, :, None]                       # (1,Nphi,1)
    ph0 = phi0[None, None, :]                      # (1,1,Nphi0)

    # Angles in radians
    ang_pps = np.deg2rad(ph - psi)                 # (1,Nphi,1) if psi scalar
    ang_pp0 = np.deg2rad(ph - ph0)                 # (1,Nphi,Nphi0)

    cospps = nu * np.cos(ang_pps)                  # (1,Nphi,1)
    sinpps = nu * np.sin(ang_pps)                  # (1,Nphi,1)
    rsinpp0 = rho * np.sin(ang_pp0)                # (Nr,Nphi,Nphi0)
    rcospp0 = rho * np.cos(ang_pp0)                # (Nr,Nphi,Nphi0)

    dphi = (ph - ph0 + 180.0) % 360.0 - 180.0      # (1,Nphi,Nphi0)    
    extra = (np.abs(dphi) < 90.0)

    sqrtarg = 1-rsinpp0**2
    sqrtcond = (sqrtarg  > 0)
    sqsinp = np.sqrt(np.maximum(sqrtarg, 1.0e-8))
    LminVac = np.where(sqrtcond,R1 * (rcospp0 - sqsinp) / theta_c, 0.) # Vacanti solution, np.maximum condition avoids unnecessary warnings
    
    return np.where((rho < 1.0),
                    0.,
                    np.where(extra & sqrtcond,LminVac,0.)
    )

