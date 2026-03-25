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

def global_shadow_condition_from_quadratic_camera(rhoR,phi0,nu,psi,   # muon parameters 
                                                  phi,theta_c,        # photon parameters 
                                                  R1,                 # primary mirror parameters
                                                  A,D):               # camera parameters
    ''' 
    Implements the shadow conditions for the quadratic camera
    
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

    M2 parameters: 

    - A:       Camera half side (see Fig. 28 of https://iopscience.iop.org/article/10.3847/1538-4365/ab2123)
               in m 
    - D:       Vertical separation of M1 pole to camera
               in m

    rhoR, phi0 and phi may be np arrays 

    returns an array of booleans with dimension (rhoR.size, phi.size, phi0.size)
    
    '''

    rhoR = np.atleast_1d(rhoR).astype(np.float64)  # (Nr,)
    phi  = np.atleast_1d(phi).astype(np.float64)   # (Nphi,)
    phi0 = np.atleast_1d(phi0).astype(np.float64)  # (Nphi0,)

    rho = R1 * rhoR[:, None, None]                 # (Nr,1,1)
    ph  = phi[None, :, None]                       # (1,Nphi,1)
    ph0 = phi0[None, None, :]                      # (1,1,Nphi0)

    
    # Angles in radians
    ang_pps = np.deg2rad(ph - psi)                 # (1,Nphi,1) if psi scalar
    ang_pp0 = np.deg2rad(ph - ph0)                 # (1,Nphi,Nphi0)
    ang_ph  = np.deg2rad(ph)
    ang_ps  = np.deg2rad(psi)
    ang_p0  = np.deg2rad(ph0)
    
    rsinph0 = rho * np.sin(ang_pp0)                # (Nr,Nphi,Nphi0)
    nsinpsi = D * nu * np.sin(ang_pps)             # psi is scalar

    sinph   = np.sin(ang_ph)               # (1,Nphi,1)
    cosph   = np.cos(ang_ph)               # (1,Nphi,1)

    Rproj   = rsinph0 + nsinpsi                    # (Nr,Nphi,Nphi0)
    cond = (np.abs(Rproj) <= A*(np.abs(sinph)+np.abs(cosph)))

    # Extra condition when rho*R1 > Rsh: require |phi - phi0| < 90 deg
    dphi = (ph - ph0 + 180.0) % 360.0 - 180.0      # (1,Nphi,Nphi0)
    extra = (np.abs(dphi) < 90.0)

    # Now the condition that the MUON impact point (ρcos⁡(ϕ_0+π), ρsin⁡(ϕ_0+π))=(−ρcos⁡ϕ_0, −ρsin⁡ϕ_0)
    # lies inside the projection of the square camera onto the mirror plane:
    
    # For the muon direction projected onto the transverse plane, the displacement from 
    # z=0 to D is (D*ν*cos⁡ψ, D*ν*sin⁡ψ), so the square camera centered at the origin in the plane 
    # z=D and ∣x∣≤A,∣y∣≤A,  projects onto  z=0 as the translated square
    # ∣x+D*ν*cos⁡ψ∣≤A,∣y+D*ν*sin⁡ψ∣≤A.

    # Therefore, the condition that the muon impact point lies within that projection is
    # ∣−ρcos⁡ϕ_0+D*ν*cos⁡ψ∣≤A  and ∣−ρsin⁡ϕ_0+D*ν*sin⁡ψ∣≤A.

    condproj = (np.abs(-rho * np.cos(ang_p0) + D*nu*np.cos(ang_ps)) < A) & (np.abs(-rho * np.sin(ang_p0) + D*nu*np.sin(ang_ps)) < A)
    
    return np.where(condproj, cond, cond & extra)


def min_to_abs(u, v):
    return (u + v - abs(u - v))/2

def max_to_abs(u, v):
    return (u + v + abs(u - v))/2

def Lmax2min2_from_quadratic_camera(rhoR,phi0,nu,psi,   # muon parameters 
                                    phi,theta_c,        # photon parameters 
                                    R1,                 # primary mirror parameters
                                    A,D):               # camera parameters
    ''' 
    Implements Lmax and Lmin for the quadratic camera
    
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

    M2 parameters: 

    - A:       Camera half side (see Fig. 28 of https://iopscience.iop.org/article/10.3847/1538-4365/ab2123)
               in m 
    - D:       Vertical separation of M1 pole to camera
               in m

    rhoR, phi0 and phi may be np arrays 

    returns two arrays of booleans with dimension (rhoR.size, phi.size, phi0.size) corresponding to Lmax and Lmin 

    !!! The condition  global_shadow_condition_from_quadratic_camera must be applied separately !!!
    
    '''

    rhoR = np.atleast_1d(rhoR).astype(np.float64)  # (Nr,)
    phi  = np.atleast_1d(phi).astype(np.float64)   # (Nphi,)
    phi0 = np.atleast_1d(phi0).astype(np.float64)  # (Nphi0,)

    rho = rhoR[:, None, None]                      # (Nr,1,1)
    ph  = phi[None, :, None]                       # (1,Nphi,1)
    ph0 = phi0[None, None, :]                      # (1,1,Nphi0)

    
    # Angles in radians
    angpsi = np.deg2rad(psi)                       # psi is scalar
    angpp0 = np.deg2rad(ph0)                       # (1,1,Nphi0)
    angphi = np.deg2rad(ph)                        # (1,Nphi,1)

    ncospsi = D * nu * np.cos(angpsi)              # psi is scalar 
    nsinpsi = D * nu * np.sin(angpsi)              # psi is scalar
    tcosphi = theta_c * np.cos(angphi)             # (1,Nphi,1)
    tsinphi = theta_c * np.sin(angphi)             # (1,Nphi,1)
    rcosph0 = R1 * rho * np.cos(angpp0)            # (Nr,1,Nphi0)
    rsinph0 = R1 * rho * np.sin(angpp0)            # (Nr,1,Nphi0)

    Lx0 = D + (ncospsi + rcosph0) / tcosphi        # (Nr,Nphi,Nphi0)
    Ly0 = D + (nsinpsi + rsinph0) / tsinphi        # (Nr,Nphi,Nphi0)

    Lx_min = Lx0 - A/np.abs(tcosphi)
    Lx_max = Lx0 + A/np.abs(tcosphi)
    Ly_min = Ly0 - A/np.abs(tsinphi)
    Ly_max = Ly0 + A/np.abs(tsinphi)

    return min_to_abs(Lx_max, Ly_max),  max_to_abs(Ly_min, Lx_min)



