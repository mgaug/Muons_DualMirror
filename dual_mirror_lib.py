import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def global_shadow_condition_from_M2(rhoR,phi0,nu,psi,   # muon parameters 
                                    phi,theta_c,        # photon parameters 
                                    R1,                 # primary mirror parameters
                                    Rsb,Dpb,Des):       # secondary mirror parameters
    ''' 
    Implements the shadow conditions Eq. 56 and Eq. 57 
    
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

    - Rsb:     Radius of M2, including support and baffles 
               in m 
    - Dpb:     Vertical separation of M1 to baffles
               in m
    - Des:     Vertical separation of M1 to end of M2 support structure
               in m

    rhoR, phi0 and phi may be np arrays 

    returns an array of booleans with dimension (rhoR.size, phi.size, phi0.size)
    
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
    ang_p0p = np.deg2rad(ph0 - psi)                # (1,Nphi,Nphi0)

    cospps = nu * np.cos(ang_pps)                  # (1,Nphi,1)
    sinpps = nu * np.sin(ang_pps)                  # (1,Nphi,1)
    rsinpp0 = rho * np.sin(ang_pp0)                # (Nr,Nphi,Nphi0)
    rcospp0 = rho * np.cos(ang_pp0)                # (Nr,Nphi,Nphi0)
    cosp0p = nu * np.cos(ang_p0p)

    m = (sinpps <= 0)                              # (1,Nphi,1)

    Res = Des * sinpps
    Rpb = Dpb * sinpps
    
    Dl  = np.where(m, Rpb, Res)  # (1,Nphi,1)
    Dr  = np.where(m, Res, Rpb)  # (1,Nphi,1)

    Resproj = Des * (cospps + theta_c)             # Eq. 43
    Rpbproj = Dpb * (cospps + theta_c)             # Eq. 43

    Rl2 = np.where(m,
                   np.sqrt(Rsb**2 + Rpbproj**2 + Rpb**2),
                   np.sqrt(Rsb**2 + Resproj**2 + Res**2))
    Rr2 = np.where(m,
                   np.sqrt(Rsb**2 + Resproj**2 + Res**2),
                   np.sqrt(Rsb**2 + Rpbproj**2 + Rpb**2))

    Rsh = np.sqrt(Rsb**2 + Resproj**2)   # (1,Nphi,1)

    x = R1 * rsinpp0                               # (Nr,Nphi,Nphi0)

    res = ((-Rl2 - Dl) < x) & (x < (Rr2 - Dr))     # broadcasts to (Nr,Nphi,Nphi0)
    
    # Extra condition when rho*R1 > Rsh: require |phi - phi0| < 90 deg
    dphi = (ph - ph0 + 180.0) % 360.0 - 180.0      # (1,Nphi,Nphi0)
    extra = (np.abs(dphi) < 90.0)

    cond_Lmax_M2_pos     = (Resproj + R1 * rcospp0 + np.sqrt(np.maximum(Rsh**2-x**2-2*Des*sinpps*x,0.)) > Des*theta_c)
    cond_Lmax_baffle_pos = (Rpbproj + R1 * rcospp0 + np.sqrt(np.maximum(Rsb**2+Rpbproj**2-x**2-2*Dpb*sinpps*x,0.)) > Dpb*theta_c)
    cond_Lmax_pos = (cond_Lmax_M2_pos | cond_Lmax_baffle_pos)

    #return np.where((rho * R1 < Rsh) & res & cond_Lmax_baffle_pos, 1.,0.)

    return np.where(rho * R1 < Rsh, res & cond_Lmax_pos, res & extra & cond_Lmax_pos)  # (Nr,Nphi,Nphi0)

''' old code 

    # here we do not need mpmath precision and use the faster numpy
    cospps = np.cos((phi-psi)*np.pi/180.) * nu
    sinpps = np.sin((phi-psi)*np.pi/180.) * nu

    sinpp0 = np.sin((phi-phi0)*np.pi/180.) 

    # define Dl, Dr and the Rproj:

    if (isinstance(sinpp0, (list, tuple, np.ndarray)) and isinstance(rhoR, (list, tuple, np.ndarray))):

        # helper to bring the psi-only arrays into shape
        rhoshaper = np.ones((rhoR.size,1),dtype=np.float64)
        phishaper = np.ones((1,phi.size),dtype=np.float64)          

        # both psi and rhoR may be arrays
        Dl  = rhoshaper * np.where(sinpps <= 0,Dpb * sinpps, Des * sinpps)
        Dr  = rhoshaper * np.where(sinpps <= 0,Des * sinpps, Dpb * sinpps)
        Rl2 = rhoshaper * np.where(sinpps <= 0,np.sqrt(Rsb**2 + (Dpb * (cospps + theta_c))**2),np.sqrt(Rsb**2 + (Des * (cospps + theta_c))**2))
        Rr2 = rhoshaper * np.where(sinpps <= 0,np.sqrt(Rsb**2 + (Des * (cospps + theta_c))**2),np.sqrt(Rsb**2 + (Dpb * (cospps + theta_c))**2))
            
        #print (Dl.shape)
        #print (Rl2.shape)
        #print ((rhoR[:,None]*R1*sinpp0).shape)
        
        Rsh = rhoshaper* np.sqrt(Rsb**2 + (Des * (cospps + theta_c))**2) #np.where(rhoR[:,None]*R1*sinpp0 <= 0, Rl2, Rr2) 
        res = ((-Rl2 - Dl) < rhoR[:,None]*R1*sinpp0) & (rhoR[:,None]*R1*sinpp0 <  (Rr2 - Dr))
        #print ('Cond rho', rhoR[:,None]*psishaper*R1 < Rsh)
        #print ('Cond phi', rhoR[:,None]*psishaper*R1 < Rsh)
        #print ('test', (rhoshaper*(np.abs(phi-phi0) < 90.)).shape)

        dphi = (phi[None,:] - phi0 + 180.) % 360. - 180.   # now in range (-180, 180]
        return np.where(rhoR[:,None]*phishaper*R1 < Rsh, res, np.where(rhoshaper*(np.abs(dphi) < 90.),res,False))

    elif (isinstance(sinpps, (list, tuple, np.ndarray)) and isinstance(rhoR, (list, tuple, np.ndarray))):

        # both psi and rhoR may be arrays

        # helper to bring the psi-only arrays into shape
        rhoshaper = np.ones((rhoR.size,1),dtype=np.float64)
        psishaper = np.ones((1,psi.size),dtype=np.float64)  
        Dl  = rhoshaper * np.where(sinpps <= 0,Dpb * sinpps, Des * sinpps)
        Dr  = rhoshaper * np.where(sinpps <= 0,Des * sinpps, Dpb * sinpps)
        Rl2 = rhoshaper * np.where(sinpps <= 0,np.sqrt(Rsb**2 + (Dpb * (cospps + theta_c))**2),np.sqrt(Rsb**2 + (Des * (cospps + theta_c))**2))
        Rr2 = rhoshaper * np.where(sinpps <= 0,np.sqrt(Rsb**2 + (Des * (cospps + theta_c))**2),np.sqrt(Rsb**2 + (Dpb * (cospps + theta_c))**2))

        Rsh = rhoshaper* np.sqrt(Rsb**2 + (Des * (cospps + theta_c))**2) #np.where(rhoR[:,None]*R1*sinpp0 <= 0, Rl2, Rr2)         
        res = ((-Rl2 - Dl) < R1*sinpp0*rhoR[:,None]*psishaper) & (R1*sinpp0*rhoR[:,None]*psishaper <  (Rr2 - Dr))
        #print ('Cond rho', rhoR[:,None]*psishaper*R1 < Rsh)
        #print ('Cond phi', rhoR[:,None]*psishaper*R1 < Rsh)
        dphi = (phi - phi0 + 180.) % 360. - 180.   # now in range (-180, 180]        
        return np.where(rhoR[:,None]*psishaper*R1 < Rsh, res, res*(np.abs(dphi)*rhoshaper*psishaper < 90.))

    else:
        if sinpps <= 0:
            Dl = Dpb *  sinpps
            Dr = Des *  sinpps
            # note that the square roots can never get undefined because nu << Rsb/Dpb and theta_c << Rsb/Dpb
            Rl2 = np.sqrt(Rsb**2 + (Dpb * ( cospps + theta_c))**2)
            Rr2 = np.sqrt(Rsb**2 + (Des * ( cospps + theta_c))**2)       
        else:
            Dl = Des * nu * sinpps
            Dr = Dpb * nu * sinpps
            Rl2 = np.sqrt(Rsb**2 + (Des * ( cospps + theta_c))**2)
            Rr2 = np.sqrt(Rsb**2 + (Dpb * ( cospps + theta_c))**2)

        Rsh = Rr2 #- Dl
        #print ('rhoR: ', rhoR, ' Rsh/R1', Rsh/R1, '(-Rl2 - Dl)/R1',(-Rl2 - Dl)/R1, ' (Rr2 - Dl)/R1', (Rr2 - Dr)/R1, ' sinpp0', sinpp0, ' abs(phi-phi0) : ', np.abs(phi-phi0))

        dphi = (phi - phi0 + 180.) % 360. - 180.   # now in range (-180, 180]
        
        if rhoR*R1 <= Rsh: 
            res =  ((-Rl2 - Dl) < rhoR*R1*sinpp0) and (rhoR*R1*sinpp0 < (Rr2 - Dr))
            #print ('resS', res, 'theta_c=', theta_c)            
        else:
            res =  ((-Rl2 - Dl) < rhoR*R1*sinpp0) and (rhoR*R1*sinpp0 < (Rr2 - Dr)) and (np.abs(dphi) < 90.)
            #print ('resL', res, 'theta_c=', theta_c)            

        return res
'''

def L_Vacanti_shadow_from_M2(rhoR,phi0,          # muon parameters 
                             phi,theta_c,        # photon parameters 
                             R1,                 # primary mirror parameters
                             Rsb):               # secondary mirror parameters
    ''' 
    Implements the Lmax2 for the shadow, following Vacanti's hole condition 
    
    Input: 
    ======
    
    Muon parameters: 
    
    - rhoR: radial distance of the muon impact point in relative units, 
            rhoR = rho/R1
    - phi0: azimquthal projection of muon impact point on primary mirror plane, 
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

    - Rsb:     Radius of M2, including support and baffles 
               in m 

    rhoR, phi0 and phi may be np arrays 

    returns an array of booleans with dimension (rhoR.size, phi.size, phi0.size)
    
    '''

    rhoR = np.atleast_1d(rhoR).astype(np.float64)  # (Nr,)
    phi  = np.atleast_1d(phi).astype(np.float64)   # (Nphi,)
    phi0 = np.atleast_1d(phi0).astype(np.float64)  # (Nphi0,)

    rho = rhoR[:, None, None]                      # (Nr,1,1)
    ph  = phi[None, :, None]                       # (1,Nphi,1)
    ph0 = phi0[None, None, :]                      # (1,1,Nphi0)

    # Angles in radians
    ang_pp0 = np.deg2rad(ph - ph0)                 # (1,Nphi,Nphi0)

    rsinpp0 = rho * np.sin(ang_pp0)                # (Nr,Nphi,Nphi0)
    rcospp0 = rho * np.cos(ang_pp0)                # (Nr,Nphi,Nphi0)

    # Extra condition when rho*R1 > Rsh: require |phi - phi0| < 90 deg
    dphi = (ph - ph0 + 180.0) % 360.0 - 180.0      # (1,Nphi,Nphi0)
    extra = (np.abs(dphi) < 90.0)

    RshadowR = Rsb/R1    

    sqrtarg = RshadowR**2 - rsinpp0**2

    Lmax_shadow = np.where((sqrtarg >= 0.), 
                           R1*(rcospp0 + np.sqrt(np.maximum(sqrtarg,0.0)))/theta_c,  # np.maximum avoids unnecessary warnings
                           0.)

    Lmin_shadow = np.where((sqrtarg >= 0.),
                           R1*(rcospp0 - np.sqrt(np.maximum(sqrtarg,0.0)))/theta_c,  # np.maximum avoids unnecessary warnings
                           0.)

    return np.where(rho < RshadowR,Lmax_shadow,np.where(extra, Lmax_shadow-Lmin_shadow, 0.))

def Lmax_shadow_condition_from_M2(rhoR,phi0,nu,psi,   # muon parameters 
                                  phi,theta_c,        # photon parameters 
                                  R1,                 # primary mirror parameters
                                  Rsb,Dpb,Des):       # secondary mirror parameters
    ''' 
    Implements the Lmax2 for the shadow, Eq. 44 and Eq. 59 
    
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

    - Rsb:     Radius of M2, including support and baffles 
               in m 
    - Dpb:     Vertical separation of M1 to baffles
               in m
    - Des:     Vertical separation of M1 to end of M2 support structure
               in m

    rhoR, phi0 and phi may be np arrays 

    returns an array of booleans with dimension (rhoR.size, phi.size, phi0.size)
    
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

    Resproj = Des * (cospps + theta_c)             # Eq. 43
    Rpbproj = Dpb * (cospps + theta_c)             # Eq. 43

    Rsh = np.sqrt(Rsb**2 + Resproj**2)             # Eq. 45

    # Extra condition when rho*R1 > Rsh: require |phi - phi0| < 90 deg
    dphi = (ph - ph0 + 180.0) % 360.0 - 180.0      # (1,Nphi,Nphi0)
    extra = (np.abs(dphi) < 90.0)

    x = R1 * rsinpp0                               # (Nr,Nphi,Nphi0)

    Res = Des * sinpps

    Rcond2 = np.sqrt(Rsh**2 + Res**2)
    cond_sqrt = ((-Rcond2 - Res) < x) & (x < (Rcond2 - Res)) # broadcasts to (Nr,Nphi,Nphi0)
    
    #cond_M2 = cond_sqrt  # Eqs. 46 and 47
    Lmax2_M2 = (Resproj + R1 * rcospp0 + np.sqrt(np.maximum(Rsh**2-x**2-2*Des*sinpps*x,0.)))/theta_c

    cond_M2 = (((R1 * rho < Rsh) | extra) & cond_sqrt & (Lmax2_M2 > Des))  # Eqs. 46 and 47
    #cond_M2 = (((R1 * rho < Rsh) | extra) & cond_sqrt & (Lmax2_M2 > 0.))  # Eqs. 46 and 47

    # Now the part that is shadowed by the baffles only
    
    m = (sinpps <= 0)                              # (1,Nphi,1)

    Res = Des * sinpps
    Rpb = Dpb * sinpps

    Dl  = np.where(m, Rpb, Res)  # (1,Nphi,1)
    Dr  = np.where(m, Res, Rpb)  # (1,Nphi,1)

    Rl2 = np.where(m,
                   np.sqrt(Rsb**2 + Rpbproj**2 + Rpb**2),
                   np.sqrt(Rsb**2 + Resproj**2 + Res**2))
    Rr2 = np.where(m,
                   np.sqrt(Rsb**2 + Resproj**2 + Res**2),
                   np.sqrt(Rsb**2 + Rpbproj**2 + Rpb**2))

    cond_sqrt_baffle = ((-Rl2 - Dl) < x) & (x < (Rr2 - Dr))     # broadcasts to (Nr,Nphi,Nphi0)
    
    Lmax2_baffle = (Rpbproj + R1 * rcospp0 + np.sqrt(np.maximum(Rsb**2+Rpbproj**2-x**2-2*Dpb*sinpps*x,0.)))/theta_c
    #cond_baffle = ((R1 * rho < Rsh) & cond_sqrt_baffle & (Lmax2_baffle > 0.))   # Eqs. 46 and 47
    cond_baffle = (((R1 * rho < Rsh)  | extra) & cond_sqrt_baffle & (Lmax2_baffle > Dpb))   # Eqs. 46 and 47
    
    res = np.where(cond_M2,
                   Lmax2_M2,
                   np.where(cond_baffle,
                            Lmax2_baffle,
                            0.)
    )

    ids = np.where(cond_M2 & (res <  Des))
    #print ('HERE: ',Lmax2_M2[ids])
    ids = np.where(np.logical_not(cond_M2) & cond_baffle & (res <  Dpb))
    #print ('HERE: ',Lmax2_baffle[ids])
    
    return res
                    

def muon_traversing_hole_condition(rhoR,R1,Rhole):
    ''' 
    Implements the conditions of a muon passing through the central hole
    
    Input: 
    ======
    
    Muon parameters: 
    
    - rhoR: radial distance of the muon impact point in relative units, 
            rhoR = rho/R1
    
    Photon parameters: None
    
    M1 parameters: 
    
    - R1:      radius of primary mirror M1  
               in m
    
    - Rhole:   radius of the central hole in the primary mirror M1  
               in m
    
    rhoR may be a np array
    
    returns an array of booleans with dimension (rhoR.size)

    '''

    rhoR = np.atleast_1d(rhoR).astype(np.float64)  # (Nr,)

    RholeR = Rhole/R1

    return (rhoR < RholeR)


def muon_traversing_M2_condition(rhoR,phi0,nu,psi,R1,Rsb,Des,Dpb):
    ''' 
    Implements the conditions of a muon traversing M2, Eqs. 61
    
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
    
    Photon parameters: None
    
    M1 parameters: 
    
    - R1:      radius of primary mirror M1  
               in m
    
    M2 parameters: 

    - Rsb:     Radius of M2, including support and baffles 
               in m 
    - Des:     Vertical separation of M1 to end of M2 support structure
               in m
    - Dpb:     Vertical separation of M1 to baffles
               in m
    
    rhoR and phi0 may be np arrays
    
    returns an array of booleans with dimension (rhoR.size, 1, phi0.size)

    '''

    rhoR = np.atleast_1d(rhoR).astype(np.float64)  # (Nr,)
    phi0 = np.atleast_1d(phi0).astype(np.float64)  # (Nphi0,)

    rho = rhoR[:, None, None]                      # (Nr,1,1)
    ph0 = phi0[None, None, :]                      # (1,1,Nphi0)

    # Angles in radians
    ang_pps = np.deg2rad(ph0 - psi)                # (1,1,Nphi0) if psi scalar

    cospps = nu * np.cos(ang_pps)                  # (1,1,Nphi0)
    sinpps = nu * np.sin(ang_pps)                  # (1,1,Nphi0)

    rho_es = (Rsb - Des * cospps)

    return (R1*rho < rho_es)
    

''' old code 
    cospps = np.cos((phi_0-psi)*np.pi/180.) 
    
    if (isinstance(phi_0, (list, tuple, np.ndarray)) and isinstance(rhoR, (list, tuple, np.ndarray))):

        # helper to bring the phi_0-only arrays into shape
        rhoshaper = np.ones((rhoR.size,1),dtype=np.float64)
        phishaper = np.ones((1,phi_0.size),dtype=np.float64)          

        rho_es = rhoshaper * (Rsb - Des * nu * cospps)

        res = (rhoR[:,None]*R1*phishaper < rho_es)
        return res
        
    elif (isinstance(rhoR, (list, tuple, np.ndarray))):

        rho_es = Rsb - Des * nu * cospps

        res = (rhoR*R1 < rho_es)
        return res
        
    else:

        rho_es = Rsb - Des * nu * np.cos((phi_0-psi)*np.pi/180.)
        return (rhoR*R1 < rho_es)
'''


def muon_traversing_baffle_condition(rhoR,phi0,nu,psi,
                                     R1,
                                     Rsb,Des,Dpb):
    ''' 
    Implements the conditions of a muon traversing the protective baffle of M2, Eqs. 63 and 64 
    
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
    
    Photon parameters: None
    
    M1 parameters: 
    
    - R1:      radius of primary mirror M1  
               in m
    
    M2 parameters: 

    - Rsb:     Radius of M2, including support and baffles 
               in m 
    - Des:     Vertical separation of M1 to end of M2 support structure
               in m
    - Dpb:     Vertical separation of M1 to baffles
               in m

    
    rhoR and phi0 may be np arrays
    
    returns an array of booleans with dimension (rhoR.size, 1, phi0.size)

    '''

    rhoR = np.atleast_1d(rhoR).astype(np.float64)  # (Nr,)
    phi0 = np.atleast_1d(phi0).astype(np.float64)  # (Nphi0,)

    rho = rhoR[:, None, None]                      # (Nr,1,1)
    ph0 = phi0[None, None, :]                      # (1,1,Nphi0)

    # Angles in radians
    ang_pps = np.deg2rad(ph0 - psi)                # (1,1,Nphi0) if psi scalar

    cospps = nu * np.cos(ang_pps)                  # (1,1,Nphi0)
    sinpps = nu * np.sin(ang_pps)                  # (1,1,Nphi0)

    m = (cospps <= 0)                              # (1,1,Nphi0)

    rho_l = np.where(m,
                     Rsb - Dpb * cospps,    
                     Rsb - Des * cospps)           # (1,1,Nphi0)

    rho_r = np.where(m,
                     Rsb - Des * cospps,
                     Rsb - Dpb * cospps)           # (1,1,Nphi0)

    x = R1 * rho                                   # (Nr, 1, 1)
    
    res = ((rho_l < x) & (x < rho_r))              # broadcasts to (Nr, 1, Nphi0)
    return res

''' old code     
    if (isinstance(phi_0, (list, tuple, np.ndarray)) and isinstance(rhoR, (list, tuple, np.ndarray))):

        # helper to bring the phi_0-only arrays into shape
        rhoshaper = np.ones((rhoR.size,1),dtype=np.float64)
        phishaper = np.ones((1,phi_0.size),dtype=np.float64)          

        rho_l = rhoshaper * np.where(cospps <= 0, Rsb - Dpb * nu * np.cos((phi_0-psi)*np.pi/180.), Rsb - Des * nu * np.cos((phi_0-psi)*np.pi/180.))
        rho_r = rhoshaper * np.where(cospps <= 0, Rsb - Des * nu * np.cos((phi_0-psi)*np.pi/180.), Rsb - Dpb * nu * np.cos((phi_0-psi)*np.pi/180.))

        print (rho_l.shape)
        print ((rhoR[:,None]*R1*phishaper).shape)

        res = ((rho_l < rhoR[:,None]*R1*phishaper) & (rhoR[:,None]*R1*phishaper < rho_r))
        print (res.shape)
        
        return res
'''

def Lmin_from_muon_passing_hole(rhoR,phi0,nu,psi,phi,theta_c,R1,Rhole,c,Des):
    ''' 
    Implements the calculation of L_2,min for a muon passing through the central hole, 
    Eqs. 66 and 67
    
    Input: 
    ======
    
    Muon parameters: 
    
    - rhoR: radial distance of the muon impact point in relative units, 
            rhoR = rho/R1
    - phi0: azimuthal projection of muon impact point on primary mirror plane, 
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

    - Rhole:   radius of the central hole in the primary mirror M1  
               in m

    - c:       curvature of the mirror M1, =1/(4F) for a parabolic mirror
               in m^-1

    M2 parameters: 

    - Des:     Vertical separation of M1 to end of M2 support structure
               in m

    rhoR, phi0 and phi may be np arrays 

    returns an array of floats with dimension (rhoR.size, phi.size, phi0.size)
    
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

    RholeR = Rhole/R1

    sqrtarg = RholeR**2 - rsinpp0**2
    
    Lmaxh   = np.where(sqrtarg > 0.,R1*(rcospp0 + np.sqrt(np.maximum(sqrtarg, 0.0)))/theta_c,2*Des)   # np.maximum condition avoid unnecessary warnings
    LmaxhF1 = np.where(sqrtarg > 0.,R1**2 * c* (RholeR**2-rho**2) * (1. + nu/theta_c * cospps),2*Des)
    #print ((Lmaxh+LmaxhF1).shape)
    #print (D.shape)
    #print ('rhoR=',rhoR)
    #print ('RholeR=',RholeR)
    #print (rhoR[:,None]*phishaper)        
    #print ('sqrt=',np.sqrt(RholeR**2-(rhoR[:,None]*sinpp0)**2))
    #print ('Lmaxh',Lmaxh)
    #res = (Lmaxh <  D)
        
    res = np.where((Lmaxh + LmaxhF1 < Des),Des - Lmaxh - LmaxhF1, 0.)  # light emission condition below secondary if muon passes through hole 
    extra = (rho < RholeR)
    
    return np.where(extra,res,0.)

''' old code 
    elif (isinstance(sinpps, (list, tuple, np.ndarray)) and isinstance(rhoR, (list, tuple, np.ndarray))):

        # both psi and rhoR may be arrays

        # helper to bring the psi-only arrays into shape
        rhoshaper = np.ones((rhoR.size,1),dtype=np.float64)
        psishaper = np.ones((1,psi.size),dtype=np.float64)  

        D   = rhoshaper * psishaper * Des 

        Lmaxh   = R1*(rhoR*cospp0 + np.sqrt(RholeR**2 - (rhoR*sinpp0)**2))/theta_c
        LmaxhF1 = R1**2 * c* (RholeR**2-rhoR[:,None]**2) * (1. + nu/theta_c*(cospp0))

        res = np.where((Lmaxh[:,None]*psishaper + LmaxhF1 < D),D - Lmaxh[:,None]*psishaper - LmaxhF1, 0.)  # light emission condition below secondary if muon passes through hole         
        return np.where(rhoR[:,None]*psishaper < RholeR,res,0.)        

    else:

        Lmaxh   = R1*(rhoR*cospp0 + np.sqrt(RholeR**2 - (rhoR*sinpp0)**2))/theta_c
        LmaxhF1 = R1**2 * c* (RholeR**2-rhoR**2) * (1. + nu/theta_c*(cospp0))

        if rhoR < RholeR and (Lmaxh + LmaxhF1) < Des:
            return Des - Lmaxh - LmaxhF1
        return 0.
'''
    
def Lmin_hole_other_light_losses(rhoR,phi0,nu,psi,phi,theta_c,R1,Rhole,c,Dps):
    ''' 
    Implements Lmin for the case that the muon passes through M2, but not the central M1 mirror hole, 
    Eqs. 68 to 70
    
    Input: 
    ======
    
    Muon parameters: 
    
    - rhoR: radial distance of the muon impact point in relative units, 
            rhoR = rho/R1
    - phi0: azimuthal projection of muon impact point on primary mirror plane, 
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

    - Rhole:   radius of the central hole in the primary mirror M1  
               in m

    - c:       curvature of the mirror M1, =1/(4F) for a parabolic mirror
               in m^-1

    M2 parameters: 

    - Dps:     Vertical separation of M1 to M2 
               in m

    rhoR, phi0 and phi may be np arrays 

    returns an array of floats with dimension (rhoR.size, phi.size, phi0.size)
    
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

    rsinpp0 = rho * np.sin(ang_pp0)                # (Nr,Nphi,Nphi0)
    rcospp0 = rho * np.cos(ang_pp0)                # (Nr,Nphi,Nphi0)
    cospps  = nu * np.cos(ang_pps)                 # (1,Nphi,1)
    sinpps  = nu * np.sin(ang_pps)                 # (1,Nphi,1)

    RholeR = Rhole/R1
    
    #print (rhoshaper.shape)
    #print (phi.shape)
    #print ((rhoR[:,None]*R1*sinpp0).shape)

    dphi = (ph - ph0 + 180.0) % 360.0 - 180.0      # (1,Nphi,Nphi0)    
    extra = (np.abs(dphi) < 90.0)

    sqrtarg = RholeR**2 - rsinpp0**2
    
    Lminh = np.where((sqrtarg >= 0.) & extra,
                     R1*(rcospp0 - np.sqrt(np.maximum(sqrtarg,0.0)))/theta_c,  # np.maximum avoids unnecessary warnings
                     2*Dps)

    res = np.where((Lminh <  Dps), Lminh, Dps) 
        
    return np.where(rho >= RholeR,res,0.)

''' old code 
    elif (isinstance(rhoR, (list, tuple, np.ndarray))):

        # both psi and rhoR may be arrays

        # helper to bring the psi-only arrays into shape
        rhoshaper = np.ones((rhoR.size,1),dtype=np.float64)

        # both psi and rhoR may be arrays
        D   = rhoshaper * Dps
            
        #print (rhoshaper.shape)
        #print (phi.shape)
        #print ((rhoR[:,None]*R1*sinpp0).shape)
        
        dphi = (phi - phi0 + 180.) % 360. - 180.   # now in range (-180, 180]        
        Lminh = np.where((rhoR*np.abs(sinpp0) < RholeR) & ((np.abs(dphi) < 90.)),
                         R1*(rhoR*cospp0 - np.sqrt(RholeR**2 - (rhoR*sinpp0)**2))/theta_c,
                         2*D)

        res = np.where((Lminh <  D), Lminh, D) 
        
        return np.where(rhoR >= RholeR,res,0.)

    else:

        dphi = (phi - phi0 + 180.) % 360. - 180.   # now in range (-180, 180]

        if (rhoR*np.abs(sinpp0) < RholeR) and ((np.abs(dphi) < 90.)):
            Lminh = R1*(rhoR*cospp0 - np.sqrt(RholeR**2 - (rhoR*sinpp0)**2))/theta_c

            if Lminh <  Dps:
                return Lminh
            else:
                return Dps

        return 0.
'''


def Lmin_light_losses_no_M2_traverse(rhoR,phi0,nu,psi,
                                     phi,theta_c,
                                     R1,
                                     Rsb,Des):
    ''' 
    Implements Lmin for the case that the muon does *not* traverse M2
    Eq. 72
    
    Input: 
    ======
    
    Muon parameters: 
    
    - rhoR: radial distance of the muon impact point in relative units, 
            rhoR = rho/R1
    - phi0: azimuthal projection of muon impact point on primary mirror plane, 
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

    - Rsb:     Radius of M2, including support and baffles 
               in m 
    - Des:     Vertical separation of M1 to end of M2 support structure
               in m

    rhoR, phi0 and phi may be np arrays 

    returns an array of floats with dimension (rhoR.size, phi.size, phi0.size)
    
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

    rsinpp0 = rho * np.sin(ang_pp0)                # (Nr,Nphi,Nphi0)
    rcospp0 = rho * np.cos(ang_pp0)                # (Nr,Nphi,Nphi0)
    cospps  = nu * np.cos(ang_pps)                 # (1,Nphi,1)
    sinpps  = nu * np.sin(ang_pps)                 # (1,Nphi,1)

    sqrtarg = Rsb**2 + (Des * (cospps + theta_c))**2 - (R1*rsinpp0)**2 - 2*Des*R1*rsinpp0*sinpps

    Lmin = np.where(sqrtarg >= 0.,
                    Des + (R1*rcospp0 + Des * cospps - np.sqrt(np.maximum(sqrtarg,0)))/theta_c,
                    0.)
    
    return np.where(Lmin  >= 0.,Lmin, 0.)

''' old code    
    if (isinstance(sinpp0, (list, tuple, np.ndarray)) and isinstance(rhoR, (list, tuple, np.ndarray))):

        # helper to bring the psi-only arrays into shape
        rhoshaper = np.ones((rhoR.size,1),dtype=np.float64)
        phishaper = np.ones((1,phi.size),dtype=np.float64)          

        # both phi and rhoR may be arrays
        D   = rhoshaper * phishaper * Des

        sqrtarg = Rsb**2 + rhoshaper * (Des * (cospps + theta_c))**2 - (rhoR[:,None]*R1*sinpp0)**2 - 2*Des*R1*rhoR[:,None]*sinpps*sinpp0
        
        #print (rhoshaper.shape)
        #print (phi.shape)
        #print ((rhoR[:,None]*R1*sinpp0).shape)
        
        Lminh   = np.where(sqrtarg >= 0.,
                           D + (R1*rhoR[:,None]*cospp0 + Des * rhoshaper * cospps - np.sqrt(sqrtarg))/theta_c,
                           0.)

        return Lminh

    elif (isinstance(rhoR, (list, tuple, np.ndarray))):

        # both psi and rhoR may be arrays

        # helper to bring the psi-only arrays into shape
        rhoshaper = np.ones((rhoR.size,1),dtype=np.float64)

        D   = rhoshaper * Des

        sqrtarg = Rsb**2 + rhoshaper * (Des * (cospps + theta_c))**2 - (rhoR*R1*sinpp0)**2 - 2*Des*R1*rhoR*sinpps*sinpp0
        
        #print (rhoshaper.shape)
        #print (phi.shape)
        #print ((rhoR[:,None]*R1*sinpp0).shape)
        
        Lminh   = np.where(sqrtarg >= 0.,
                           D + (R1*rhoR*cospp0 + Des * rhoshaper * cospps - np.sqrt(sqrtarg))/theta_c,
                           0.)

        return Lminh

    else:

        sqrtarg = Rsb**2 + (Des * (cospps + theta_c))**2 - (rhoR*R1*sinpp0)**2 - 2*Des*R1*rhoR*sinpps*sinpp0
        
        #print (rhoshaper.shape)
        #print (phi.shape)
        #print ((rhoR[:,None]*R1*sinpp0).shape)
        
        if sqrtarg >= 0.:
            return Des + (R1*rhoR*cospp0 + Des * cospps - np.sqrt(sqrtarg))/theta_c

        return 0.
'''


def muon_baffle_crossing_vertical_distance(rhoR,phi0,nu,psi,R1,Rsb,Des,Dpb):
    ''' 
    Implements the vertical distance from M1 at the point where a muon passing through the baffles, 
    Eq. 65
    
    Input: 
    ======
    
    Muon parameters: 
    
    - rhoR: radial distance of the muon impact point in relative units, 
            rhoR = rho/R1
    - phi0: azimuthal projection of muon impact point on primary mirror plane, 
            in deg 
    - nu:   inclination angle of muon, 
            in rad  
    - psi:  azimuthal projection of muon inclination on primary mirror plane, 
            in deg
    
    Photon parameters: None
    
    M1 parameters: 
    
    - R1:      radius of primary mirror M1  
               in m

    M2 parameters: 
    
    - Rsb:     Radius of M2, including support and baffles 
               in m 
    - Des:     Vertical separation of M1 to end of M2 support structure
               in m
    - Dpb:     Vertical separation of M1 to baffles
               in m
    
    rhoR may be a np array
    
    returns an array of floats with dimension (rhoR.size, 1, phi0.size)
 
    '''

    rhoR = np.atleast_1d(rhoR).astype(np.float64)  # (Nr,)
    phi0 = np.atleast_1d(phi0).astype(np.float64)  # (Nphi0,)

    rho = rhoR[:, None, None]                      # (Nr,1,1)
    ph0 = phi0[None, None, :]                      # (1,1,Nphi0)

    # Angles in radians
    ang_p0ps = np.deg2rad(ph0 - psi)               # (1,1,Nphi0) if psi scalar
    
    sinp0ps = nu * np.sin(ang_p0ps)              # (1,1,Nphi0)
    cosp0ps = nu * np.cos(ang_p0ps)              # (1,1,Nphi0)
    rsinp0ps = rho * np.sin(ang_p0ps)            # (Nr,1,Nphi0)
    rcosp0ps = rho * np.cos(ang_p0ps)            # (Nr,1,Nphi0)

    RsbR   = Rsb/R1

    m = (cosp0ps <= 0)                           # (1,1,Nphi0)

    rho_l = np.where(m,
                     Rsb - Dpb * cosp0ps,    
                     Rsb - Des * cosp0ps)           # (1,1,Nphi0)

    sqrtarg = RsbR**2 - rsinp0ps**2

    mm = (R1*rho <= Rsb)
    sqrtcond = (sqrtarg >= 0.)
    
    return np.where(mm & sqrtcond,
                    R1*(-rcosp0ps + np.sqrt(np.maximum(sqrtarg, 0.0)))/nu,
                    np.where(sqrtcond,
                             R1 * (-rcosp0ps - np.sqrt(np.maximum(sqrtarg, 0.0)))/nu,
                             0.
                    )
    )
    # delta_rho = R1*rho - rho_l
    #return np.where(delta_rho > 0., delta_rho / nu, 0.)

''' old code 
    if (isinstance(phi_0, (list, tuple, np.ndarray)) and isinstance(rhoR, (list, tuple, np.ndarray))):

        # helper to bring the phi_0-only arrays into shape
        rhoshaper = np.ones((rhoR.size,1),dtype=np.float64)
        phishaper = np.ones((1,phi_0.size),dtype=np.float64)          

        Dc = np.where(RsbR**2 - (rhoR[:,None]*sinpps)**2 > 0.,(-R1*rhoR[:,None]*cospps + R1*np.sqrt(RsbR**2 - (rhoR[:,None]*sinpps)**2))/nu,0.)        
        return Dc
        
    elif (isinstance(rhoR, (list, tuple, np.ndarray))):

        # helper to bring the phi_0-only arrays into shape
        rhoshaper = np.ones((rhoR.size,1),dtype=np.float64)

        Dc = np.where(RsbR**2 - (rhoR*sinpps)**2 > 0.,(-R1*rhoR*cospps + R1*np.sqrt(RsbR**2 - (rhoR*sinpps)**2))/nu,0.)        
        return Dc
        
    else:
        if RsbR**2 - (rhoR*sinpps)**2 > 0:
            return (-R1*rhoR*cospps + R1*np.sqrt(RsbR**2 - (rhoR*sinpps)**2))/nu

        return 0.
'''


def Lmin_from_hole(rhoR,phi0,nu,psi,phi,theta_c,R1,Rhole,c,Des,Dps):
    ''' 
    Implements calculation of Lmin for the case that the muon passes through M2, 
    and Cherenkov light losses from the central hole are obtained, 
    Eqs. 66 to 70
    
    Input: 
    ======
    
    Muon parameters: 
    
    - rhoR: radial distance of the muon impact point in relative units, 
            rhoR = rho/R1
    - phi0: azimuthal projection of muon impact point on primary mirror plane, 
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

    - Rhole:   radius of the central hole in the primary mirror M1  
               in m

    - c:       curvature of the mirror M1, =1/(4F) for a parabolic mirror
               in m^-1

    M2 parameters: 

    - Des:     Vertical separation of M1 to end of M2 support structure
               in m
    - Dps:     Vertical separation of M1 to M2 
               in m

    rhoR, phi0 and phi may be np arrays 

    returns an array of floats with dimension (rhoR.size, phi.size, phi0.size)
    
    '''

    return np.where(muon_traversing_hole_condition(rhoR,R1,Rhole)[:,None,None],   # has shape of rhoR --> broadcasted to (rhoR.size,phi.size,phi0.size)
                    Lmin_from_muon_passing_hole(rhoR,phi0,nu,psi,phi,theta_c,R1,Rhole,c,Des),
                    Lmin_hole_other_light_losses(rhoR,phi0,nu,psi,phi,theta_c,R1,Rhole,c,Dps)
    )

def Lmin_from_nobaffle(rhoR,phi0,nu,psi,phi,theta_c,R1,Rhole,c,Rsb,Des,Dpb,Dps):
    ''' 
    Implements calculation of Lmin for the case that the muon passes through M2, 
    but not through the protecting baffles, 
    Eqs. 61 to 70
    
    Input: 
    ======
    
    Muon parameters: 
    
    - rhoR: radial distance of the muon impact point in relative units, 
            rhoR = rho/R1
    - phi0: azimuthal projection of muon impact point on primary mirror plane, 
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

    - Rhole:   radius of the central hole in the primary mirror M1  
               in m

    - c:       curvature of the mirror M1, =1/(4F) for a parabolic mirror
               in m^-1

    M2 parameters: 

    - Rsb:     Radius of M2, including support and baffles 
               in m 
    - Des:     Vertical separation of M1 to end of M2 support structure
               in m
    - Dps:     Vertical separation of M1 to M2 
               in m
    - Dpb:     Vertical separation of M1 to baffle/support structure of M2 
               in m

    rhoR, phi0 and phi may be np arrays 

    returns an array of floats with dimension (rhoR.size, phi.size, phi0.size)
    
    '''

    cond_M2 = muon_traversing_M2_condition(rhoR,phi0,nu,psi,R1,Rsb,Des,Dpb)      # has shape of rhoR

    return np.where(cond_M2,
                    Lmin_from_hole(rhoR,phi0,nu,psi,phi,theta_c,R1,Rhole,c,Des,Dps),
                    Lmin_light_losses_no_M2_traverse(rhoR,phi0,nu,psi,phi,theta_c,R1,Rsb,Des)
                    )

def Lmin_for_M2_shadows(rhoR,phi0,nu,psi,phi,theta_c,R1,Rhole,c,Rsb,Dpb,Des,Dps):
    ''' 
    Implements calculation of Lmin for the case that the muon passes through M2, 
    or the baffles, Eqs. 61 through 70
    
    Input: 
    ======
    
    Muon parameters: 
    
    - rhoR: radial distance of the muon impact point in relative units, 
            rhoR = rho/R1
    - phi0: azimuthal projection of muon impact point on primary mirror plane, 
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

    - Rhole:   radius of the central hole in the primary mirror M1  
               in m

    - c:       curvature of the mirror M1, =1/(4F) for a parabolic mirror
               in m^-1

    M2 parameters: 

    - Rsb:     Radius of M2, including support and baffles 
               in m 
    - Des:     Vertical separation of M1 to end of M2 support structure
               in m
    - Dps:     Vertical separation of M1 to M2 
               in m
    - Dpb:     Vertical separation of M1 to baffle/support structure of M2 
               in m

    rhoR, phi0 and phi may be np arrays 

    returns an array of floats with dimension (rhoR.size, phi.size, phi0.size)
    
    '''
            
    return np.where(muon_traversing_baffle_condition(rhoR,phi0,nu,psi,R1,Rsb,Des,Dpb),   # has shape of rhoR_mean x 1 x 1
                    muon_baffle_crossing_vertical_distance(rhoR,phi0,nu,psi,R1,Rsb,Des,Dpb),    # Dc in the article
                    Lmin_from_nobaffle(rhoR,phi0,nu,psi,phi,theta_c,R1,Rhole,c,Rsb,Des,Dpb,Dps)
                    )
