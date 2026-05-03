import math
import numpy as np

class Telescope:
    def __init__(self, name,
                 R1, Rhole,
                 FOV, F, Omega_pix, Npix,
                 R2=None,   
                 Rsb=None, Des=None, Dpb=None,
                 Dps=None, Dscam=None,
                 alpha=None, demag=None,
                 Dpcam=None, Acam=None):

        self.name = name

        # Primary geometry
        self.R1 = R1          # Outer radius of M1 , in m 
        self.Rhole = Rhole    # Outer radius of hole in M1, in m 

        # Optical parameters
        self.FOV = FOV        # Field-of-view of the camera, in deg
        self.F = F            # Focal length of the full telescope, in m
        self.Omega_pix = np.deg2rad(Omega_pix) # Field-of-view of indiv. pixel in rad
        self.Npix = Npix

        # Secondary / support structure
        self.R2 = R2          # Outer radius of M2, in m        

        self.Rsb = Rsb        # Outer radius of full M2 support structure, including baffles
        self.Des = Des        # Separation primary - end of support structure, in m 
        self.Dpb = Dpb        # Separation primary - baffles/support structure, in m

        self.Dps = Dps        # Separation primary-secondary mirror (pole to pole), in m
        self.Dscam = Dscam    # Separation secondary-focal plane, (pole to pole) in m 

        self.Dpcam = Dpcam    # Separation primary-focal plane, (pole to pole) in m
        self.Acam = Acam      # Half-side length of square camera, in m 

        self.alpha = alpha    # Ratio of Dps and Fp, see Fig. 2 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527
        self.demag = demag    # De-magnification of the secondary mirror (= 1+eta), see Eq. 7 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527

        # Derived quantities for dual-mirror telescopes
        if demag is None: 
            self.Fp = self.F
        else:
            self.Fp = self.F * self.demag  # Focal length of M1, in m, see Eq. 7 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527

        # Focal length M2, in m, see Eq. (13) of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527
        if demag is None or alpha is None:
            self.Fs = None
        else:
            # Focal length M2, in m, see Eq. (13) of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527            
            self.Fs = (1 - self.alpha) * self.Fp / (self.demag - 1)

    def print(self):
        print(f"\nTelescope: {self.name}")
        print(f"Primary mirror radius: {self.R1:.2f} m")
        print(f"Primary hole radius: {self.Rhole:.2f} m")
        print(f"Focal length (system): {self.F:.2f} m")
        print(f"Field of view: {self.FOV:.2f} deg")

        if self.Dscam is not None:
            print(f"Separation M1 pole to focal plane: {self.Dscam:.2f} m")            
        if self.Acam is not None:
            print(f"Half-side length of square camera: {self.Acam:.2f} m")            
        
        if self.R2 is not None:
            print("Primary focal length: ", self.Fp," m")
            print("Secondary mirror radius: ", self.R2," m")
            print("Secondary focal length: ", self.Fs," m")
            print("Outer radius of full M2 support structure, including baffles: ",self.Rsb," m")
            print("Separation primary - end of support structure: ",self.Des," m")
            print("Separation primary - baffles/support structure: ",self.Dpb," m")
            print("Separation primary-secondary mirror (pole to pole): ",self.Dps," m")
            print("Separation secondary-focal plane: ",self.Dscam," m")
            print("alpha=",self.alpha)
            print("De-magnification secondary (1+eta): ",self.demag)
        


# ---------- Factory constructors ----------

def SST():
    return Telescope(
        name="SST",
        R1=2.03,      # Outer radius of M1 , in m        
        Rhole=0.48,   # Outer radius of hole in M1, in m 
        FOV=10.5,     # Field-of-view of the camera, in deg    
        F=2.15,       # Focal length of the full telescope, in m
        Omega_pix=0.2,# Field-of-view of pixel, in deg
        Npix=1550,    # Number of pixels
        R2=0.9,       # Outer radius of M2, in m        
        Rsb=1.07,     # Outer radius of full M2 support structure, including baffles
        Des=3.0,      # Separation primary - end of support structure, in m 
        Dpb=2.82,     # Separation primary - baffles/support structure, in m
        Dps=3.11,     # Separation primary-secondary mirror (pole to pole), in m
        Dscam=0.52,   # Separation secondary-focal plane, in m 
        alpha=0.758,  # Ratio of Dps and Fp, see Fig. 2 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527
        demag=1.84    # De-magnification of the secondary mirror (= 1+eta), see Eq. 7 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527
    )


def SCT():
    F = 5.5863

    return Telescope(
        name="SCT", 
        R1=4.83,      # Outer radius of M1 , in m       
        Rhole=2.19,   # Outer radius of hole in M1, in m
        FOV=8.2,      # Field-of-view of the camera, in deg     
        F=F,          # Focal length of the full telescope, in m
        Omega_pix=0.067,# Field-of-view of pixel, in deg
        Npix=11328,   # Number of pixels        
        R2=2.71,      # Outer radius of M2, in m        
        Rsb=2.80,     # The following number has been guessed from Figure 7 of https://pos.sissa.it/236/1029
        Des=10.0,     # These numbers have been obtained from MC Telescope Models
        Dpb=6.80,     # Separation primary - baffles/support structure, in m
        Dps=3/2 * F,  # Separation primary-secondary mirror (pole to pole), yields 8.38 m, the factor 3/2 is accurate by design
        Dscam=F / 3,  # Separation secondary-focal plane, yields 1.86 m, the factor 1/3 is accurate by design
        alpha=2/3,    # Ratio of Dps and Fp, see Fig. 2 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527 
        demag=9/4     # De-magnification of the secondary mirror (= 1+eta), see Eq. 7 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527, 9/4 is accurate by design
    )

def LST():

    R1 = 23/2.
    Dfacet = 1.51     #  flat-to-flat distance of one facet 
    
    return Telescope(
        name="LST",
        R1 = R1,
        Rhole = math.sqrt(2*math.sqrt(3.)*(Dfacet/2)**2/math.pi)  ,   # average radius of hexagon from flat-to-flat distance, such that r^2*pi = A_hexagon, see 10.1063/1.4969022
        FOV = 4.5,    # see 10.1063/1.4969022
        F = 1.2*R1*2, # see f/D from 10.1063/1.4969022
        Omega_pix=0.1,# Field-of-view of pixel, in deg
        Npix=1855,    # Number of pixels        
        Dpcam = 28.0,
        Acam = 3.0/2
        )

def MST():

    R1 = 12.3/2
    Dfacet = 1.2     #  flat-to-flat distance of one facet, see 10.1117/12.2055395
    
    return Telescope(
        name="MST",
        R1 = R1,
        Rhole = math.sqrt(2*math.sqrt(3.)*(Dfacet/2)**2/math.pi)  ,   # average radius of hexagon from flat-to-flat distance, such that r^2*pi = A_hexagon, see 10.1063/1.4969022
        FOV = 4.5,    # see 10.1063/1.4969022
        F = 16.0,     # see MC parameter description: MST Structure
        Omega_pix=0.175,# Field-of-view of pixel, in deg
        Npix=1800,    # Number of pixels        
        Dpcam = 16.0,
        Acam = 3.0/2
        )

'''
# SST
R1 = 2.03                    # Outer radius of M1 , in m 
Rhole = 0.48                 # Outer radius of hole in M1, in m 
R2 = 0.9                     # Outer radius of M2, in m        

Dps = 3.11                   # Separation primary-secondary mirror (pole to pole), in m
# The following three numbers have been obtained from technical drawings provided by Amaya Paredes
Rsb = 1.07                   # Outer radius of full M2 support structure, including baffles
Des = 3                      # Separation primary - end of support structure, in m
Dpb = 2.82                   # Separation primary - baffles/support structure, in m
                             
FOV = 10.5                   # Field-of-view of the camera, in deg
F = 2.15                     # Focal length of the full telescope, in m
Dscam = 0.52                 # Separation secondary-focal plane, in m 
# Dscam = (1-alpha) * F -- alpha = 0.758
alpha = 0.758                # Ratio of Dps and Fp, see Fig. 2 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527
# -- alpha * Fp = Dsp = alpha * (1+eta)* F --  (1+eta) = Dsp/alpha / F = 1.84
demag = 1.84                 # De-magnification of the secondary mirror (= 1+eta), see Eq. 7 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527
Fp = F * demag               # Focal length of M1, in m, see Eq. 7 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527
Fs = (1-alpha)*Fp/(demag-1)  # Focal length M2, in m, see Eq. (13) of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527

# SCT
R1 = 4.83                    # Outer radius of M1 , in m       
Rhole = 2.19                 # Outer radius of hole in M1, in m
R2 = 2.71                    # Outer radius of M2, in m        

# The following number has been guessed from Figure 7 of https://pos.sissa.it/236/1029
Rsb = 2.80                   # Outer radius of full M2 support structure, including baffles
# These numbers have been obtained from MC Telescope Models
Des = 10.0                   # Separation primary - end of support structure, in m 
Dpb = 6.80                   # Separation primary - baffles/support structure, in m

FOV = 8.2                    # Field-of-view of the camera, in deg     
F = 5.5863                   # Focal length of the full telescope, in m
Dps  = 3/2. * F              # Separation primary-secondary mirror (pole to pole), yields 8.38 m, the factor 3/2 is accurate by design
# -- alpha * Fp = 3./2 * F = 3./2 * Fp / (1+eta)  -- 2 * alpha = 3 /(1+eta) or 1+eta = 3/(2 alpha)
Dscam = 1./3 * F             # Separation secondary-focal plane, yields 1.86 m, the factor 1/3 is accurate by design
# Dscam = (1-alpha)/(1+eta) * Fp = (1-alpha) * F --  (1-alpha) = 1/3 -- alpha = 2/3 and (1+eta) = 9/4
# 
alpha = 2./3                 # Ratio of Dps and Fp, see Fig. 2 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527
demag = 9/4.                 # De-magnification of the secondary mirror (= 1+eta), see Eq. 7 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527, 9/4 is accurate by design
Fp = F * demag               # Focal length of M1, in m, see Eq. 7 of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527
Fs = (1-alpha)*Fp/(demag-1)  # Focal length M2, in m, see Eq. (13) of https://www.sciencedirect.com/science/article/abs/pii/S0927650507000527

# LST
#R1 = 23/2
#R2 = 0
#D2 = 0
#D2max = 0
#Fp = 1.2*R1*2

'''
