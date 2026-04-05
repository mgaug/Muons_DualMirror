import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.special import ellipe
import matplotlib.pyplot as plt

from rayleigh import Rayleigh
from mumodel_helper_v import ev2nm, nm2ev, D

isplot = False

class AtmosphereHelper:
    """
    Class-based version of atmosphere_helper_v.py with configurable:

      - obs_h
      - Hgamma
      - scale_h
      - atm_file

      - R1: outer radius of the primary mirror 
      - Robst: radius of central hole in primary mirror, or radius of approximated roundish camera

    The instance stores the atmosphere table and all derived quantities.
    """

    def __init__(
        self,
        obs_h: float = 2200.0,       # default altitude of telescope asl.
        Hgamma: float | None = None, # default maximum altitude of photon emission from muon
        scale_h: float = 9700.0,     # default scale-height of density profile
        atm_file: str = "data/atm_trans_2147_1_10_0_0_2147.dat", # default corsika atm transmission file
        R1: float = 23./2,           # primary mirror radius 
        Robst: float = 1.5           # hole or camera shadow radius approximated
    ):
        self.obs_h = float(obs_h)
        self.scale_h = float(scale_h)
        self.atm_file = atm_file

        self.R1 = R1
        self.Robst = Robst

        if Hgamma is None:
            # preserve old default logic: LST-like default max emission height
            self.Hgamma = 10000.0 - self.obs_h
        else:
            self.Hgamma = float(Hgamma)

        self.atm_tab = pd.read_table(
            self.atm_file,
            sep=r"\s+",
            skip_blank_lines=True,
            comment="#",
        )

    # ----------------------------
    # convenience constructors
    # ----------------------------

    @classmethod
    def from_lst(cls, obs_h: float = 2200.0, scale_h: float = 9700.0,
                 atm_file: str = "data/atm_trans_2147_1_10_0_0_2147.dat"):
        return cls(obs_h=obs_h, Hgamma=10000.0 - obs_h, scale_h=scale_h, atm_file=atm_file,
                   R1=23/2, Robst=1.5)

    @classmethod
    def from_mst(cls, obs_h: float = 2200.0, scale_h: float = 9700.0,
                 atm_file: str = "data/atm_trans_2147_1_10_0_0_2147.dat"):
        return cls(obs_h=obs_h, Hgamma=8000.0 - obs_h, scale_h=scale_h, atm_file=atm_file,
                   R1=12.3/2, Rhole=1.5)

    @classmethod
    def from_sst(cls, obs_h: float = 2200.0, scale_h: float = 9700.0,
                 atm_file: str = "data/atm_trans_2147_1_10_0_0_2147.dat"):
        return cls(obs_h=obs_h, Hgamma=6000.0 - obs_h, scale_h=scale_h, atm_file=atm_file,
                   R1=2.03, Rhole=0.9)

    # ----------------------------
    # update helpers
    # ----------------------------

    def set_obs_h(self, obs_h: float) -> None:
        self.obs_h = float(obs_h)

    def set_Hgamma(self, Hgamma: float) -> None:
        self.Hgamma = float(Hgamma)

    def set_scale_h(self, scale_h: float) -> None:
        self.scale_h = float(scale_h)

    def set_atm_file(self, atm_file: str) -> None:
        self.atm_file = atm_file
        self.atm_tab = pd.read_table(
            self.atm_file,
            sep=r"\s+",
            skip_blank_lines=True,
            comment="#",
        )

    def set_R1(self, R1: float) -> None:
        self.R1 = R1

    def set_Robst(self, Robst: float) -> None:
        self.Robst = Robst
        
    # ----------------------------
    # summary / printing
    # ----------------------------

    def summary(self) -> dict:
        return {
            "obs_h_m": self.obs_h,
            "Hgamma_m": self.Hgamma,
            "scale_h_m": self.scale_h,
            "atm_file": self.atm_file,
            "atm_columns": list(self.atm_tab.columns),
            "atm_nrows": int(len(self.atm_tab)),
            "available_height_columns": [
                c for c in self.atm_tab.columns if c not in ("wl",)
            ],
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("AtmosphereHelper summary")
        print("-" * 40)
        print(f"obs_h        = {s['obs_h_m']:.3f} m")
        print(f"Hgamma       = {s['Hgamma_m']:.3f} m")
        print(f"scale_h      = {s['scale_h_m']:.3f} m")
        print(f"atm_file     = {s['atm_file']}")
        print(f"atm_nrows    = {s['atm_nrows']}")
        print(f"atm_columns  = {s['atm_columns']}")
        print(f"height cols  = {s['available_height_columns']}")

    def __repr__(self) -> str:
        return (
            f"AtmosphereHelper(obs_h={self.obs_h}, "
            f"Hgamma={self.Hgamma}, "
            f"scale_h={self.scale_h}, "
            f"atm_file='{self.atm_file}')"
        )        

    # ----------------------------
    # cached Rayleigh helpers
    # ----------------------------

    def rayleigh_alpha0_from_energy(self, e):
        """
        alpha_0(E) for molecular attenuation.
        e may be scalar or ndarray in eV.
        """
        e = np.asarray(e, dtype=float)
        wl = ev2nm(e)
        alpha = np.array([Rayleigh(w).alpha for w in np.ravel(wl)], dtype=float)
        alpha = alpha.reshape(wl.shape)
        return alpha * 1e-3 * np.exp(-self.obs_h / self.scale_h)

    # ----------------------------
    # scalar path transmission, vectorized in energy/path
    # ----------------------------

    def transmission_mol(self, e, rmax, costheta):
        """
        Molecular transmission for energies e and path length rmax.

        e: photon energy in eV 
        rmax: largest distance of photon emission from muon 
        costheta: cos of observation zenith angle 

        An exponentially falling molecular density model has been assumed, 

        VOD = int_0^infty  alpha_mol(h) dh 

        with: 

        alpha_mol(h) = alpha_0 * exp(-h/scale_h) 

        and hence: 

        VOD = alpha_0 * scale_h

        For a stratified atmosphere, 
        
        alpha_mol(r) = alpha_mol(h) / cos(theta) 

        can be assumed and therefore: 

        OD(theta) = VOD / cos(theta)
        """
        e = np.asarray(e, dtype=float)
        rmax = np.asarray(rmax, dtype=float)
        alpha0 = self.rayleigh_alpha0_from_energy(e)
        tau = (alpha0 / costheta) * self.scale_h * (
            1.0 - np.exp(-rmax * costheta / self.scale_h)
        )
        return np.exp(-tau)

    @staticmethod
    def alpha0_from_vaod(vaod,Haer=None,HPBL=None,HElterman=None,debug=False):
        """
        If Haer is provided and HPBL is None, an exponentially falling aerosol 
        density model is assumed: 

        VAOD = int_0^infty  alpha_aer (h) dh 

        with: 

        alpha_aer(h) = alpha_0 * exp(-h/Haer) 

        and hence: 

        VAOD = alpha_0 * Haer


        If HPBL is provided and Haer is None, a turbulent well-mixed aerosol layer
        with constant density until HPBL is assumed and: 

        VAOD = alpha_0 * HPBL

        If all Haer, HPBL and HElterman are provided, then a La-Palma like aerosol 
        layer following the clear-night aerosol 
        model of https://academic.oup.com/mnras/article/515/3/4520/6608884,
        Section 6 is assumed: 

        VAOD = int_0^H_PBL α0 * exp( -H_PBL / H_aer ) * exp( -(h - H_PBL) / H_Elterman ) dh
             + int_{H_PBL}^{∞} α0 * exp( -h / H_aer ) dh
        
        with the solution: 

        VAOD = α0 * exp(-H_PBL / H_aer) * ( H_Elterman * (exp(H_PBL / H_Elterman) - 1) + H_aer )

        """

        if HPBL is not None and Haer is None:
            return vaod / HPBL
        if Haer is not None and HPBL is None:
            return vaod / Haer
        if HPBL is not None and Haer is not None and HElterman is not None:
            return vaod * np.exp(HPBL/Haer) / (HElterman*(np.exp(HPBL/HElterman)-1.0) + Haer)
        
    @staticmethod
    def tau_from_alpha0(alpha0,Haer=None,HPBL=None,HElterman=None,debug=False):
        """
        If Haer is provided and HPBL is None, an exponentially falling aerosol 
        density model is assumed: 

        VAOD = int_0^infty  alpha_aer (h) dh 

        with: 

        alpha_aer(h) = alpha_0 * exp(-h/Haer) 

        and hence: 

        VAOD = alpha_0 * Haer


        If HPBL is provided and Haer is None, a turbulent well-mixed aerosol layer
        with constant density until HPBL is assumed and: 

        VAOD = alpha_0 * HPBL

        If all Haer, HPBL and HElterman are provided, then a La-Palma like aerosol 
        layer following the clear-night aerosol 
        model of https://academic.oup.com/mnras/article/515/3/4520/6608884,
        Section 6 is assumed: 

        VAOD = int_0^H_PBL α0 * exp( -H_PBL / H_aer ) * exp( -(h - H_PBL) / H_Elterman ) dh
             + int_{H_PBL}^{∞} α0 * exp( -h / H_aer ) dh
        
        with the solution: 

        VAOD = α0 * exp(-H_PBL / H_aer) * ( H_Elterman * (exp(H_PBL / H_Elterman) - 1) + H_aer )

        """

        if HPBL is None and Haer is not None:
            return vaod / HPBL
        if Haer is None and HPBL is not None:
            return vaod / Haer
        if HPBL is not None and Haer is not None and HElterman is not None:
            return vaod * np.exp(HPBL/Haer) / (HElterman*(np.exp(HPBL/HElterman)-1.0) + Haer)
        
        

    def transmission_aer(self, e, rmax, costheta, vaod, AE,
                         Haer=None, HPBL=None, HElterman=None,
                         debug=False):
        """
        Aerosol transmission for energies e and path length rmax.

        e: photon energy in eV 
        rmax: largest distance of photon emission from muon 
        costheta: cos of observation zenith angle 

        vaod: assumed ground-layer Vertical Aerorosl Optical Depth  
        HPBL: assumed nocturnal PBL height
        Helterman: assumed ground-layer scale height before HPBL (in m)
        Haer: assumed ground-layer scale height after HPBL (in m)
        AE: assumed Angstrom exponent of aerosol extinction

        If Haer is provided and HPBL is None, an exponentially falling aerosol 
        density model is assumed: 

        VAOD = int_0^infty  alpha_aer (h) dh 

        with: 

        alpha_aer(h) = alpha_0 * exp(-h/Haer) 

        and hence: 

        VAOD = alpha_0 * Haer

        If HPBL is provided and Haer is None, a turbulent well-mixed aerosol layer
        with constant density until HPBL is assumed and: 

        VAOD = alpha_0 * HPBL

        If all Haer, HPBL and HElterman are provided, then a La-Palma like aerosol 
        layer following the clear-night aerosol 
        model of https://academic.oup.com/mnras/article/515/3/4520/6608884,
        Section 6 is assumed: 

        VAOD = int_0^H_PBL α0 * exp( -H_PBL / H_aer ) * exp( -(h - H_PBL) / H_Elterman ) dh
             + int_{H_PBL}^{∞} α0 * exp( -h / H_aer ) dh
        
        with the solution: 

        VAOD = α0 * exp(-H_PBL / H_aer) * ( H_Elterman * (exp(H_PBL / H_Elterman) - 1) + H_aer )

        For a stratified atmosphere (which is not really the case for the ORM), 
        
        alpha_aer(r) = alpha_aer(h) / cos(theta) 

        can be assumed and therefore: 

        AOD(theta) = VAOD / cos(theta)
        """
        e = np.asarray(e, dtype=float)
        rmax = np.asarray(rmax, dtype=float)

        if Haer is None and HPBL is None and HElterman is None:
            raise ValueError("At least one of HPBL, Haer and HElterman must be provided")

        if (Haer is not None and Haer < 0) or (HPBL is not None and HPBL < 0) or (HElterman is not None and HElterman < 0):
            raise ValueError("HPBL, Haer and HElterman must be larger than zero, instead got: ",HPBL,Haer,HElterman)            
        alpha0 =self.alpha0_from_vaod(vaod,Haer,HPBL,HElterman) * np.power(e / nm2ev(532.0), AE)

        if HPBL is not None and Haer is None:
            tau = np.where(rmax*costheta < HPBL, rmax*alpha0, 0.)
        elif HPBL is None and Haer is not None:
            tau = (alpha0 / costheta) * Haer * (
                1.0 - np.exp(-rmax * costheta / Haer)
            )
        else:
            tau = np.where(rmax*costheta < HPBL,
                           (alpha0 / costheta) * HElterman * np.exp(-HPBL/Haer+HPBL/HElterman) * (
                               1.0 - np.exp(-rmax * costheta / HElterman)
                           ),
                           (alpha0 / costheta) * HElterman * np.exp(-HPBL/Haer) * (
                               np.exp(HPBL / HElterman) - 1.0
                           ) + (alpha0 / costheta) * Haer * (
                               np.exp(-HPBL/Haer) - np.exp(-rmax * costheta / Haer)
                           )
            )
        if debug:
            print ('rmax*costheta: ', rmax*costheta,
                   'HPBL: ', HPBL,
                   'HElterman: ', HElterman,
                   'Haer: ', Haer,
                   'tau: ', tau)
            
        return np.exp(-tau)

    # ------------------
    # average along path
    # ------------------

    def av_transmission_mol(self, e, rmax, costheta, n_path=256, rmin=None, debug=False):
        """
        Average molecular transmission over x in [0, rmax].

        e: photon energy in eV 
        rmax: largest distance of photon emission from muon 
        costheta: cos of observation zenith angle 

        n_path: numbers of segments for numerical integration

        An exponentially falling molecular density model has been assumed, 

        VOD = int_0^infty  alpha_mol(h) dh 

        with: 

        alpha_mol(h) = alpha_0 * exp(-h/scale_h) 

        and hence: 

        VOD = alpha_0 * scale_h

        For a stratified atmosphere, 
        
        alpha_mol(r) = alpha_mol(h) / cos(theta) 

        can be assumed and therefore: 

        OD(theta) = VOD / cos(theta)
        """
        e = np.asarray(e, dtype=float)
        rmax = np.asarray(rmax, dtype=float)

        if np.any(rmax < 0):
            raise ValueError("rmax must be >= 0", rmax)
        if rmin is not None and np.any(rmin < 0):
            raise ValueError("rmin must be >= 0", rmin)
        if rmin is not None and np.any(rmax-rmin < 0):
            raise ValueError("rmax must be >= rmin", rmax-rmin)

        u = np.linspace(0.0, 1.0, n_path)
        rmax = np.asarray(rmax, dtype=float)

        shape_u = (n_path,) + (1,) * rmax.ndim
        if rmin is not None:
            # define array of emission points on the imaged path
            x = rmin[None, ...] + u.reshape(shape_u) * (rmax[None, ...] - rmin[None, ...])
        else:
            x = u.reshape(shape_u) * rmax[None, ...]
        #x = np.outer(u, np.atleast_1d(rmax)) # compute the outer product
        if debug:
            print ('x before subtraction: ',x)

        alpha0 = np.atleast_1d(self.rayleigh_alpha0_from_energy(e)) 
        alpha0 = alpha0.reshape((len(alpha0),) + (1,) * x.ndim)        
        xx = x[None, ...]

        # integral of exponentially falling alpha_mol(r)
        # light travels from xx to 0 ! 
        # tau = (alpha0 / costheta) * self.scale_h * (
        #    1.0 - np.exp(-xx * costheta / self.scale_h)
        # )

        # 
        tau = (alpha0 / costheta) * self.scale_h * (
            1.0 
            - np.exp(-xx * costheta / self.scale_h)
        )
        if debug:
            print ('tau: ',tau)
        T = np.exp(-tau)

        integ = trapezoid(T, x=x[None, ...], axis=1)
        if rmin is not None:
            denom = np.maximum((rmax-rmin)[None, ...], 1e-30)
        else:
            denom = np.maximum(rmax[None, ...], 1e-30)

        out = integ / denom

        if np.ndim(rmax) == 0:
            out = out[:, 0]
        return out

    def av_transmission_aer(self, e, rmax, costheta, vaod, AE,
                            Haer=None, HPBL=None, HElterman=None,
                            n_path=256, rmin=None, debug=False):
        """
        Weighted average aerosol transmission over x in [0, rmax].

        e: photon energy in eV 
        rmax: largest distance of photon emission from muon 
        costheta: cos of observation zenith angle 

        vaod: assumed ground-layer Vertical Aerorosl Optical Depth  
        Haer: assumed ground-layer scale height (in m)
        AE: assumed Angstrom exponent of aerosol extinction

        n_path: numbers of segments for numerical integration

        If Haer is provided and HPBL is None, an exponentially falling aerosol 
        density model is assumed: 

        VAOD = int_0^infty  alpha_aer (h) dh 

        with: 

        alpha_aer(h) = alpha_0 * exp(-h/Haer) 

        and hence: 

        VAOD = alpha_0 * Haer

        If HPBL is provided and Haer is None, a turbulent well-mixed aerosol layer
        with constant density until HPBL is assumed and: 

        VAOD = alpha_0 * HPBL

        If all Haer, HPBL and HElterman are provided, then a La-Palma like aerosol 
        layer following the clear-night aerosol 
        model of https://academic.oup.com/mnras/article/515/3/4520/6608884,
        Section 6 is assumed: 

        VAOD = int_0^H_PBL α0 * exp( -H_PBL / H_aer ) * exp( -(h - H_PBL) / H_Elterman ) dh
             + int_{H_PBL}^{∞} α0 * exp( -h / H_aer ) dh
        
        with the solution: 

        VAOD = α0 * exp(-H_PBL / H_aer) * ( H_Elterman * (exp(H_PBL / H_Elterman) - 1) + H_aer )

        For a stratified atmosphere (which is not really the case for the ORM), 
        
        alpha_aer(r) = alpha_aer(h) / cos(theta) 

        can be assumed and therefore: 

        AOD(theta) = VAOD / cos(theta)

        """
        e = np.asarray(e, dtype=float)
        rmax = np.asarray(rmax, dtype=float)

        if np.any(rmax < 0):
            raise ValueError("rmax must be >= 0", rmax)
        if rmin is not None and np.any(rmin < 0):
            raise ValueError("rmin must be >= 0", rmin)
        if rmin is not None and np.any(rmax-rmin < 0):
            raise ValueError("rmax must be >= rmin", rmax-rmin)

        if Haer is None and HPBL is None and HElterman is None:
            raise ValueError("At least one of HPBL, Haer and HElterman must be provided")
        if Haer < 0 or HPBL < 0 or HElterman < 0:
            raise ValueError("HPBL, Haer and HElterman must be larger than zero, instead got: ",
                             HPBL,Haer,HElterman)            

        u = np.linspace(0.0, 1.0, n_path)
        rmax = np.asarray(rmax, dtype=float)

        shape_u = (n_path,) + (1,) * rmax.ndim
        if rmin is not None:
            # define array of emission points on the imaged path
            x = rmin[None, ...] + u.reshape(shape_u) * (rmax[None, ...] - rmin[None, ...])
        else:
            x = u.reshape(shape_u) * rmax[None, ...]

        if debug:
            print ('x before subtraction: ',x)

        alpha0 = self.alpha0_from_vaod(vaod,Haer,HPBL,HElterman) * np.power(np.atleast_1d(e) / nm2ev(532.0), AE)
        alpha0 = alpha0.reshape((len(alpha0),) + (1,) * x.ndim)        

        xx = x[None, ...]
        
        if HPBL is not None and Haer is None:
            tau = np.where(xx *costheta < HPBL, rmax*alpha0, 0.)
        elif HPBL is None and Haer is not None:
            tau = (alpha0 / costheta) * Haer * (
                1.0 - np.exp(-xx * costheta / Haer)
            )
        else:
            # the La Palma aerosol model
            tau = np.where(rmax*costheta < HPBL,
                           (alpha0 / costheta) * HElterman * np.exp(-HPBL/Haer+HPBL/HElterman) * (
                               1.0 - np.exp(-xx * costheta / HElterman)
                           ),
                           (alpha0 / costheta) * HElterman * np.exp(-HPBL/Haer) * (
                               np.exp(HPBL / HElterman) - 1.0
                           ) + (alpha0 / costheta) * Haer * (
                               np.exp(-HPBL/Haer) - np.exp(-xx * costheta / Haer)
                           )
            )

        if debug:
            print ('tau: ',tau)
            
        T = np.exp(-tau)

        integ = trapezoid(T, x=x[None, ...], axis=1)
        if rmin is not None:
            denom = np.maximum((rmax-rmin)[None, ...], 1e-30)
        else:
            denom = np.maximum(rmax[None, ...], 1e-30)
        #denom = np.maximum(np.atleast_1d(rmax)[None, :], 1e-30)
        out = integ / denom

        if np.ndim(rmax) == 0:
            out = out[:, 0]
        return out

    # ----------------------------
    # phi-averaged transmission
    # ----------------------------

    @staticmethod
    def rmax_for_phi(R1, Robst, thetac, rhoR, phi, debug=False):
        """
        Geometric maximum path length used in the original phi-averaged functions.
        """
        rhoR = np.asarray(rhoR, dtype=float)
        phi = np.asarray(phi, dtype=float)

        if rhoR.ndim == 0:
            rhoR_grid = rhoR
            phi_grid = phi
        else:
            rhoR_grid = rhoR[:, None]
            phi_grid = phi[None, :]
            
        Lmax = R1 / np.tan(thetac) * D(rhoR_grid, phi_grid)
        Lmin = Robst / np.tan(thetac) * D(rhoR_grid * R1 / Robst, phi_grid)
        if debug:
            print("rhoR shape:", np.shape(rhoR))
            print("phi shape:", np.shape(phi))
            print("Lmax shape:", np.shape(Lmax))
            print("Lmin shape:", np.shape(Lmin))
            print ('Lmax: ', Lmax, ' Lmin: ', Lmin)
        return Lmax, Lmin

    def av_transmission_rho_mol(self, e, thetac, costheta,
                                rho_min=0., rho_max=1., 
                                n_rho=128, n_phi=512, n_path=256, debug=False):
        """
        weighted av_transmission_mol over rhoR values starting from
        rho_min to rho_max and using the av_transmission_phi_mol
        to obtain a phi-averaged value for av_transmission_mol

        e: photon energy in eV 
        thetac: Cherenkov angle in radians 
        costheta: cos of observation zenith angle 
        rho_min: minimum rhoR value considered
        rho_max: maximum rhoR value considered
        n_rho, n_phi, n_path: numbers of segments for numerical integration

        """
        e = np.atleast_1d(np.asarray(e, dtype=float))

        if rho_min < 0 or rho_max < 0 or rho_max < rho_min:
            raise ValueError("rho_min, rho_max must be larger than zero and rho_max>rho_min, instead got: ",rho_min, rho_max)                        
        
        rhoR = np.linspace(rho_min, rho_max, n_rho)
        T = self.av_transmission_phi_mol(e, thetac, rhoR, costheta,
                                         n_phi=n_phi, n_path=n_path, debug=debug)
        if debug:
            print ('T: ',T)

        # weight each transmission with rhoR to account for high probability of
        # occurrence when rhoR lies further outwards
        norm = 0.5 * (rho_max**2 - rho_min**2)
        out = trapezoid(T * rhoR, x=rhoR, axis=-1) / norm
        return out

    def av_transmission_phi_mol(self, e, thetac, rhoR, costheta,
                                n_phi=512, n_path=256, debug=False):
        """
        weighted av_transmission_mol over circle from 0 to 2pi, 
        using the rmax_for_phi to obtain a phi-dependent rmax value 
        for av_transmission_mol

        e: photon energy in eV 
        thetac: Cherenkov angle in radians 
        rhoR: normalized muon impact distance from mirror center
        costheta: cos of observation zenith angle 
        n_phi, n_path: numbers of segments for numerical integration

        """
        e = np.atleast_1d(np.asarray(e, dtype=float))

        rhoR = np.asarray(rhoR, dtype=float)        
        phi = np.linspace(0.0, np.pi, n_phi)
        rmax, rmin = self.rmax_for_phi(self.R1, self.Robst, thetac, rhoR, phi, debug)

        av = self.av_transmission_mol(e, rmax, costheta, n_path=n_path, rmin=rmin, debug=debug)

        if debug:
            print("rhoR shape:", rhoR.shape)
            print("phi shape:", phi.shape)
            print("rmax shape:", np.shape(rmax))
            print("av shape:", np.shape(av))

        if debug:
            print ('rmax: ',rmax)
            print ('av: ',av)

        if rhoR.ndim == 0:
            if rhoR >= 1:
                maxphi = np.arcsin(1./rhoR)
            else:
                maxphi = np.pi
        else:
            maxphi = np.where(rhoR >= 1., np.arcsin(1./rhoR), np.pi)
            
        out = (1. / maxphi) * trapezoid(av, x=phi, axis=-1)
        return out

    def av_transmission_phi_aer(self, e, thetac, rhoR, costheta,
                                vaod, AE, Haer=None, HPBL=None, HElterman=None,
                                n_phi=512, n_path=256, debug=False):
        """
        weighted av_transmission_aer over circle from 0 to 2pi, 
        using the rmax_from_phi to obtain a phi-dependent rmax value 
        for av_transmission_aer

        e: photon energy in eV 
        thetac: Cherenkov angle in radians 
        rhoR: normalized muon impact distance from mirror center
        costheta: cos of observation zenith angle 

        vaod: assumed ground-layer Vertical Aerorosl Optical Depth  
        HPBL: assumed nocturnal PBL height
        Helterman: assumed ground-layer scale height before HPBL (in m)
        Haer: assumed ground-layer scale height after HPBL (in m)
        AE: assumed Angstrom exponent of aerosol extinction

        n_phi, n_path: numbers of segments for numerical integration

        If Haer is provided and HPBL is None, an exponentially falling aerosol 
        density model is assumed: 

        VAOD = int_0^infty  alpha_aer (h) dh 

        with: 

        alpha_aer(h) = alpha_0 * exp(-h/Haer) 

        and hence: 

        VAOD = alpha_0 * Haer

        If HPBL is provided and Haer is None, a turbulent well-mixed aerosol layer
        with constant density until HPBL is assumed and: 

        VAOD = alpha_0 * HPBL

        If all Haer, HPBL and HElterman are provided, then a La-Palma like aerosol 
        layer following the clear-night aerosol 
        model of https://academic.oup.com/mnras/article/515/3/4520/6608884,
        Section 6 is assumed: 

        VAOD = int_0^H_PBL α0 * exp( -H_PBL / H_aer ) * exp( -(h - H_PBL) / H_Elterman ) dh
             + int_{H_PBL}^{∞} α0 * exp( -h / H_aer ) dh
        
        with the solution: 

        VAOD = α0 * exp(-H_PBL / H_aer) * ( H_Elterman * (exp(H_PBL / H_Elterman) - 1) + H_aer )

        """
        e = np.atleast_1d(np.asarray(e, dtype=float))
        rhoR = np.asarray(rhoR, dtype=float)
        
        phi = np.linspace(0.0, 2*np.pi, n_phi)
        rmax,rmin = self.rmax_for_phi(self.R1, self.Robst, thetac, rhoR, phi)

        av = self.av_transmission_aer(e, rmax, costheta, vaod, AE,
                                      Haer=Haer, HPBL=HPBL, HElterman=HElterman,
                                      n_path=n_path,rmin=rmin, debug=debug)
        if rhoR.ndim == 0:
            if rhoR >= 1:
                maxphi = np.arcsin(1./rhoR)
            else:
                maxphi = np.pi
        else:
            maxphi = np.where(rhoR >= 1., np.arcsin(1./rhoR), np.pi)
            
        out = (1. / maxphi) * trapezoid(av, x=phi, axis=-1)
        return out

    # ----------------------------
    # transmission from atmosphere table
    # ----------------------------

    def get_trans_from_trans_file(
        self,
        Hgamma=None,
        aod_wrong=0.1,
        H_wrong=1300,
        AA_wrong=2.5,
        aod_corr=0.03,
        H_corr=600,
        AA_corr=1.2,
    ):
        """
        Vectorized version of get_trans_from_trans_file(Hgamma).
        If Hgamma is None, uses self.Hgamma.
        """
        if Hgamma is None:
            Hgamma = self.Hgamma

        es_corr = nm2ev(np.asarray(self.atm_tab["wl"], dtype=float))

        tgamma_aer_corr_wrong = self.av_transmission_aer(
            es_corr, Hgamma, 1.0, aod_wrong, H_wrong, AA_wrong
        )
        tgamma_aer_corr_right = self.av_transmission_aer(
            es_corr, Hgamma, 1.0, aod_corr, H_corr, AA_corr
        )

        col = f"{int((Hgamma + self.obs_h) / 1000.0)}.000"
        ods = (
            np.asarray(self.atm_tab[col], dtype=float)
            - np.asarray(self.atm_tab["2.197"], dtype=float)
            + np.log(tgamma_aer_corr_wrong)
            - np.log(tgamma_aer_corr_right)
        )
        return np.exp(-ods)

    @staticmethod
    def get_e0(rho):
        rho = np.asarray(rho, dtype=float)
        return ellipe(rho * rho) * 2.0 / np.pi

    @staticmethod
    def add_wl_axis(ax):
            ax_c = ax.twiny()
            wls  = np.array([700,600,500,450,400,350,300,250,200])
            #wls  = np.array([800,700,600,500,400,300,250,200])
            wlticks = [nm2ev(wl) for wl in wls]
            ax_c.set_xticks(wlticks);
            ax_c.set_xticklabels(['{:g}'.format(wl) for wl in wls]);
            ax_c.set_xlabel('Wavelength (nm)')
    
            xmin, xmax = 1.75, 5.25
            ax.set_xlim(xmin, xmax)
            ax_c.set_xlim(xmin, xmax);
    
    # ----------------------------
    # plotting helpers
    # ----------------------------

    @staticmethod
    def _finalize_plot(ax, xlabel, ylabel, title=None, legend=True, grid=True):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if grid:
            ax.grid(True, alpha=0.3)
        if legend:
            ax.legend()

    @staticmethod
    def _save_or_show(fig, filename=None, show=True):
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_transmission_mol(
        self,
        energies_ev,
        rmax,
        costheta,
        filename=None,
        show=True,
        ax=None,
        label=None,
    ):
        energies_ev = np.asarray(energies_ev, dtype=float)
        y = self.transmission_mol(energies_ev, rmax, costheta)

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(energies_ev, y, label=label or f"rmax={rmax:.0f}")            
        self._finalize_plot(
            ax,
            xlabel="Photon energy (eV)",
            ylabel="Transmission",
            title="Molecular transmission",
            legend=True,
        )

        if created or show:
            self._save_or_show(ax.get_figure(), filename=filename, show=show)
        return ax

    def plot_transmission_aer(
        self,
        energies_ev,
        rmax,
        costheta,
        vaod,
        AE,
        Haer=None,
        HPBL=None,
        HElterman=None,
        filename=None,
        show=True,
        ax=None,
        label=None,
        debug=False
    ):
        energies_ev = np.asarray(energies_ev, dtype=float)
        y = self.transmission_aer(energies_ev, rmax, costheta, vaod, AE, Haer, HPBL, HElterman, debug)

        if debug:
            print ('y =',y)        
        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(energies_ev, y, label=label or f"rmax={rmax:.0f}")
        self._finalize_plot(
            ax,
            xlabel="Photon energy (eV)",
            ylabel="Transmission",
            title=f"Aerosol transmission, vaod={vaod}, Haer={Haer}, HPBL={HPBL}, HElterman={HElterman}, AE={AE}",
            legend=True,
        )

        if created or show:
            self._save_or_show(ax.get_figure(), filename=filename, show=show)
        return ax

    def plot_av_transmission_mol(
        self,
        energies_ev,
        rmax,
        costheta,
        n_path=256,
        rmin=None,
        filename=None,
        show=True,
        ax=None,
        label=None,
        debug=False
    ):
        energies_ev = np.asarray(energies_ev, dtype=float)
        y = self.av_transmission_mol(energies_ev, rmax, costheta, n_path=n_path, rmin=rmin, debug=debug)

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(energies_ev, y, label=label or f"avg mol, rmax={rmax}")
        self._finalize_plot(
            ax,
            xlabel="Photon energy (eV)",
            ylabel="Average transmission",
            title="Average molecular transmission",
            legend=True,
        )

        if created:
            self._save_or_show(fig, filename=filename, show=show)
        return ax

    def plot_av_transmission_aer(
        self,
        energies_ev,
        rmax,
        costheta,
        aod,
        H,
        AA,
        n_path=256,
        filename=None,
        show=True,
        ax=None,
        label=None,
    ):
        energies_ev = np.asarray(energies_ev, dtype=float)
        y = self.av_transmission_aer(
            energies_ev, rmax, costheta, aod, H, AA, n_path=n_path
        )

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(energies_ev, y, label=label or f"avg aer, aod={aod}, H={H}, AA={AA}")
        self._finalize_plot(
            ax,
            xlabel="Photon energy (eV)",
            ylabel="Average transmission",
            title="Average aerosol transmission",
            legend=True,
        )

        if created:
            self._save_or_show(fig, filename=filename, show=show)
        return ax

    def plot_av_transmission_phi_mol(
        self,
        energies_ev,
        thetac,
        rhoR,
        costheta,
        n_phi=512,
        n_path=256,
        filename=None,
        show=True,
        ax=None,
        label=None,
    ):
        energies_ev = np.asarray(energies_ev, dtype=float)
        y = self.av_transmission_phi_mol(
            energies_ev, thetac, rhoR, costheta, n_phi=n_phi, n_path=n_path
        )

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(energies_ev, y, label=label or "phi-avg mol")
        self._finalize_plot(
            ax,
            xlabel="Photon energy (eV)",
            ylabel="Phi-averaged transmission",
            title="Phi-averaged molecular transmission",
            legend=True,
        )

        if created:
            self._save_or_show(fig, filename=filename, show=show)
        return ax

    def plot_av_transmission_rho_mol(
        self,
        energies_ev,
        thetac,
        costheta,
        rhoR_min,
        rhoR_max,
        n_rho=128,
        n_phi=512,
        n_path=256,
        filename=None,
        show=True,
        ax=None,
        label=None,
        debug=False,
        **kwargs
    ):
        energies_ev = np.asarray(energies_ev, dtype=float)
        y = self.av_transmission_rho_mol(
            energies_ev, thetac, costheta,
            rho_min=rhoR_min, rho_max=rhoR_max, 
            n_rho=n_rho, n_phi=n_phi, n_path=n_path, debug=debug
        )

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(energies_ev, y, label=label or "rho-avg mol", **kwargs)
        self._finalize_plot(
            ax,
            xlabel="Photon energy (eV)",
            ylabel="rhoR-averaged transmission",
            title="rho-averaged molecular transmission",
            legend=True,
        )

        if created:
            self._save_or_show(fig, filename=filename, show=show)
        return ax

    
    
    def plot_av_transmission_phi_aer(
        self,
        energies_ev,
        thetac,
        rhoR,
        costheta,
        aod,
        H,
        AA,
        n_phi=512,
        n_path=256,
        filename=None,
        show=True,
        ax=None,
        label=None,
    ):
        energies_ev = np.asarray(energies_ev, dtype=float)
        y = self.av_transmission_phi_aer(
            energies_ev, thetac, rhoR, costheta,
            aod, H, AA, n_phi=n_phi, n_path=n_path
        )

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(energies_ev, y, label=label or "phi-avg aer")
        self._finalize_plot(
            ax,
            xlabel="Photon energy (eV)",
            ylabel="Phi-averaged transmission",
            title="Phi-averaged aerosol transmission",
            legend=True,
        )

        if created:
            self._save_or_show(fig, filename=filename, show=show)
        return ax

    def plot_trans_from_trans_file(
        self,
        Hgamma=None,
        filename=None,
        show=True,
        ax=None,
        label=None,
        **kwargs,
    ):
        wl = np.asarray(self.atm_tab["wl"], dtype=float)
        trans = self.get_trans_from_trans_file(Hgamma=Hgamma, **kwargs)

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(wl, trans, label=label or f"table-corrected, Hgamma={Hgamma or self.Hgamma:.0f} m")
        self._finalize_plot(
            ax,
            xlabel="Wavelength (nm)",
            ylabel="Transmission",
            title="Transmission from atmosphere table",
            legend=True,
        )

        if created:
            self._save_or_show(fig, filename=filename, show=show)
        return ax

    def plot_all_core_transmissions(
        self,
        energies_ev,
        rmax,
        costheta,
        aod,
        H,
        AA,
        filename=None,
        show=True,
    ):
        energies_ev = np.asarray(energies_ev, dtype=float)
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(
            energies_ev,
            self.transmission_mol(energies_ev, rmax, costheta),
            label=r'$T_{mol}$'+f' from {rmax:.0f}m distance',            
        )
        ax.plot(
            energies_ev,
            self.transmission_aer(energies_ev, rmax, costheta, aod, H, AA),
            label=r'$T_{aer}$'+f' from {rmax:.0f}m distance',
        )
        ax.plot(
            energies_ev,
            self.av_transmission_mol(energies_ev, rmax, costheta),
            label=r'av. $T_{mol}$ along path',
        )
        ax.plot(
            energies_ev,
            self.av_transmission_aer(energies_ev, rmax, costheta, aod, H, AA),
            label=r'av. $T_{aer}$ along path',
        )

        self._finalize_plot(
            ax,
            xlabel="Photon energy (eV)",
            ylabel="Transmission",
            title=f'T from {rmax:.0f}m, ZA: {np.rad2deg(np.arccos(costheta)):.0f} deg, AOD={aod:.2f}, AE={AA:.1f},'+r' $H_{aer}$='+f'{H:.0f}m',
            legend=True,
        )
        self._save_or_show(fig, filename=filename, show=show)
        return ax


    # ----------------------------
    # plot averaged phi transmissions vs rhoR / costheta
    # ----------------------------

    def plot_av_phi_mol_vs_rhoR(
        self,
        energies,
        thetac,
        rhoR_values,
        costheta,
        n_phi=512,
        n_path=256,
        filename=None,
        show=True,
        ax=None,
        cmap="viridis",
        debug=False
    ):
        """
        Plot av_transmission_phi_mol as a function of photon energy
        for different rhoR values at fixed costheta.
        """
        energies = np.asarray(energies, dtype=float)
        rhoR_values = np.asarray(rhoR_values, dtype=float)

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(rhoR_values)))

        for rhoR, color in zip(rhoR_values, colors):
            y = self.av_transmission_phi_mol(
                energies, thetac, rhoR, costheta,
                n_phi=n_phi, n_path=n_path, debug=debug
            )
            if debug:
                print ('rhoR=',rhoR,' y=',y)
            ax.plot(energies, y, color=color, label=fr"$\rho_R={rhoR:.3f}$")

        self._finalize_plot(
            ax,
            xlabel="Photon energy (eV)",
            ylabel="Phi-averaged molecular transmission",
            title=fr"$\langle t_{{\mu,\mathrm{{mol}}}}\rangle_\phi$ vs energy, fixed $\cos\theta={costheta:.3f}$",
            legend=True,
        )

        self.add_wl_axis(ax)
        
        if created:
            self._save_or_show(fig, filename=filename, show=show)

        return ax

    def plot_av_phi_aer_vs_rhoR(
        self,
        energies_ev,
        thetac,
        rhoR_values,
        costheta,
        vaod,
        AE,
        Haer=None,
        HPBL=None,
        HElterman=None,
        n_phi=512,
        n_path=256,
        filename=None,
        show=True,
        ax=None,
        cmap="plasma",
        debug=False
    ):
        """
        Plot av_transmission_phi_aer as a function of photon energy
        for different rhoR values at fixed costheta.
        """
        energies_ev = np.asarray(energies_ev, dtype=float)
        rhoR_values = np.asarray(rhoR_values, dtype=float)

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(rhoR_values)))

        for rhoR, color in zip(rhoR_values, colors):
            y = self.av_transmission_phi_aer(
                energies_ev, thetac, rhoR, costheta,
                vaod, AE, Haer, HPBL, HElterman,
                n_phi=n_phi, n_path=n_path, debug=debug
            )
            if debug:
                print ('rhoR=',rhoR,' y=',y)
            ax.plot(energies_ev, y, color=color, label=fr"$\rho_R={rhoR:.3f}$")

        self._finalize_plot(
            ax,
            xlabel="Photon energy (eV)",
            ylabel="Phi-averaged aerosol transmission",
            title=fr"$\langle t_{{\mu,\mathrm{{aer}}}}\rangle_\phi$ vs energy, fixed $\cos\theta={costheta:.3f}$",
            legend=True,
        )

        self.add_wl_axis(ax)

        if created:
            self._save_or_show(fig, filename=filename, show=show)
        return ax

    def plot_av_phi_mol_vs_costheta(
        self,
        energies_ev,
        thetac,
        rhoR,
        costheta_values,
        n_phi=256,
        n_path=256,
        filename=None,
        show=True,
        ax=None,
        cmap="viridis",
    ):
        """
        Plot av_transmission_phi_mol as a function of photon energy
        for different costheta values at fixed rhoR.
        """
        energies_ev = np.asarray(energies_ev, dtype=float)
        costheta_values = np.asarray(costheta_values, dtype=float)

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(costheta_values)))

        for ct, color in zip(costheta_values, colors):
            y = self.av_transmission_phi_mol(
                energies_ev, thetac, rhoR, ct,
                n_phi=n_phi, n_path=n_path
            )
            ax.plot(energies_ev, y, color=color, label=fr"$\cos\theta={ct:.3f}$")

        self._finalize_plot(
            ax,
            xlabel="Photon energy (eV)",
            ylabel="Phi-averaged molecular transmission",
            title=fr"$\langle t_{{\mu,\mathrm{{mol}}}}\rangle_\phi$ vs energy, fixed $\rho_R={rhoR:.3f}$",
            legend=True,
        )

        self.add_wl_axis(ax)
        
        if created:
            self._save_or_show(fig, filename=filename, show=show)
        return ax

    def plot_av_phi_aer_vs_costheta(
        self,
        energies_ev,
        thetac,
        rhoR,
        costheta_values,
        aod,
        Haer,
        AA,
        n_phi=256,
        n_path=256,
        filename=None,
        show=True,
        ax=None,
        cmap="plasma",
    ):
        """
        Plot av_transmission_phi_aer as a function of photon energy
        for different costheta values at fixed rhoR.
        """
        energies_ev = np.asarray(energies_ev, dtype=float)
        costheta_values = np.asarray(costheta_values, dtype=float)

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(costheta_values)))

        for ct, color in zip(costheta_values, colors):
            y = self.av_transmission_phi_aer(
                energies_ev, thetac, rhoR, ct,
                aod, Haer, AA, n_phi=n_phi, n_path=n_path
            )
            ax.plot(energies_ev, y, color=color, label=fr"$\cos\theta={ct:.3f}$")

        self._finalize_plot(
            ax,
            xlabel="Photon energy (eV)",
            ylabel="Phi-averaged aerosol transmission",
            title=fr"$\langle t_{{\mu,\mathrm{{aer}}}}\rangle_\phi$ vs energy, fixed $\rho_R={rhoR:.3f}$",
            legend=True,
        )

        self.add_wl_axis(ax)
        
        if created:
            self._save_or_show(fig, filename=filename, show=show)
        return ax

    
# plots and tests
if isplot:

    atm = AtmosphereHelper.from_lst()
    atm.print_summary()
    
    energies = np.linspace(1.5, 5.0, 200)
    
    atm.plot_transmission_mol(energies, rmax=5000.0, costheta=1.0)
    atm.plot_transmission_aer(energies, rmax=5000.0, costheta=1.0, aod=0.03, H=600, AA=1.2)
    atm.plot_av_transmission_mol(energies, rmax=5000.0, costheta=1.0)
    atm.plot_av_transmission_aer(energies, rmax=5000.0, costheta=1.0, aod=0.03, H=600, AA=1.2)
    
    atm.plot_trans_from_trans_file()

    atm.plot_av_transmission_phi_mol(
        energies_ev=energies,
        R=11.8,
        r=1.5,
        thetac=np.deg2rad(1.23),
        rhoR=0.32,
        costheta=1.0,
    )

    atm.plot_av_transmission_phi_aer(
        energies_ev=energies,
        R=11.8,
        r=1.5,
        thetac=np.deg2rad(1.23),
        rhoR=0.32,
        costheta=1.0,
        aod=0.03,
        H=600,
        AA=1.2,
    )
