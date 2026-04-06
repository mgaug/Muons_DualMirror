import numpy as np
from dataclasses import dataclass
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from bandwidth_helper_v import BandwidthHelper
from atmosphere_helper_v import AtmosphereHelper 
import telescope as tel


@dataclass
class UncertaintyConfig:
    sigma_vaod: float = 0.01
    sigma_Haer: float = 100.0
    sigma_AE: float = 0.15
    sigma_scale_h: float = 300
    sigma_theta_c_deg: float = 0.02
    sigma_rhoR_min: float = 0.02
    sigma_HPBL: float = 100.0
    sigma_HElterman: float = 200.0

    rel_mirror_unc: float = 0.01
    rel_window_unc: float = 0.01

    n_mc: int = 200
    random_seed: int | None = 12345


class MuonModel:
    """
    High-level muon Cherenkov light model built from:
      - AtmosphereHelper
      - Telescope
      - bandwidth_helper detector curves

    It computes detector-weighted muon and gamma transmission bandwidths.
    """

    def __init__(
        self,
        telescope_obj=None,           # optional Telescope class
        telescope_name: str | None = "LST",  #  Telescope name 
        atmosphere: AtmosphereHelper | None = None, # optional AtmospherHelper class
        bandwidth: BandwidthHelper | None = None,            
        obs_height: float = 2200.0,   # Telescope altitude asl.
        Hgamma: float | None = None,  # Average gamma-ray emission height 
        scale_h: float = 9700.0,      # Average molecular density scale height 
        atm_file: str = "data/atm_trans_2147_1_10_0_0_2147.dat",  # default atmospheric extinction file 
        theta_tel_deg: float = 0.0,   # Telescope observing zenith angle 
        theta_c_deg: float = 1.,      # muon Cherenkov angle in deg.
        rhoR_min: float = 0.1,        # lower muon impact distance cut (in relative rhoR = rho/R1)
        vaod: float = 0.03,           # default vertical aerosol optical depth at 532 nm 
        Haer: float = 500.0,          # default aerosol scale height above HPBL
        AE: float = 1.45,             # default Angstrom exponent of ground-layer aerosols 
        HPBL: float = 800.0,          # default boundary layer height 
        HElterman: float = 1200.0,    # default aerosol scale height below HBPL 
    ):
        # telescope
        if telescope_obj is not None:
            self.tel = telescope_obj
        else:
            self.tel = self._make_telescope(telescope_name)

        self.telescope_name = self.tel.name

        # atmosphere
        if atmosphere is not None:
            self.atm = atmosphere
        else:
            if Hgamma is None:
                Hgamma = self._default_Hgamma(self.telescope_name, obs_height)
            self.atm = AtmosphereHelper(
                obs_height=obs_height,
                Hgamma=Hgamma,
                atm_file=atm_file,
                R1=tel.R1, Robst=tel.Rhole
            )
            
        self.bh = bandwidth if bandwidth is not None else BandwidthHelper()
            
        # observing / model parameters
        self.theta_tel_deg = float(theta_tel_deg)
        self.theta_c_deg = float(theta_c_deg)
        self.thetac = np.deg2rad(theta_c_deg)
        self.costheta = np.cos(np.deg2rad(theta_tel_deg))
        self.rhoR_min = float(rhoR_min)

        self.scale_h = float(scale_h)
        
        self.vaod = float(vaod)
        self.Haer = float(Haer)
        self.AE = float(AE)
        self.HPBL = float(HPBL)
        self.HElterman = float(HElterman)

        self._setup_detector()
        self._setup_gamma_transmission()

    @classmethod
    def from_LSTN(cls, bandwidth: BandwidthHelper | None = None,
                  theta_tel_deg=0., theta_c_deg=1., rhoR_min=0.1, scale_h=9700., 
                  atm_file: str = "data/atm_trans_2147_1_10_0_0_2147.dat"):
        
        tel_object = tel.LST()
        atm = AtmosphereHelper.from_lst()
        bh = bandwidth if bandwidth is not None else BandwidthHelper()        
        costheta = np.cos(np.deg2rad(theta_tel_deg))

        # AE from non-dusty periods day-time,
        # see Fig. 3 of https://www.aanda.org/articles/aa/full_html/2023/05/aa45787-22/F3.html
        
        # La-Palma like aerosol layer following the clear-night aerosol model 
        # of https://academic.oup.com/mnras/article/515/3/4520/6608884, Section 6 

        return cls(telescope_obj=tel_object, atmosphere=atm, bandwidth=bh, 
                   scale_h=scale_h,theta_tel_deg=theta_tel_deg, theta_c_deg=theta_c_deg,rhoR_min=rhoR_min,
                   vaod=0.03, AE=1.45, Haer=500., HPBL=800. * costheta ** 0.77,HElterman=1200,
                   )

    @classmethod
    def from_MSTN(cls, bandwidth: BandwidthHelper | None = None,
                  theta_tel_deg=0., theta_c_deg=1., rhoR_min=0.1, scale_h=9700., 
                  atm_file: str = "data/atm_trans_2147_1_10_0_0_2147.dat"):
        
        tel_object = tel.MST()
        atm = AtmosphereHelper.from_mst()
        bh = bandwidth if bandwidth is not None else BandwidthHelper()                
        costheta = np.cos(np.deg2rad(theta_tel_deg))
        
        # AE from non-dusty periods day-time,
        # see Fig. 3 of https://www.aanda.org/articles/aa/full_html/2023/05/aa45787-22/F3.html

        # La-Palma like aerosol layer following the clear-night aerosol model 
        # of https://academic.oup.com/mnras/article/515/3/4520/6608884, Section 6 
        
        return cls(telescope_obj=tel_object, atmosphere=atm, bandwidth=bh, 
                   scale_h=scale_h,theta_tel_deg=theta_tel_deg, theta_c_deg=theta_c_deg,rhoR_min=rhoR_min,
                   vaod=0.03, AE=1.45, Haer=500., HPBL=800. * costheta ** 0.77,HElterman=1200,
                   )

    @classmethod
    def from_LSTS(cls, bandwidth: BandwidthHelper | None = None,
                  theta_tel_deg=0., theta_c_deg=1., rhoR_min=0.1, scale_h=9700., 
                  atm_file: str = "data/atm_trans_2147_1_10_0_0_2147.dat"):
        
        tel_object = tel.LST()
        atm = AtmosphereHelper.from_lst()
        bh = bandwidth if bandwidth is not None else BandwidthHelper()                        
        
        # extremely shallow nocturnal turbulent surface layer PBL at Paranal, see
        # https://academic.oup.com/mnras/article/492/1/934/5674124, afterwards exponential decay

        return cls(telescope_obj=tel_object, atmosphere=atm, bandwidth=bh,
                   scale_h=scale_h,theta_tel_deg=theta_tel_deg, theta_c_deg=theta_c_deg,rhoR_min=rhoR_min,
                   vaod=0.03, AE=1.45, Haer=500., HPBL=100.,HElterman=9000,
                   )

    @classmethod
    def from_MSTS(cls, bandwidth: BandwidthHelper | None = None,
                  theta_tel_deg=0., theta_c_deg=1., rhoR_min=0.1, scale_h=9700., 
                  atm_file: str = "data/atm_trans_2147_1_10_0_0_2147.dat"):
        
        tel_object = tel.MST()
        atm = AtmosphereHelper.from_mst()
        bh = bandwidth if bandwidth is not None else BandwidthHelper()                                
        
        # extremely shallow nocturnal turbulent surface layer PBL at Paranal, see
        # https://academic.oup.com/mnras/article/492/1/934/5674124, afterwards exponential decay

        return cls(telescope_obj=tel_object, atmosphere=atm, bandwidth=bh,
                   scale_h=scale_h,theta_tel_deg=theta_tel_deg, theta_c_deg=theta_c_deg,rhoR_min=rhoR_min,
                   vaod=0.03, AE=1.45, Haer=500., HPBL=100.,HElterman=9000,
                   )

    @classmethod
    def from_SSTS(cls, bandwidth: BandwidthHelper | None = None,
                  theta_tel_deg=0., theta_c_deg=1., rhoR_min=0.1, scale_h=9700., 
                  atm_file: str = "data/atm_trans_2147_1_10_0_0_2147.dat"):
        
        tel_object = tel.SST()
        atm = AtmosphereHelper.from_sst()
        bh = bandwidth if bandwidth is not None else BandwidthHelper()                                        
        
        # extremely shallow nocturnal turbulent surface layer PBL at Paranal, see
        # https://academic.oup.com/mnras/article/492/1/934/5674124, afterwards exponential decay

        return cls(telescope_obj=tel_object, atmosphere=atm, bandwidth=bh, 
                   scale_h=scale_h,theta_tel_deg=theta_tel_deg, theta_c_deg=theta_c_deg,rhoR_min=rhoR_min,
                   vaod=0.03, AE=1.45, Haer=500., HPBL=100.,HElterman=9000,
                   )
    
    @staticmethod
    def _make_telescope(name: str):
        name = name.upper()
        if name == "LST":
            return tel.LST()
        if name == "MST":
            return tel.MST()
        if name == "SST":
            return tel.SST()
        if name == "SCT":
            return tel.SCT()
        raise ValueError(f"Unknown telescope name: {name}")

    @staticmethod
    def _default_Hgamma(name: str, obs_h: float) -> float:
        name = name.upper()
        if name == "LST":
            # Median Hgamma for zenith angle of 0 deg. is about 11 km, in the energy range below 1 TeV
            # see Fig. 1 of https://www.mdpi.com/2072-4292/17/6/1074
            return 11000.0 - obs_h
        if name in ("MST","SCT"):
            # Median Hgamma for zenith angle of 0 deg. is about 10 km, in the energy range from 500 GeV to 10 TeV
            # see Fig. 1 of https://www.mdpi.com/2072-4292/17/6/1074
            return 10000.0 - obs_h
        if name == "SST":
            # Median Hgamma for zenith angle of 0 deg. is about 9 km, in the energy range above 1 TeV
            # see Fig. 1 of https://www.mdpi.com/2072-4292/17/6/1074       
            return 9000.0 - obs_h
        return 10000.0 - obs_h

    @staticmethod
    def _get_Robst(tel: tel.Telescope) -> float:
        name = name.upper()
        if name == "LST":
            return tel.Acam
        if name == "MST":
            return tel.Acam
        if name in ("SST", "SCT"):
            return tel.R2
        return None
        
    def _setup_detector(self):
        """
        Select detector response from bandwidth_helper according to telescope type.
        Single-mirror LST/MST -> PMT
        Dual-mirror SST/SCT   -> SiPM
        """
        if self.telescope_name in ("LST", "MST"):
            self.detector_type = "PMT"
            self.energy = np.asarray(self.bh.xi_e_pmt, dtype=float)
            self.wavelength = np.asarray(self.bh.xi_wl_pmt, dtype=float)

            self.qe_nominal = np.asarray(self.bh.qe_int(self.energy), dtype=float)
            self.qe_sigma = np.asarray(self.bh.detector_efficiency_uncertainty("PMT", self.energy), dtype=float)

            self.mirror_nominal = np.asarray(self.bh.mi_int(self.energy), dtype=float)
            self.window_nominal = np.asarray(self.bh.ca_int(self.energy), dtype=float)

        else:
            self.detector_type = "SIPM"
            self.energy = np.asarray(self.bh.xi_e_sipm, dtype=float)
            self.wavelength = np.asarray(self.bh.xi_wl_sipm, dtype=float)

            self.qe_nominal = np.asarray(self.bh.si_int(self.energy), dtype=float)
            self.qe_sigma = np.asarray(self.bh.detector_efficiency_uncertainty("SIPM", self.energy), dtype=float)

            self.mirror_nominal = np.asarray(self.bh.mi_int(self.energy), dtype=float)
            self.window_nominal = np.asarray(self.bh.cs_int(self.energy), dtype=float)

        self.energy_step = float(self.bh.xi_steps)
        self.det_eff_nominal = self.detector_efficiency()

    def _setup_gamma_transmission(self):
        """
        Interpolate corrected gamma transmission from atmosphere table
        onto the detector wavelength grid.
        """
        trans = self.atm.get_trans_from_trans_file()
        wl_tab = np.asarray(self.atm.atm_tab["wl"], dtype=float)

        self.gamma_trans_wl_interp = interp1d(
            wl_tab,
            trans,
            kind="linear",
            bounds_error=False,
            fill_value=(trans[0], trans[-1]),
        )

        self.gamma_transmission_nominal = np.asarray(
            self.gamma_trans_wl_interp(self.wavelength),
            dtype=float,
        )

    def detector_efficiency(self, qe=None, mirror=None, window=None):
        qe = self.qe_nominal if qe is None else np.asarray(qe, dtype=float)
        mirror = self.mirror_nominal if mirror is None else np.asarray(mirror, dtype=float)
        window = self.window_nominal if window is None else np.asarray(window, dtype=float)
        return np.clip(qe * mirror * window, 0.0, 1.0)
        
    def gamma_transmission(self):
        return self.gamma_transmission_nominal.copy()

    def muon_transmission_mol(self, rhoR_min=None, costheta=None, thetac=None, scale_h=None, **kwargs):
        rhoR_min = self.rhoR_min if rhoR_min is None else rhoR_min
        costheta = self.costheta if costheta is None else costheta
        thetac = self.thetac if thetac is None else thetac
        scale_h = self.scale_h if scale_h is None else scale_h

        return self.atm.av_transmission_rho_mol(
            self.energy,
            thetac=thetac,
            costheta=costheta,
            scale_h=scale_h,
            rho_min=rhoR_min,
            rho_max=1.,
            **kwargs,
        )

    def muon_transmission_aer(
        self,
        rhoR_min=None,
        costheta=None,
        thetac=None,
        vaod=None,
        Haer=None,
        AE=None,
        HPBL=None,
        HElterman=None,
        **kwargs,
    ):
        rhoR_min = self.rhoR_min if rhoR_min is None else rhoR_min
        costheta = self.costheta if costheta is None else costheta
        thetac = self.thetac if thetac is None else thetac

        vaod = self.vaod if vaod is None else vaod
        Haer = self.Haer if Haer is None else Haer
        AE = self.AE if AE is None else AE
        HPBL = self.HPBL if HPBL is None else HPBL
        HElterman = self.HElterman if HElterman is None else HElterman

        return self.atm.av_transmission_rho_aer(
            self.energy,
            thetac=thetac,
            costheta=costheta,
            vaod=vaod,
            AE=AE,
            rho_min=rhoR_min,
            rho_max=1.0,
            Haer=Haer,
            HPBL=HPBL,
            HElterman=HElterman,
            **kwargs,
        )

    def muon_transmission(self, **kwargs):
        tmol = self.muon_transmission_mol(**kwargs)
        taer = self.muon_transmission_aer(**kwargs)
        return tmol * taer

    def gamma_response(self):
        return self.det_eff_nominal * self.gamma_transmission()

    def muon_response(self, **kwargs):
        return self.det_eff_nominal * self.muon_transmission(**kwargs)

    def bandwidth_gamma(self):
        return float(np.sum(self.gamma_response()) * self.energy_step)

    def bandwidth_muon(self, **kwargs):
        return float(np.sum(self.muon_response(**kwargs)) * self.energy_step)

    def ratio_gamma_to_muon(self, **kwargs):
        return self.bandwidth_gamma() / self.bandwidth_muon(**kwargs)

    def cumulative_blindness_curve(self, **kwargs):
        gamma_comb = self.gamma_response()
        muon_comb = self.muon_response(**kwargs)

        c_gamma = np.cumsum(gamma_comb) * self.energy_step
        c_muon = np.cumsum(muon_comb) * self.energy_step
        total_gamma = c_gamma[-1]
        total_muon = c_muon[-1]

        loss = (c_gamma - c_muon * total_gamma / total_muon) / np.maximum(c_gamma, 1e-30)
        return self.energy.copy(), loss

    def summary(self):
        return {
            "telescope": self.telescope_name,
            "detector_type": self.detector_type,
            "R1_m": self.tel.R1,
            "Rhole_m": self.tel.Rhole,
            "theta_tel_deg": self.theta_tel_deg,
            "theta_c_deg": self.theta_c_deg,
            "rhoR_min": self.rhoR_min,
            "costheta": self.costheta,
            "obs_h_m": self.atm.obs_h,
            "Hgamma_m": self.atm.Hgamma,
            "scale_h_m": self.scale_h,
            "atm_file": self.atm.atm_file,
            "vaod": self.vaod,
            "Haer_m": self.Haer,
            "AE": self.AE,
            "HPBL_m": self.HPBL,
            "HElterman_m": self.HElterman,
            "n_energy": len(self.energy),
        }

    def print_summary(self):
        s = self.summary()
        print("MuonModel summary")
        print("-" * 50)
        for k, v in s.items():
            print(f"{k:16s}: {v}")

    def propagate_uncertainty(self, cfg: UncertaintyConfig | None = None, **kwargs):
        if cfg is None:
            cfg = UncertaintyConfig()

        rng = np.random.default_rng(cfg.random_seed)

        bmu = np.empty(cfg.n_mc, dtype=float)
        bgam = np.empty(cfg.n_mc, dtype=float)
        ratio = np.empty(cfg.n_mc, dtype=float)

        for i in range(cfg.n_mc):
            scale_h_i = max(100.0, rng.normal(self.scale_h, cfg.sigma_scale_h))            
            vaod_i = max(1e-1, rng.normal(self.vaod, cfg.sigma_vaod))
            Haer_i = max(1000., rng.normal(self.Haer, cfg.sigma_Haer))
            AE_i = rng.normal(self.AE, cfg.sigma_AE)
            theta_c_i = max(0.5, rng.normal(self.theta_c_deg, cfg.sigma_theta_c_deg))
            rhoR_min_i = max(0.0, rng.normal(self.rhoR_min, cfg.sigma_rhoR_min))
            HPBL_i = max(30.0, rng.normal(self.HPBL, cfg.sigma_HPBL))
            HElterman_i = max(100.0, rng.normal(self.HElterman, cfg.sigma_HElterman))

            if self.detector_type == "PMT":
                qe_i = self.qe_nominal + rng.normal(0.0, self.qe_sigma, size=self.qe_nominal.shape)
            else:
                qe_i = self.qe_nominal.copy()

            mirror_i = self.mirror_nominal * (
                1.0 + rng.normal(0.0, cfg.rel_mirror_unc, size=self.mirror_nominal.shape)
            )
            window_i = self.window_nominal * (
                1.0 + rng.normal(0.0, cfg.rel_window_unc, size=self.window_nominal.shape)
            )

            det_i = self.detector_efficiency(qe=qe_i, mirror=mirror_i, window=window_i)

            mu_i = det_i * (
                self.atm.av_transmission_rho_mol(
                    self.energy,
                    thetac=np.deg2rad(theta_c_i),
                    costheta=self.costheta,
                    rhoR_min=rhoR_min_i,
                    scale_h=scale_h_i,
                    **kwargs,
                )
                *
                self.atm.av_transmission_rho_aer(
                    self.energy,
                    thetac=np.deg2rad(theta_c_i),
                    costheta=self.costheta,
                    rhoR_min=rhoR_min_i,
                    vaod=vaod_i,
                    Haer=Haer_i,
                    AE=AE_i,
                    HPBL=HPBL_i,
                    HElterman=HElterman_i,
                    **kwargs,
                )
            )

            ga_i = det_i * self.gamma_transmission()

            bmu[i] = np.sum(mu_i) * self.energy_step
            bgam[i] = np.sum(ga_i) * self.energy_step
            ratio[i] = bgam[i] / max(bmu[i], 1e-30)

        def summary(x):
            return {
                "mean": float(np.mean(x)),
                "std": float(np.std(x, ddof=1)),
                "median": float(np.median(x)),
                "p16": float(np.percentile(x, 16)),
                "p84": float(np.percentile(x, 84)),
            }

        return {
            "B_mu": summary(bmu),
            "B_gamma": summary(bgam),
            "ratio_gamma_to_muon": summary(ratio),
            "samples": {
                "B_mu": bmu,
                "B_gamma": bgam,
                "ratio_gamma_to_muon": ratio,
            },
        }


    @staticmethod
    def build_standard_models(theta_tel_deg=0., theta_c_deg=1., rhoR_min=0.1, scale_h=9700.):
        """
        Convenience constructor for the standard comparison set.
        """
        return {
            "LSTN": MuonModel.from_LSTN(theta_tel_deg=theta_tel_deg, theta_c_deg=theta_c_deg, rhoR_min=rhoR_min, scale_h=scale_h),
            "LSTS": MuonModel.from_LSTS(theta_tel_deg=theta_tel_deg, theta_c_deg=theta_c_deg, rhoR_min=rhoR_min, scale_h=scale_h),
            "MSTN": MuonModel.from_MSTN(theta_tel_deg=theta_tel_deg, theta_c_deg=theta_c_deg, rhoR_min=rhoR_min, scale_h=scale_h),
            "MSTS": MuonModel.from_MSTS(theta_tel_deg=theta_tel_deg, theta_c_deg=theta_c_deg, rhoR_min=rhoR_min, scale_h=scale_h),
            "SSTS": MuonModel.from_SSTS(theta_tel_deg=theta_tel_deg, theta_c_deg=theta_c_deg, rhoR_min=rhoR_min, scale_h=scale_h),
        }
    

    # ============================================================
    # plotting helpers
    # ============================================================

    @staticmethod
    def _save_show(fig, filename=None, show=True):
        
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")

        if show:
            plt.show()
        #if show:
        #    plt.show()
        #else:
        #    plt.close(fig)


    @staticmethod
    def _energy_to_wavelength_twin(ax, wavelengths_nm, xmin=None, xmax=None, xlabel="Photon wavelength (nm)"):
        ax_c = ax.twiny()
        wlticks = [self.bh.nm2ev(wl) for wl in wavelengths_nm]
        ax_c.set_xticks(wlticks)
        ax_c.set_xticklabels([f"{wl:g}" for wl in wavelengths_nm])
        ax_c.set_xlabel(xlabel)
        if xmin is not None and xmax is not None:
            ax_c.set_xlim(xmin, xmax)
        return ax_c

    # ============================================================
    # 1) detector contributions plot
    # ============================================================

    def plot_contributions(self, filename=None, show=True, ax=None):
        """
        Equivalent of old plot_contributions block.
        Detector-only plot, independent of telescope instance.
        """
        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(self.bh.qe_e, self.bh.qe_ham, "m--", lw=2, label="Photomultiplier QE")
        ax.plot(self.bh.si_e, self.bh.qe_si,  "c-.", lw=2, label="SiPM PDE")
        ax.plot(self.bh.mi_e, self.bh.mi_ref, "b-", label="Mirror Reflectivity")
        ax.plot(self.bh.ca_e, self.bh.ca_ref, "r-", lw=2, label="Protection Window Transparency")

        ax.legend(bbox_to_anchor=(0.99, 0.98), loc="upper right")
        ax.set_xlabel(r"Photon energy $\epsilon$ (eV)")
        ax.set_ylabel(r"efficiency $\xi$")
        xmin, xmax = 1.2, 6.2
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.05, 1.05)

        self._energy_to_wavelength_twin(ax, [900, 800, 700, 600, 500, 400, 300, 250, 200], xmin, xmax)

        self._save_show(ax.get_figure(), filename=filename, show=show)
        return ax

    # ============================================================
    # internal spectrum helpers
    # ============================================================

    def _muon_atm_only(self):
        return self.muon_transmission()

    def _gamma_atm_only(self):
        return self.gamma_transmission()

    def _muon_combined(self):
        return self.det_eff_nominal * self._muon_atm_only()

    def _gamma_combined(self):
        return self.det_eff_nominal * self._gamma_atm_only()

    def _energy_grid(self, with_camera=True):
        if self.detector_type == "PMT":
            return self.bh.xi_e_pmt if with_camera else self.bh.xi_e_pmt_nocam
        return self.bh.xi_e_sipm if with_camera else self.bh.xi_e_sipm_nocam

    def _wavelength_grid(self, with_camera=True):
        if self.detector_type == "PMT":
            return self.bh.xi_wl_pmt if with_camera else self.bh.xi_wl_pmt_nocam
        return self.bh.xi_wl_sipm if with_camera else self.bh.xi_wl_sipm_nocam

    def _detector_grid(self, with_camera=True):
        if self.detector_type == "PMT":
            return self.bh.xi_det_pmt if with_camera else self.bh.xi_det_pmt_nocam
        return self.bh.xi_det_sipm if with_camera else self.bh.xi_det_sipm_nocam

    def _gamma_interp_on_grid(self, wavelength_grid):
        trans = self.atm.get_trans_from_trans_file()
        wl_tab = np.asarray(self.atm.atm_tab["wl"], dtype=float)
        interp = interp1d(wl_tab, trans, kind="linear", bounds_error=False,
                          fill_value=(trans[0], trans[-1]))
        return interp(wavelength_grid)

    def _muon_atm_on_grid(self, energy_grid):
        """
        Evaluate muon atm transmission on an arbitrary detector energy grid.
        """
        old_energy = self.energy
        try:
            self.energy = np.asarray(energy_grid, dtype=float)
            return self.muon_transmission()
        finally:
            self.energy = old_energy

    # ============================================================
    # 2) xi_det * transmission comparison (old plot_xidet)
    # ============================================================

    @classmethod
    def plot_xidet_comparison(cls, models: dict | None = None, filename=None, show=True, ax=None):
        """
        Comparison plot for LST and SST, equivalent to old plot_xidet.
        """
        if models is None:
            models = cls.build_standard_models()

        lstn = models["LSTN"]
        ssts = models["SSTS"]

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        tmulst_comb = lstn._muon_combined()
        tgammalst_comb = lstn._gamma_combined()

        tmusst_comb = ssts._muon_combined()
        tgammasst_comb = ssts._gamma_combined()

        bh = next(iter(models.values())).bh

        ax.plot(
            self.bh.xi_e_pmt, tmulst_comb, "-", color="b",
            label=rf"$\xi_{{det}}\cdot t_{{\mu}}$ LST, B$_\mu$: {tmulst_comb.sum()*self.bh.xi_steps:.3f} eV"
        )
        ax.plot(
            self.bh.xi_e_pmt, tgammalst_comb, "--", color="cornflowerblue",
            label=rf"$\xi_{{det}}\cdot t_{{\gamma}}$ LST, B$_\gamma$: {tgammalst_comb.sum()*self.bh.xi_steps:.3f} eV"
        )
        ax.plot(
            self.bh.xi_e_sipm, tmusst_comb, "-", color="r",
            label=rf"$\xi_{{det}}\cdot t_{{\mu}}$ SST, B$_\mu$: {tmusst_comb.sum()*self.bh.xi_steps:.3f} eV"
        )
        ax.plot(
            self.bh.xi_e_sipm, tgammasst_comb, "--", color="darkorange",
            label=rf"$\xi_{{det}}\cdot t_{{\gamma}}$ SST, B$_\gamma$: {tgammasst_comb.sum()*self.bh.xi_steps:.3f} eV"
        )

        ax.legend(bbox_to_anchor=(1.0, 1.0), loc=1)
        ax.set_xlabel(r"Photon energy $\epsilon$ (eV)")
        ax.set_ylabel(r"efficiency $\xi$")
        xmin, xmax = 1.4, 5.0
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0.0, 0.48)

        cls._energy_to_wavelength_twin(ax, [800, 700, 600, 500, 400, 300, 250], xmin, xmax)
        cls._save_show(ax.get_figure(), filename=filename, show=show)
        return ax

    # ============================================================
    # 3) blindness ratio plot (old plot_ratio)
    # ============================================================

    @classmethod
    def plot_ratio_comparison(cls, models: dict | None = None, filename=None, show=True, ax=None):
        """
        Equivalent of old plot_ratio block.
        """
        if models is None:
            models = cls.build_standard_models()

        lstn = models["LSTN"]
        mstn = models["MSTN"]
        lsts = models["LSTS"]
        msts = models["MSTS"]
        ssts = models["SSTS"]

        created = ax is None
        if created:
            fig, ax = plt.subplots(constrained_layout=True)

        # with camera
        tmulst_atm = lstn._muon_atm_only()
        tgammalst_atm = lstn._gamma_atm_only()
        tmulst_comb = lstn._muon_combined()
        tgammalst_comb = lstn._gamma_combined()

        tmumst_atm = mstn._muon_atm_only()
        tgammamst_atm = mstn._gamma_atm_only()
        tmumst_comb = mstn._muon_combined()
        tgammamst_comb = mstn._gamma_combined()

        tmusst_atm = ssts._muon_atm_only()
        tgammasst_atm = ssts._gamma_atm_only()
        tmusst_comb = ssts._muon_combined()
        tgammasst_comb = ssts._gamma_combined()

        # no-camera grids
        e_pmt_nc = self.bh.xi_e_pmt_nocam
        w_pmt_nc = self.bh.xi_wl_pmt_nocam
        det_pmt_nc = self.bh.xi_det_pmt_nocam

        e_sipm_nc = self.bh.xi_e_sipm_nocam
        w_sipm_nc = self.bh.xi_wl_sipm_nocam
        det_sipm_nc = self.bh.xi_det_sipm_nocam

        tmulst_atm_nc = lstn._muon_atm_on_grid(e_pmt_nc)
        tgammalst_atm_nc = lstn._gamma_interp_on_grid(w_pmt_nc)
        tmulst_comb_nc = det_pmt_nc * tmulst_atm_nc
        tgammalst_comb_nc = det_pmt_nc * tgammalst_atm_nc

        tmumst_atm_nc = mstn._muon_atm_on_grid(e_pmt_nc)
        tgammamst_atm_nc = mstn._gamma_interp_on_grid(w_pmt_nc)
        tmumst_comb_nc = det_pmt_nc * tmumst_atm_nc
        tgammamst_comb_nc = det_pmt_nc * tgammamst_atm_nc

        tmusst_atm_nc = ssts._muon_atm_on_grid(e_sipm_nc)
        tgammasst_atm_nc = ssts._gamma_interp_on_grid(w_sipm_nc)
        tmusst_comb_nc = det_sipm_nc * tmusst_atm_nc
        tgammasst_comb_nc = det_sipm_nc * tgammasst_atm_nc

        min_pmt = self.bh.get_low_idx(self.bh.xi_det_pmt, 2e-2)
        min_sipm = self.bh.get_low_idx(self.bh.xi_det_sipm, 2e-2)
        min_pmt_nc = self.bh.get_low_idx(self.bh.xi_det_pmt_nocam, 2e-2)
        min_sipm_nc = self.bh.get_low_idx(self.bh.xi_det_sipm_nocam, 2e-2)

        max_pmt = self.bh.get_high_idx(self.bh.xi_det_pmt, 2e-2)
        max_sipm = self.bh.get_high_idx(self.bh.xi_det_sipm, 2e-2)
        max_pmt_nc = self.bh.get_high_idx(tgammalst_comb_nc, 5e-4)
        max_sipm_nc = self.bh.get_high_idx(tgammasst_comb_nc, 5e-4)

        tmulst_tot = (np.cumsum(tmulst_atm)[max_pmt] - np.cumsum(tmulst_atm)[min_pmt]) * self.bh.xi_steps
        tmumst_tot = (np.cumsum(tmumst_atm)[max_pmt] - np.cumsum(tmumst_atm)[min_pmt]) * self.bh.xi_steps
        tmusst_tot = (np.cumsum(tmusst_atm)[max_sipm] - np.cumsum(tmusst_atm)[min_sipm]) * self.bh.xi_steps

        tmulst_tot_nc = (np.cumsum(tmulst_atm_nc)[max_pmt_nc] - np.cumsum(tmulst_atm_nc)[min_pmt_nc]) * self.bh.xi_steps
        tmumst_tot_nc = (np.cumsum(tmumst_atm_nc)[max_pmt_nc] - np.cumsum(tmumst_atm_nc)[min_pmt_nc]) * self.bh.xi_steps
        tmusst_tot_nc = (np.cumsum(tmusst_atm_nc)[max_sipm_nc] - np.cumsum(tmusst_atm_nc)[min_sipm_nc]) * self.bh.xi_steps

        tgammalst_tot = (np.cumsum(tgammalst_atm)[max_pmt] - np.cumsum(tgammalst_atm)[min_pmt]) * self.bh.xi_steps
        tgammamst_tot = (np.cumsum(tgammamst_atm)[max_pmt] - np.cumsum(tgammamst_atm)[min_pmt]) * self.bh.xi_steps
        tgammasst_tot = (np.cumsum(tgammasst_atm)[max_sipm] - np.cumsum(tgammasst_atm)[min_sipm]) * self.bh.xi_steps

        tgammalst_tot_nc = (np.cumsum(tgammalst_atm_nc)[max_pmt_nc] - np.cumsum(tgammalst_atm_nc)[min_pmt_nc]) * self.bh.xi_steps
        tgammamst_tot_nc = (np.cumsum(tgammamst_atm_nc)[max_pmt_nc] - np.cumsum(tgammamst_atm_nc)[min_pmt_nc]) * self.bh.xi_steps
        tgammasst_tot_nc = (np.cumsum(tgammasst_atm_nc)[max_sipm_nc] - np.cumsum(tgammasst_atm_nc)[min_sipm_nc]) * self.bh.xi_steps

        ax.plot(
            self.bh.xi_e_pmt,
            (np.cumsum(tgammalst_comb)*self.bh.xi_steps - np.cumsum(tmulst_comb)*self.bh.xi_steps*tgammalst_tot/tmulst_tot)
            / np.cumsum(tgammalst_comb) / self.bh.xi_steps,
            "b-", label="LSTN (with window)", lw=1
        )
        ax.plot(
            self.bh.xi_e_pmt_nocam,
            (np.cumsum(tgammalst_comb_nc)*self.bh.xi_steps - np.cumsum(tmulst_comb_nc)*self.bh.xi_steps*tgammalst_tot_nc/tmulst_tot_nc)
            / np.cumsum(tgammalst_comb_nc) / self.bh.xi_steps,
            "--", color="steelblue", label="LSTN (no window)"
        )
        ax.plot(
            self.bh.xi_e_pmt,
            (np.cumsum(tgammamst_comb)*self.bh.xi_steps - np.cumsum(tmumst_comb)*self.bh.xi_steps*tgammamst_tot/tmumst_tot)
            / np.cumsum(tgammamst_comb) / self.bh.xi_steps,
            "g-", label="MSTN (with window)", lw=1
        )
        ax.plot(
            self.bh.xi_e_pmt_nocam,
            (np.cumsum(tgammamst_comb_nc)*self.bh.xi_steps - np.cumsum(tmumst_comb_nc)*self.bh.xi_steps*tgammamst_tot_nc/tmumst_tot_nc)
            / np.cumsum(tgammamst_comb_nc) / self.bh.xi_steps,
            "--", color="darkseagreen", label="MSTN (no window)"
        )
        ax.plot(
            self.bh.xi_e_sipm,
            (np.cumsum(tgammasst_comb)*self.bh.xi_steps - np.cumsum(tmusst_comb)*self.bh.xi_steps*tgammasst_tot/tmusst_tot)
            / np.cumsum(tgammasst_comb) / self.bh.xi_steps,
            "r-", label="SSTS", lw=1
        )

        ax.legend(bbox_to_anchor=(0.99, 0.99), loc=1)
        ax.set_xlabel(r"Start of sudden detector blindness $\epsilon_{blind}$ (eV)")
        ax.set_ylabel(r"$\Delta B_{\gamma} / B_{\gamma}$")
        xmin, xmax = 2.0, 5.25
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0.0, 0.25)

        cls._energy_to_wavelength_twin(ax, [600, 500, 400, 350, 300, 250, 200], xmin, xmax, xlabel="Wavelength (nm)")
        cls._save_show(ax.get_figure(), filename=filename, show=show)
        return ax

    # ============================================================
    # 4) PDE + transparency summary plot (old plot_pmt)
    # ============================================================

    @classmethod
    def plot_pde_and_transparency(cls, models: dict | None = None, filename=None, show=True, ax=None):
        """
        Equivalent of old plot_pmt block.
        """
        if models is None:
            models = cls.build_standard_models()

        lstn = models["LSTN"]
        ssts = models["SSTS"]

        es = np.arange(1.2, 5.5, 0.1)

        # evaluate on custom energy grid
        tmulst_atm = lstn._muon_atm_on_grid(es)
        tmusst_atm = ssts._muon_atm_on_grid(es)

        trans_lst = lstn._gamma_interp_on_grid(self.bh.ev2nm(es))
        trans_sst = ssts._gamma_interp_on_grid(self.bh.ev2nm(es))

        created = ax is None
        if created:
            fig, ax = plt.subplots(2, 1, constrained_layout=True)
        

        ax[1].plot(self.bh.qe_e, self.bh.qe_ham, "--", color="darkmagenta", label=r"$\xi_{pde}$ (PMT$_1$)")
        ax[1].plot(self.bh.ete_e, self.bh.qe_ete, "--", color="teal", label=r"$\xi_{pde}$ (PMT$_2$)")
        ax[1].plot(self.bh.si_e, self.bh.qe_si, "-", color="green", label=r"$\xi_{pde}$ (SiPM)")
        ax[1].legend(bbox_to_anchor=(0.8, 0.92), loc=2)

        ax[0].plot(es, tmulst_atm, "-", color="b", label=r"$t_{\mu}$ (LST)")
        ax[0].plot(es, trans_lst, "--", color="cornflowerblue", label=r"$t_{\gamma}$ (LST)")
        ax[0].plot(es, tmusst_atm, "-", color="r", label=r"$t_{\mu}$ (SST)")
        ax[0].plot(es, trans_sst, "--", color="darkorange", label=r"$t_{\gamma}$ (SST)")
        ax[0].legend(bbox_to_anchor=(0.8, 0.82), loc=2)

        ax[1].set_xlabel(r"Photon energy $\epsilon$ (eV)")
        ax[0].set_ylabel("atmospheric transparency")
        ax[1].set_ylabel("photon detection efficiency")
        ax[0].set_ylim(0.0, 1.05)
        ax[1].set_ylim(0.0, 0.5)

        xmin, xmax = 1.3, 5.8
        ax[0].set_xlim(xmin, xmax)
        ax[1].set_xlim(xmin, xmax)
        cls._energy_to_wavelength_twin(ax[0], [700, 600, 500, 400, 300, 250], xmin, xmax)

        cls._save_show(ax.get_figure(), filename=filename, show=show)
        return ax


''' old code 
import numpy as np
from dataclasses import dataclass
from math import cos, pi
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

import bandwidth_helper as bh
from atmosphere_helper_v import AtmosphereHelper as ah
from telescope import Telescope as tel
from rayleigh import Rayleigh
from mumodel_helper_v import ev2nm, nm2ev, D


@dataclass
class UncertaintyConfig:
    # detector
    use_detector_qe_uncertainty: bool = True

    # atmosphere / geometry nuisance parameters
    sigma_aod: float = 0.01
    sigma_H: float = 100.0          # m
    sigma_AA: float = 0.15
    sigma_theta_c_deg: float = 0.02
    sigma_rhoR: float = 0.02

    # optional relative uncertainties for optics curves lacking tabulated stdev
    rel_mirror_unc: float = 0.01
    rel_window_unc: float = 0.01

    # MC control
    n_mc: int = 200
    random_seed: int | None = 12345


class VectorizedCherenkovModel:
    """
    Vectorized Cherenkov bandwidth model with uncertainty propagation.

    Main outputs:
      - muon_response()
      - gamma_response()
      - bandwidth_muon()
      - bandwidth_gamma()
      - propagate_uncertainty()
    """

    def __init__(
        self,
        telescope: str = "LST",
        aod: float = 0.03,
        H: float = 600.0,
        AA: float = 1.2,
        theta_tel_deg: float = 0.0,
        theta_c_deg: float = 1.23,
        rhoR: float = 0.32,
        n_path_grid: int = 400,
        n_phi_grid: int = 512,
    ):
        self.telescope = telescope.upper()
        self.aod = float(aod)
        self.H = float(H)
        self.AA = float(AA)
        self.theta_tel_deg = float(theta_tel_deg)
        self.theta_c_deg = float(theta_c_deg)
        self.rhoR = float(rhoR)

        self.costheta = cos(theta_tel_deg * pi / 180.0)
        self.thetac = theta_c_deg * pi / 180.0

        self.n_path_grid = int(n_path_grid)
        self.n_phi_grid = int(n_phi_grid)

        self._set_telescope()
        self._set_detector_grid()
        self._build_interpolators()
        self._cache_rayleigh_alpha()
        self._cache_gamma_transmission()

    def _set_telescope(self) -> None:
        if self.telescope == "LST":
            self.R = R_LST
            self.inner_R = inner_R_LST
            self.Hgamma = Hgamma_LST
            self.detector_type = "PMT"
        elif self.telescope == "MST":
            self.R = R_MST
            self.inner_R = inner_R_MST
            self.Hgamma = Hgamma_MST
            self.detector_type = "PMT"
        elif self.telescope == "SST":
            self.R = R_SST
            self.inner_R = inner_R_SST
            self.Hgamma = Hgamma_SST
            self.detector_type = "SIPM"
        else:
            raise ValueError(f"Unknown telescope: {self.telescope}")

    def _set_detector_grid(self) -> None:
        if self.detector_type == "PMT":
            self.energy = np.asarray(bh.xi_e_pmt, dtype=float)
            self.wavelength = np.asarray(bh.xi_wl_pmt, dtype=float)
            self.det_eff_nominal = np.asarray(bh.xi_det_pmt, dtype=float)
            self.qe_nominal = np.asarray(bh.qe_int(self.energy), dtype=float)
            self.qe_sigma = np.asarray(bh.qe_dev, dtype=float)
            # qe_dev is tabulated on bh.qe_e, so interpolate to model grid
            self.qe_sigma = np.interp(self.energy, bh.qe_e, bh.qe_dev)
            self.mirror_nominal = np.asarray(bh.mi_int(self.energy), dtype=float)
            self.window_nominal = np.asarray(bh.ca_int(self.energy), dtype=float)
        else:
            self.energy = np.asarray(bh.xi_e_sipm, dtype=float)
            self.wavelength = np.asarray(bh.xi_wl_sipm, dtype=float)
            self.det_eff_nominal = np.asarray(bh.xi_det_sipm, dtype=float)
            self.qe_nominal = np.asarray(bh.si_int(self.energy), dtype=float)
            self.qe_sigma = None
            self.mirror_nominal = np.asarray(bh.mi_int(self.energy), dtype=float)
            self.window_nominal = np.asarray(bh.cs_int(self.energy), dtype=float)

        self.energy_step = float(bh.xi_steps)

    def _build_interpolators(self) -> None:
        trans_wl = np.asarray(atm_tab["wl"], dtype=float)
        trans_val = np.asarray(get_trans_from_trans_file(self.Hgamma), dtype=float)

        self.gamma_trans_wl_interp = interp1d(
            trans_wl,
            trans_val,
            kind="linear",
            bounds_error=False,
            fill_value=(trans_val[0], trans_val[-1]),
        )

    def _cache_gamma_transmission(self) -> None:
       trans = get_trans_from_trans_file_vec(self.Hgamma)
        wl = np.asarray(atm_tab["wl"], dtype=float)
        self.gamma_transmission_nominal = np.interp(self.wavelength, wl, trans)

    @staticmethod
    def _safe_clip(x: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        return np.clip(x, low, high)

    def _rmax_phi(self, R: float, inner_R: float, thetac: float, rhoR: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized geometric upper path length in radial coordinate.
        """
        phi = np.linspace(0.0, pi / 2.0, self.n_phi_grid)
        d1 = np.vectorize(D)(rhoR, phi)
        d2 = np.vectorize(D)(rhoR * R / inner_R, phi)
        rmax = R / np.tan(thetac) * d1 - inner_R / np.tan(thetac) * d2
        rmax = np.maximum(rmax, 0.0)
        return phi, rmax

    def _transmission_mol_vectorized(self, rmax_phi: np.ndarray) -> np.ndarray:
        """
        Returns average molecular transmission over phi for each energy.
        Shape returned: (n_energy,)
        """
        # path coordinate grid normalized per phi
        u = np.linspace(0.0, 1.0, self.n_path_grid)[:, None]              # (n_path, 1)
        rmax = rmax_phi[None, :]                                          # (1, n_phi)
        x = u * rmax                                                      # (n_path, n_phi)

        # average transmission from 0..rmax for every phi and energy
        # T_mol(e, x) = exp[-alpha0/ct * integral exp(-s/scale_h) ds]
        #               = exp[-alpha0/ct * scale_h * (1-exp(-x*ct/scale_h))]
        tau = (self.alpha0_mol[:, None, None] / self.costheta) * scale_h * (
            1.0 - np.exp(-x[None, :, :] * self.costheta / scale_h)
        )                                                                 # (n_energy, n_path, n_phi)

        T = np.exp(-tau)
        av_over_r = trapezoid(T, x=x[None, :, :], axis=1) / np.maximum(rmax[None, :], 1e-12)
        # phi average: factor in original helper is 2/pi * ∫_0^{pi/2} ...
        phi = np.linspace(0.0, pi / 2.0, self.n_phi_grid)
        return (2.0 / pi) * trapezoid(av_over_r, x=phi, axis=1)

    def _transmission_aer_vectorized(self, rmax_phi: np.ndarray, aod: float, H: float, AA: float) -> np.ndarray:
        """
        Returns average aerosol transmission over phi for each energy.
        Shape returned: (n_energy,)
        """
        u = np.linspace(0.0, 1.0, self.n_path_grid)[:, None]
        rmax = rmax_phi[None, :]
        x = u * rmax

        if H > 0:
            alpha0 = (aod / H) * np.power(self.energy / nm2ev(532.0), AA)
            H_eff = H
        else:
            alpha0 = (aod / (-H)) * np.power(self.energy / nm2ev(532.0), AA)
            H_eff = 1e10

        tau = (alpha0[:, None, None] / self.costheta) * H_eff * (
            1.0 - np.exp(-x[None, :, :] * self.costheta / H_eff)
        )

        T = np.exp(-tau)
        av_over_r = trapezoid(T, x=x[None, :, :], axis=1) / np.maximum(rmax[None, :], 1e-12)
        phi = np.linspace(0.0, pi / 2.0, self.n_phi_grid)
        return (2.0 / pi) * trapezoid(av_over_r, x=phi, axis=1)

        self.gamma_transmission_nominal = np.asarray(
            self.gamma_trans_wl_interp(self.wavelength),
            dtype=float,
        )


    def gamma_transmission(self) -> np.ndarray:
        return self.gamma_transmission_nominal.copy()

    def detector_efficiency(self, qe=None, mirror=None, window=None) -> np.ndarray:
        qe = self.qe_nominal if qe is None else np.asarray(qe, dtype=float)
        mirror = self.mirror_nominal if mirror is None else np.asarray(mirror, dtype=float)
        window = self.window_nominal if window is None else np.asarray(window, dtype=float)
        return self._safe_clip(qe * mirror * window)

    def muon_response(self, **kwargs) -> np.ndarray:
        return self.detector_efficiency() * self.muon_transmission(**kwargs)

    def gamma_response(self) -> np.ndarray:
        return self.detector_efficiency() * self.gamma_transmission()

    def bandwidth_muon(self, **kwargs) -> float:
        return float(np.sum(self.muon_response(**kwargs)) * self.energy_step)

    def bandwidth_gamma(self) -> float:
        return float(np.sum(self.gamma_response()) * self.energy_step)

    def ratio_gamma_to_muon(self, **kwargs) -> float:
        return self.bandwidth_gamma() / self.bandwidth_muon(**kwargs)

    def cumulative_blindness_curve(self, blind_energy_ev: np.ndarray | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Equivalent in spirit to the blindness / cutoff scans in bandwidth.py.
        Returns (energy_grid, fractional loss curve).
        """
        if blind_energy_ev is None:
            blind_energy_ev = self.energy

        blind_energy_ev = np.asarray(blind_energy_ev, dtype=float)
        gamma_comb = self.gamma_response()
        muon_comb = self.muon_response(**kwargs)

        c_gamma = np.cumsum(gamma_comb) * self.energy_step
        c_muon = np.cumsum(muon_comb) * self.energy_step
        total_gamma = c_gamma[-1]
        total_muon = c_muon[-1]

        loss = (c_gamma - c_muon * total_gamma / total_muon) / np.maximum(c_gamma, 1e-30)
        return self.energy.copy(), loss

    def propagate_uncertainty(self, cfg: UncertaintyConfig | None = None) -> dict:
        """
        Monte Carlo propagation for:
          - B_mu
          - B_gamma
          - B_gamma / B_mu
        """
        if cfg is None:
            cfg = UncertaintyConfig()

        rng = np.random.default_rng(cfg.random_seed)

        bmu = np.empty(cfg.n_mc, dtype=float)
        bgam = np.empty(cfg.n_mc, dtype=float)
        ratio = np.empty(cfg.n_mc, dtype=float)

        for i in range(cfg.n_mc):
            # sample atmosphere / geometry nuisances
            aod_i = max(1e-6, rng.normal(self.aod, cfg.sigma_aod))
            H_i = max(1e-6, rng.normal(self.H, cfg.sigma_H))
            AA_i = rng.normal(self.AA, cfg.sigma_AA)
            theta_c_i = rng.normal(self.theta_c_deg, cfg.sigma_theta_c_deg)
            rhoR_i = rng.normal(self.rhoR, cfg.sigma_rhoR)

            # detector nuisance sampling
            if cfg.use_detector_qe_uncertainty and self.qe_sigma is not None:
                qe_i = self.qe_nominal + rng.normal(0.0, self.qe_sigma, size=self.qe_nominal.shape)
            else:
                qe_i = self.qe_nominal.copy()

            mirror_i = self.mirror_nominal * (1.0 + rng.normal(0.0, cfg.rel_mirror_unc, size=self.mirror_nominal.shape))
            window_i = self.window_nominal * (1.0 + rng.normal(0.0, cfg.rel_window_unc, size=self.window_nominal.shape))

            det_i = self.detector_efficiency(qe=qe_i, mirror=mirror_i, window=window_i)
            mu_i = det_i * self.muon_transmission(
                aod=aod_i,
                H=H_i,
                AA=AA_i,
                theta_c_deg=theta_c_i,
                rhoR=rhoR_i,
            )
            ga_i = det_i * self.gamma_transmission()

            bmu[i] = np.sum(mu_i) * self.energy_step
            bgam[i] = np.sum(ga_i) * self.energy_step
            ratio[i] = bgam[i] / max(bmu[i], 1e-30)

        def summary(x: np.ndarray) -> dict:
            return {
                "mean": float(np.mean(x)),
                "std": float(np.std(x, ddof=1)),
                "median": float(np.median(x)),
                "p16": float(np.percentile(x, 16)),
                "p84": float(np.percentile(x, 84)),
            }

        return {
            "B_mu": summary(bmu),
            "B_gamma": summary(bgam),
            "ratio_gamma_to_muon": summary(ratio),
            "samples": {
                "B_mu": bmu,
                "B_gamma": bgam,
                "ratio_gamma_to_muon": ratio,
            },
        }


print("Muon bandwidth:", model.bandwidth_muon())
print("Gamma bandwidth:", model.bandwidth_gamma())
print("Ratio:", model.ratio_gamma_to_muon())

unc = model.propagate_uncertainty(
    UncertaintyConfig(
        n_mc=300,
        sigma_aod=0.01,
        sigma_H=100.0,
        sigma_AA=0.15,
        sigma_theta_c_deg=0.02,
        sigma_rhoR=0.02,
    )
)

print(unc["B_mu"])
print(unc["B_gamma"])
print(unc["ratio_gamma_to_muon"])
'''
