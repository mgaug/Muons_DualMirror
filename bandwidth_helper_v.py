import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.interpolate import interp1d


def nm2ev(x):
    x = np.asarray(x, dtype=float)
    out = 1239.842 / x
    return float(out) if np.ndim(out) == 0 else out


def ev2nm(x):
    x = np.asarray(x, dtype=float)
    out = 1239.842 / x
    return float(out) if np.ndim(out) == 0 else out


@dataclass
class Measurements:
    wavelength_nm: np.ndarray
    energy_ev: np.ndarray
    values: np.ndarray
    sigma: np.ndarray | None = None


class BandwidthHelper:
    """
    Vectorized replacement for bandwidth_helper.py
    """

    def __init__(self,
                 xi_steps: float = 0.05,
                 qe_file: str    = "data/qe_R12992-100-05.dat",   # QE of the R12992 PMT
                 qe_file_mult    = 1.,
                 si_file: str    = "data/qe_S13360_75pe_Hamamatsu.dat", # QE of the SiPM S13360
                 si_file_mult    = 0.01,                 
                 mi_file: str    = "data/ref_AlSiO2HfO2.dat",  # Reflectivity of the mirrors used for Prod3
                 mi_file_mult    = 1.,                 
                 ca_file: str    = "data/Aclylite8_tra_v2013ref.dat",  # Camera protection window transparency
                 ca_file_mult    = 1.,                 
                 cs_file: str    = "data/ASTRI_fullslitheight_0deg.csv", # Transparency of the ASTRI camera protection window
                 cs_file_mult    = 0.01,
                 e_pmt_min       = 1.5596,
                 e_pmt_max       = 4.8,
                 e_sipm_min      = 1.3777,
                 e_sipm_max      = 4.5,
                 e_pmt_nocam_max = 5.5
    ):
        self.xi_steps = float(xi_steps)

        self._load_tables(
            qe_file, qe_file_mult,
            si_file, si_file_mult,
            mi_file, mi_file_mult,
            ca_file, ca_file_mult,
            cs_file, cs_file_mult,
        )        
        self._build_interpolators()
        self._build_energy_grids(e_pmt_min=e_pmt_min,e_pmt_max=e_pmt_max,
                                 e_sipm_min=e_sipm_min,e_sipm_max=e_sipm_max,
                                 e_pmt_nocam_max=e_pmt_nocam_max)
        self._build_element_products()

        self.summary = {
            "QE of PMTs from ": qe_file,
            "QE of SiPMs from ": si_file,
            "Mirror reflectivity from ": mi_file,
            "Camera window transparency from ": ca_file,
            "SST camera transparency from ": cs_file,
            "QE PMTs multiplier ": qe_file_mult,
            "QE SiPMs multiplier ": si_file_mult,
            "Mirror refl. multiplier ": mi_file_mult,
            "Camera window trans. multiplier ": ca_file_mult,
            "SST camera trans. multiplier ": cs_file_mult,
            "Interpolated energy grid steps (eV) ": xi_steps,
            "Min. photon energy PMT (eV) ": e_pmt_min,
            "Max. photon energy PMT (eV) ": e_pmt_max,
            "Min. photon energy SiPM (eV) ": e_sipm_min,
            "Max. photon energy SiPM (eV) ": e_sipm_max,
            "Max. photon energy PMT w/o camera (eV) ": e_pmt_nocam_max,
            "Min. photon wavelength PMT (nm) ": ev2nm(e_pmt_max),
            "Max. photon wavelength PMT (nm) ": ev2nm(e_pmt_min),
            "Min. photon wavelength SiPM (nm) ": ev2nm(e_sipm_max),
            "Max. photon wavelength SiPM (nm) ": ev2nm(e_sipm_min),
            "Min. photon wavelength PMT w/o camera (nm) " : ev2nm(e_pmt_nocam_max)
        }

        self.print_summary()


    def print_summary(self):
        s = self.summary
        print("Bandwidth summary")
        print("-" * 50)
        for k, v in s.items():
            print(f"{k:40s}: {v}")
        print('\n')            

        
    # ----------------------------
    # loading
    # ----------------------------
    def _load_tables(self,
                     qe_file, qe_file_mult,
                     si_file,si_file_mult,
                     mi_file,mi_file_mult,
                     ca_file,ca_file_mult,
                     cs_file,cs_file_mult):
        qe_tab = pd.read_table(
            qe_file,
            sep=r"\s+",
            skip_blank_lines=True,
            comment="#",
            names=["Wavelength", "meanQE", "stdev", "minQE", "maxQE"],
        )

        si_tab = pd.read_table(
            si_file,
            sep=r"\s+",
            skip_blank_lines=True,
            comment="#",
            names=["Wavelength", "meanQE"],
        )

        mi_tab = pd.read_table(
            mi_file,
            sep=r"\s+",
            skip_blank_lines=True,
            comment="#",
            names=["Wavelength", "meanRef"],
        )

        ca_tab = pd.read_table(
            ca_file,
            sep=r"\s+",
            skip_blank_lines=True,
            comment="#",
            names=["Wavelength", "meanRef"],
        )

        cs_tab = pd.read_table(
            cs_file,
            sep=",",
            skip_blank_lines=True,
            comment="#",
            names=["Wavelength", "meanRef", "dummy"],
        )

        self.qe_pmt = Measurements(
            wavelength_nm=np.asarray(qe_tab["Wavelength"], dtype=float),
            energy_ev=nm2ev(qe_tab["Wavelength"].to_numpy(dtype=float)),
            values=np.asarray(qe_tab["meanQE"], dtype=float),
            sigma=np.asarray(qe_tab["stdev"], dtype=float) * qe_file_mult,
        )

        self.qe_sipm = Measurements(
            wavelength_nm=np.asarray(si_tab["Wavelength"], dtype=float),
            energy_ev=nm2ev(si_tab["Wavelength"].to_numpy(dtype=float)),
            values=np.asarray(si_tab["meanQE"], dtype=float) * si_file_mult,
        )

        self.mirror = Measurements(
            wavelength_nm=np.asarray(mi_tab["Wavelength"], dtype=float),
            energy_ev=nm2ev(mi_tab["Wavelength"].to_numpy(dtype=float)),
            values=np.asarray(mi_tab["meanRef"], dtype=float) * mi_file_mult,
        )

        self.cam_pmt = Measurements(
            wavelength_nm=np.asarray(ca_tab["Wavelength"], dtype=float),
            energy_ev=nm2ev(ca_tab["Wavelength"].to_numpy(dtype=float)),
            values=np.asarray(ca_tab["meanRef"], dtype=float) * ca_file_mult,
        )

        self.cam_sipm = Measurements(
            wavelength_nm=np.asarray(cs_tab["Wavelength"], dtype=float),
            energy_ev=nm2ev(cs_tab["Wavelength"].to_numpy(dtype=float)),
            values=np.asarray(cs_tab["meanRef"], dtype=float) * cs_file_mult,
        )

    # ----------------------------
    # interpolators
    # ----------------------------
    @staticmethod
    def _make_interp(x, y):
        order = np.argsort(x)
        x = np.asarray(x)[order]
        y = np.asarray(y)[order]
        return interp1d(
            x,
            y,
            kind="linear",
            bounds_error=False,
            fill_value=(y[0], y[-1]),
            assume_sorted=True,
        )

    def _build_interpolators(self):
        self.qe_int = self._make_interp(self.qe_pmt.energy_ev, self.qe_pmt.values)
        self.qe_sigma_int = self._make_interp(self.qe_pmt.energy_ev, self.qe_pmt.sigma)

        self.si_int = self._make_interp(self.qe_sipm.energy_ev, self.qe_sipm.values)
        self.mi_int = self._make_interp(self.mirror.energy_ev, self.mirror.values)
        self.ca_int = self._make_interp(self.cam_pmt.energy_ev, self.cam_pmt.values)
        self.cs_int = self._make_interp(self.cam_sipm.energy_ev, self.cam_sipm.values)

    # ----------------------------
    # element grids
    # ----------------------------
    def _build_energy_grids(self, e_pmt_min=1.5596, e_pmt_max=4.8, e_sipm_min=1.3777,e_sipm_max=4.5, e_pmt_nocam_max=5.5):
        self.xi_e_pmt = np.arange(e_pmt_min, e_pmt_max, self.xi_steps)
        self.xi_e_pmt_nocam = np.arange(e_pmt_min, e_pmt_nocam_max, self.xi_steps)

        self.xi_e_sipm = np.arange(e_sipm_min, e_sipm_max, self.xi_steps)
        self.xi_e_sipm_nocam = np.arange(e_sipm_min, e_sipm_max, self.xi_steps)

        self.xi_wl_pmt = ev2nm(self.xi_e_pmt)
        self.xi_wl_pmt_nocam = ev2nm(self.xi_e_pmt_nocam)
        self.xi_wl_sipm = ev2nm(self.xi_e_sipm)
        self.xi_wl_sipm_nocam = ev2nm(self.xi_e_sipm_nocam)

    # ----------------------------
    # precomputed element products
    # ----------------------------
    def _build_element_products(self):
        self.xi_det_pmt = self.detector_efficiency("PMT", self.xi_e_pmt, with_camera=True)
        self.xi_det_pmt_nocam = self.detector_efficiency("PMT", self.xi_e_pmt_nocam, with_camera=False)

        self.xi_det_sipm = self.detector_efficiency("SIPM", self.xi_e_sipm, with_camera=True)
        self.xi_det_sipm_nocam = self.detector_efficiency("SIPM", self.xi_e_sipm_nocam, with_camera=False)

        self.integrated_xi_det_pmt = np.sum(self.xi_det_pmt) * self.xi_steps
        self.integrated_xi_det_sipm = np.sum(self.xi_det_sipm) * self.xi_steps


    def detector_efficiency(self, element: str, energy_ev, with_camera: bool = True):
        energy_ev = np.asarray(energy_ev, dtype=float)

        if element.upper() == "PMT":
            qe = self.qe_int(energy_ev)
            mirror = self.mi_int(energy_ev)
            if with_camera:
                window = self.ca_int(energy_ev)
                out = qe * mirror * window
            else:
                out = qe * mirror

        elif element.upper() == "SIPM":
            qe = self.si_int(energy_ev)
            mirror = self.mi_int(energy_ev)
            if with_camera:
                window = self.cs_int(energy_ev)
                out = qe * mirror * window
            else:
                out = qe * mirror

        else:
            raise ValueError(f"Unknown element: {element}")

        return np.clip(out, 0.0, 1.0)

    def detector_efficiency_uncertainty(self, element: str, energy_ev):
        """
        Returns 1-sigma uncertainty on element QE/PDE where available.
        Currently only PMT QE has tabulated sigma in the original file.
        """
        energy_ev = np.asarray(energy_ev, dtype=float)

        if element.upper() == "PMT":
            return np.clip(self.qe_sigma_int(energy_ev), 0.0, None)

        if element.upper() == "SIPM":
           return np.ones_like(energy_ev) * 0.01   # current guess TO BE FIXED!!!!

        raise ValueError(f"Unknown element: {element}")

    def integrated_efficiency(self, element: str, with_camera: bool = True):
        if element.upper() == "PMT":
            grid = self.xi_e_pmt if with_camera else self.xi_e_pmt_nocam
        elif element.upper() == "SIPM":
            grid = self.xi_e_sipm if with_camera else self.xi_e_sipm_nocam
        else:
            raise ValueError(f"Unknown element: {element}")

        return np.sum(self.detector_efficiency(element, grid, with_camera=with_camera)) * self.xi_steps


    @staticmethod
    def get_low_idx(arr, xi_limit):
        """
        First index where arr > xi_limit.
        """
        arr = np.asarray(arr)
        idx = np.searchsorted(arr > xi_limit, True)
        return int(idx)

    @staticmethod
    def get_high_idx(arr, xi_limit):
        """
        Last index before arr falls below xi_limit, searching from the mid-point.
        """
        arr = np.asarray(arr)
        start = arr.size // 2
        sub = arr[start:]
        below = np.flatnonzero(sub < xi_limit)
        if below.size == 0:
            return int(arr.size - 1)
        return int(start + below[0] - 1)
    
