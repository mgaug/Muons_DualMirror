import numpy as np

def D(rhoR, phi):
    """

    A simplified version of the chord, following Vacanti's formula
    10.1016/0927-6505(94)90012-4, Eq. (6)

    This function is used both for approximations of the maximum 
    and minimum emission heights and for shadows following Eq. (38) 
    of https://iopscience.iop.org/article/10.3847/1538-4365/ab2123. 

    Parameters
    ----------
    rhoR : float or array-like
        Normalized impact parameter.
    phi : float or array-like
        Angle in radians.

    Returns
    -------
    ndarray or float
        Same broadcasted shape as np.broadcast(rhoR, phi).
    """
    rhoR_arr, phi_arr = np.broadcast_arrays(
        np.asarray(rhoR, dtype=float),
        np.asarray(phi, dtype=float),
    )


    sx = np.sin(phi_arr)
    cx = np.cos(phi_arr)

    arg = rhoR_arr * np.sin(phi_arr)
    out = np.zeros_like(arg, dtype=float)

    valid = arg <= 1.0

    mask_eq = valid & np.isclose(rhoR_arr, 1.0) 
    out[mask_eq] = np.abs(np.cos(phi_arr[mask_eq])) + np.cos(phi_arr[mask_eq])

    mask_gt = valid & (rhoR_arr > 1.0) & ~np.isclose(rhoR_arr, 1.0) & (cx > 0.0)
    out[mask_gt] = 2.0 * np.sqrt(np.clip(1.0 - arg[mask_gt] ** 2, 0.0, None))

    mask_lt = valid & (rhoR_arr < 1.0) & ~np.isclose(rhoR_arr, 1.0)
    out[mask_lt] = (
        np.sqrt(1.0 - arg[mask_lt] ** 2)
        + rhoR_arr[mask_lt] * cx[mask_lt]
    )

    if out.ndim == 0:
        return float(out)
    return out


def ev2nm(eV):

    eV = np.asarray(eV, dtype=float)
    out = 1239.842 / eV
    if np.isscalar(eV):
        return float(out)
    return out


def nm2ev(nm):
    
    nm = np.asarray(nm, dtype=float)
    out = 1239.842 / nm
    if np.isscalar(nm):
        return float(out)
    return out
