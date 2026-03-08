import matplotlib as mpl

def SetUp():
    mpl.rcParams.update({
        
        # --- Figure ---
        "figure.figsize": (7.5, 6.0),   # inches
        "figure.dpi": 100,
        "savefig.dpi": 300,    
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        
        # --- Fonts ---
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "mathtext.fontset": "stix",
        
        # --- Axes ---
        "axes.linewidth": 1.2,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        
        # --- Ticks ---
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        
        # --- Lines ---
        "lines.linewidth": 3,
        "lines.markersize": 6,
        
        # --- Legend ---
        "legend.frameon": False,
        "legend.loc": "best",
        "legend.handlelength": 2.5,
        
        # --- Errorbars ---
        "errorbar.capsize": 3,
        
        "axes.titley": 1.07,
        
        # --- LaTeX (optional) ---
        # "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{amsmath}",

    })


