import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, sqrt, pi

# own modules
from bandwidth_helper_v import BandwidthHelper
from muonlight_averager_v import MuonModel, UncertaintyConfig, AtmFileConfig
import telescope as tel
import matplotlib
import setup
from setup import has_display

# script configurations
plot_contributions = True
plot_xidet         = True
plot_zenith        = True
full_accuracy      = False     # fully accurate simulations, but very slow!
verbose_output     = False     # more information about all used parameters

# general configuration
theta_tel = 0.    # default telescope pointing zenith angle, in degrees
theta_c   = 1.1   # average muon Cherenkov angle used in analysis, in degrees
rhoR_min  = 0.1   # reference lower impact distance cut, in rhoR/R1
scale_h   = 9500  # La Palma Winter reference molecular density scale height
tel_height = 2200 # default telescope height asl in meters, SHOULD BE ADJUSTED FOR EACH TELESCOPE INDIVIDUALLY

bh = BandwidthHelper(verbose=verbose_output) # TO MODIFY THE BANDWIDTH MEASUREMENTS of the optical elements, EDIT THE FILE bandwidth_helper_v.py

# FOR THE CALCULATIONS OF B_gamma, PLEASE CAREFULLY REVISE THE ATMOSPHERIC TRANSMISSION TABLE USED FOR atm_file
# AND THE (WRONG) ASSUMPTIONS USED IN IT AND THEIR UNCERTAINTIES IN THE DATACLASS AtmFileConfig in atmosphere_helper_v.py
atm_file = "data/atm_trans_2147_1_10_0_0_2147.dat"
atm_config = AtmFileConfig()

# DEFAULT MUON MODEL IS A CLEAR NIGHT. TO SIMULATE ANYTHING, USE THE DEFAULT CONSTRUCTOR MuonModel() with the corresponding arguments, see muonlight_averager_v.py

# LST-N
#========

model = MuonModel.from_LSTN(bandwidth=bh, tel_height=tel_height, theta_tel_deg=theta_tel, theta_c_deg=theta_c, rhoR_min=rhoR_min, scale_h=scale_h, atm_file=atm_file, atm_cfg=atm_config)
if verbose_output:
    model.print_summary()

print("\n\nLST-N:")    
print("B_mu =", model.bandwidth_muon())
print("B_gamma =", model.bandwidth_gamma())
print("B_gamma/B_mu   =", model.ratio_gamma_to_muon())

n_mc = 200 if full_accuracy else 20
unc = model.simulate_uncertainty(        
    UncertaintyConfig(), 
    n_mc=n_mc, verbose = verbose_output,
    full_accuracy=full_accuracy    
)
print(f"B_mu simulation results for theta_tel={theta_tel:.0f}: ",unc["B_mu"])
print(f"B_gamma simulation results for theta_tel={theta_tel:.0f}: ",unc["B_gamma"])
print(f"B_gamma/B_mu simulation results for theta_tel={theta_tel:.0f}: ",unc["ratio_gamma_to_muon"])

# MST-N
#========

model = MuonModel.from_MSTN(bandwidth=bh, tel_height=tel_height, theta_tel_deg=theta_tel, theta_c_deg=theta_c, rhoR_min=rhoR_min, scale_h=scale_h, atm_file=atm_file, atm_cfg=atm_config)
if verbose_output:
    model.print_summary()

print("\n\nMST-N:")
print("B_mu =", model.bandwidth_muon())
print("B_gamma =", model.bandwidth_gamma())
print("B_gamma/B_mu   =", model.ratio_gamma_to_muon())

n_mc = 200 if full_accuracy else 20
unc = model.simulate_uncertainty(        
    UncertaintyConfig(), 
    n_mc=n_mc, verbose = verbose_output,
    full_accuracy=full_accuracy    
)
print(f"B_mu simulation results for theta_tel={theta_tel:.0f}: ",unc["B_mu"])
print(f"B_gamma simulation results for theta_tel={theta_tel:.0f}: ",unc["B_gamma"])
print(f"B_gamma/B_mu simulation results for theta_tel={theta_tel:.0f}: ",unc["ratio_gamma_to_muon"])

# LST-S
#========

model = MuonModel.from_LSTS(bandwidth=bh, tel_height=tel_height, theta_tel_deg=theta_tel, theta_c_deg=theta_c, rhoR_min=rhoR_min, scale_h=scale_h, atm_file=atm_file, atm_cfg=atm_config)
if verbose_output:
    model.print_summary()

print("\n\nLST-S:")    
print("B_mu =", model.bandwidth_muon())
print("B_gamma =", model.bandwidth_gamma())
print("B_gamma/B_mu   =", model.ratio_gamma_to_muon())

n_mc = 200 if full_accuracy else 20
unc = model.simulate_uncertainty(        
    UncertaintyConfig(), 
    n_mc=n_mc, verbose = verbose_output,
    full_accuracy=full_accuracy    
)
print(f"B_mu simulation results for theta_tel={theta_tel:.0f}: ",unc["B_mu"])
print(f"B_gamma simulation results for theta_tel={theta_tel:.0f}: ",unc["B_gamma"])
print(f"B_gamma/B_mu simulation results for theta_tel={theta_tel:.0f}: ",unc["ratio_gamma_to_muon"])

# MST-S
#========

model = MuonModel.from_MSTS(bandwidth=bh, tel_height=tel_height, theta_tel_deg=theta_tel, theta_c_deg=theta_c, rhoR_min=rhoR_min, scale_h=scale_h, atm_file=atm_file, atm_cfg=atm_config)
if verbose_output:
    model.print_summary()

print("\n\nMST-S:")    
print("B_mu =", model.bandwidth_muon())
print("B_gamma =", model.bandwidth_gamma())
print("B_gamma/B_mu   =", model.ratio_gamma_to_muon())

n_mc = 200 if full_accuracy else 20
unc = model.simulate_uncertainty(        
    UncertaintyConfig(), 
    n_mc=n_mc, verbose = verbose_output,
    full_accuracy=full_accuracy    
)
print(f"B_mu simulation results for theta_tel={theta_tel:.0f}: ",unc["B_mu"])
print(f"B_gamma simulation results for theta_tel={theta_tel:.0f}: ",unc["B_gamma"])
print(f"B_gamma/B_mu simulation results for theta_tel={theta_tel:.0f}: ",unc["ratio_gamma_to_muon"])

# SST-S
#========

model = MuonModel.from_SSTS(bandwidth=bh, tel_height=tel_height, theta_tel_deg=theta_tel, theta_c_deg=theta_c, rhoR_min=rhoR_min, scale_h=scale_h, atm_file=atm_file, atm_cfg=atm_config)
if verbose_output:
    model.print_summary()

print("\n\nSST-S:")    
print("B_mu =", model.bandwidth_muon())
print("B_gamma =", model.bandwidth_gamma())
print("B_gamma/B_mu   =", model.ratio_gamma_to_muon())

n_mc = 200 if full_accuracy else 30
unc = model.simulate_uncertainty(        
    UncertaintyConfig(), 
    n_mc=n_mc, verbose = verbose_output,
    full_accuracy=full_accuracy    
)
print(f"B_mu simulation results for theta_tel={theta_tel:.0f}: ",unc["B_mu"])
print(f"B_gamma simulation results for theta_tel={theta_tel:.0f}: ",unc["B_gamma"])
print(f"B_gamma/B_mu simulation results for theta_tel={theta_tel:.0f}: ",unc["ratio_gamma_to_muon"])

#==========================================================================================================

# B vs. zenith

# BEWARE THIS PART IS VERY SLOW IF full_accuracy IS CHOSEN. IF YOU WANT TO SPEED UP,
# PLEASE SET THIS FLAG TO FALSE (AT THE PRICE OF LOWER ACCURACY OF THE RESULTS)
fig, ax = plt.subplots(3,2,constrained_layout=True,figsize=(10,10))
MuonModel.plot_bandwidth_vs_zenith(
    filename="output/B_vs_zenith{s}.pdf".format(s="_full_accuracy" if full_accuracy is True else "_reduced_accuracy"),
    ax=ax, uncertainties=True, full_accuracy=full_accuracy, verbose=verbose_output, show=has_display
)

fig, ax = plt.subplots(3,1,constrained_layout=True,figsize=(20,15))
MuonModel.plot_transmission_vs_zenith(
    filename="output/T_vs_zenith.pdf", ax=ax, show=has_display
)
# 

if plot_contributions: 

    fig, ax = plt.subplots(constrained_layout=True)
    model.plot_contributions(filename="output/Bandwidth_LSTN.pdf", ax=ax,show=has_display)


if plot_xidet: 

    # standard LST/MST/SST comparison set
    models = MuonModel.build_standard_models()

    #Should reproduce Figure 19 of https://iopscience.iop.org/article/10.3847/1538-4365/ab2123
    # with < 0.5% differences for values involving t_mu due to updated atmospheric aerosol model
    # Larger differences are found for t_gamma because of a better estimate of the median observed
    # gamma-ray emission height Hgamma 
    fig, ax = plt.subplots(constrained_layout=True)
    MuonModel.plot_xidet_comparison(
        models=models,
        filename="output/Bandwidth2.pdf", ax=ax,show=has_display
    )
    
    fig, ax = plt.subplots(constrained_layout=True)
    MuonModel.plot_ratio_comparison(
        models=models,
        filename="output/Bandwidth_Ratio.pdf", ax=ax,show=has_display
    )
    
    fig, ax = plt.subplots(2,1,constrained_layout=True)
    MuonModel.plot_pde_and_transparency(
        models=models,
        filename="output/Bandwidth_PMT.pdf", ax=ax,show=has_display
    )

