import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, sqrt, pi

# own modules
from bandwidth_helper_v import BandwidthHelper
from muonlight_averager_v import MuonModel
import telescope as tel
import atmosphere_helper_v as ah
import setup

import matplotlib
matplotlib.use("TkAgg")   # or "QtAgg"

plot_contributions = True
plot_xidet         = True
plot_ratio         = True
plot_pmt           = True
# style stuff
fig_size = plt.rcParams['figure.figsize']
lab_size = plt.rcParams['axes.labelsize']
leg_size = plt.rcParams['legend.fontsize']
fig_size[0] = fig_size[0]*1.65
#plt.rcParams['figure.figsize'] = fig_size
plt.rcParams['legend.fontsize'] = leg_size+1


# general configuration
theta_tel = 0.    # telescope pointing angle, in degrees
theta_c   = 1.23  # Cherenkov angle, in degrees
rhoR_min  = 0.1   # reference lower impact distance cut
scale_h   = 9700  # La Palma Winter molecular density scale height

bh = BandwidthHelper()        

model = MuonModel.from_LSTN(bandwidth=bh, theta_tel_deg=theta_tel, theta_c_deg=theta_c, rhoR_min=rhoR_min, scale_h=scale_h)
model.print_summary()

print("B_mu    =", model.bandwidth_muon())
print("B_gamma =", model.bandwidth_gamma())
print("ratio   =", model.ratio_gamma_to_muon())

fig, ax = plt.subplots(constrained_layout=True)
model.plot_contributions(filename="output/Bandwidth_LSTN.pdf", ax=ax)
plt.show()

# standard LST/MST/SST comparison set
models = MuonModel.build_standard_models()

fig, ax = plt.subplots(constrained_layout=True)
MuonModel.plot_xidet_comparison(
    models=models,
    filename="output/Bandwidth2.pdf", ax=ax
)
plt.show()

fig, ax = plt.subplots(constrained_layout=True)
MuonModel.plot_ratio_comparison(
    models=models,
    filename="output/Bandwidth_Ratio.pdf", ax=ax
)
plt.show()

fig, ax = plt.subplots(2,1,constrained_layout=True)
MuonModel.plot_pde_and_transparency(
    models=models,
    filename="output/Bandwidth_PMT.pdf", ax=ax
)
plt.show()

r'''
if (plot_contributions):

    f, ax = plt.subplots()
    ax.plot(bh.qe_e,bh.qe_ham, 'm--',lw=2,label='Photomultiplier QE')
    ax.plot(bh.si_e,bh.qe_si,  'c-.',lw=2,label='SiPM PDE')
    ax.plot(bh.mi_e,bh.mi_ref, 'b-', label='Mirror Reflectivity')
    ax.plot(bh.ca_e,bh.ca_ref, 'r-', lw=2,label='Protection Window Transparency')
    ax.legend(bbox_to_anchor=(0.99, 0.98),loc='upper right')

    ax.set_xlabel('Photon energy $\epsilon$ (eV)')
    ax.set_ylabel(r'efficiency $\xi$')
    ax.set_ylim([0.,1.05])
    ax.set_xlim([1.3,6.1])

    ax_c = ax.twiny()
    wls  = np.array([900,800,700,600,500,400,300,250,200])
    wlticks = [bh.nm2ev(wl) for wl in wls]
    ax_c.set_xticks(wlticks);
    ax_c.set_xticklabels(['{:g}'.format(wl) for wl in wls]);
    ax_c.set_xlabel('Photon wavelength (nm)')
    
    xmin, xmax = 1.2, 6.2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.05, 1.05)
    ax_c.set_xlim(xmin, xmax);
    
    plt.savefig('output/Bandwidth.pdf',bbox_inches='tight')
    plt.show()
    plt.close()
    

if (plot_xidet):

    plt.rcParams['legend.fontsize'] = leg_size-1

    f, ax = plt.subplots()
    
    aod = 0.03
    H   = 600
    AA  = 1.2

    tmulst_aer    = [get_av_transmission_phi_aer(e,R_LST,inner_R_LST,thetac,rhoR,costheta,aod,H,AA) for e in bh.xi_e_pmt]    
    tmulst_mol    = [get_av_transmission_phi_mol(e,R_LST,inner_R_LST,thetac,rhoR,costheta) for e in bh.xi_e_pmt]    
#    tgammalst_aer = [get_av_transmission_aer(e,Hgamma_LST,costheta,aod,H,AA) for e in bh.xi_e_pmt]    
#    tgammalst_mol = [get_av_transmission_mol(e,Hgamma_LST,costheta) for e in bh.xi_e_pmt]    

    tmusst_aer    = [get_av_transmission_phi_aer(e,R_SST,inner_R_SST,thetac,rhoR,costheta,aod,H,AA) for e in bh.xi_e_sipm]    
    tmusst_mol    = [get_av_transmission_phi_mol(e,R_SST,inner_R_SST,thetac,rhoR,costheta) for e in bh.xi_e_sipm]    
#    tgammasst_aer = [get_av_transmission_aer(e,Hgamma_SST,costheta,aod,H,AA) for e in bh.xi_e_sipm]    
#    tgammasst_mol = [get_av_transmission_mol(e,Hgamma_SST,costheta) for e in bh.xi_e_sipm]    

    tmulst_comb    = bh.xi_det_pmt * tmulst_mol    * tmulst_aer
    tgammalst_atm  = trans_lst_int(bh.xi_wl_pmt)            # tgammasst_mol  * tgammasst_aer                
    tgammalst_comb = bh.xi_det_pmt * tgammalst_atm

    tmusst_comb    = bh.xi_det_sipm * tmusst_mol    * tmusst_aer
    tgammasst_atm  = trans_sst_int(bh.xi_wl_sipm)            # tgammasst_mol  * tgammasst_aer            
    tgammasst_comb = bh.xi_det_sipm * tgammasst_atm

#    ax.plot(bh.xi_e_pmt,bh.xi_det_pmt, '--',label=r'$\xi_{det}$'+' for LST, total bandwidth: {:.3f} eV'.format(bh.integrated_xi_det_pmt))
    ax.plot(bh.xi_e_pmt,tmulst_comb, '-',color='b',label=r'$\xi_{det}\cdot t_{\mu}$'+r' LST, B$_\mu$: '+'{:.3f} eV'.format(tmulst_comb.sum()*bh.xi_steps))
    ax.plot(bh.xi_e_pmt,tgammalst_comb, '--',color='cornflowerblue',label=r'$\xi_{det}\cdot t_{\gamma}$'+r' LST, B$_\gamma$: '+'{:.3f} eV'.format(tgammalst_comb.sum()*bh.xi_steps))
#    ax.plot(bh.xi_e_sipm,bh.xi_det_sipm, '--',label=r'$\xi_{det}$'+' for SST, total bandwidth: {:.3f} eV'.format(bh.integrated_xi_det_sipm))    
    ax.plot(bh.xi_e_sipm,tmusst_comb, '-',color='r',label=r'$\xi_{det}\cdot t_{\mu}$'+r' SST, B$_\mu$: '+'{:.3f} eV'.format(tmusst_comb.sum()*bh.xi_steps))
    ax.plot(bh.xi_e_sipm,tgammasst_comb, '--',color='darkorange',label=r'$\xi_{det}\cdot t_{\gamma}$'+r' SST, B$_\gamma$: '+'{:.3f} eV'.format(tgammasst_comb.sum()*bh.xi_steps))

    ax.legend(bbox_to_anchor=(1.0, 1.0),loc=1)

    ax.set_xlabel('Photon energy $\epsilon$ (eV)')
    ax.set_ylabel(r'efficiency $\xi$')
    ax.set_ylim([0.,0.48])
#    ax.set_xlim([1.4,6.1])

    ax_c = ax.twiny()
    wls  = np.array([800,700,600,500,400,300,250])
    wlticks = [bh.nm2ev(wl) for wl in wls]
    ax_c.set_xticks(wlticks);
    ax_c.set_xticklabels(['{:g}'.format(wl) for wl in wls]);
    ax_c.set_xlabel('Photon wavelength (nm)')
    
    xmin, xmax = 1.4, 5.0
    ax.set_xlim(xmin, xmax)
    ax_c.set_xlim(xmin, xmax);
    
    plt.savefig('output/Bandwidth2.pdf',bbox_inches='tight')
    plt.show()
    plt.close()


if (plot_ratio):

    f, ax = plt.subplots()
    
    aod = 0.03
    H   = 600
    AA  = 1.2

    tmulst_aer    = np.array( [get_av_transmission_phi_aer(e,R_LST,inner_R_LST,thetac,rhoR,costheta,aod,H,AA) for e in bh.xi_e_pmt] )
    tmulst_mol    = np.array( [get_av_transmission_phi_mol(e,R_LST,inner_R_LST,thetac,rhoR,costheta)          for e in bh.xi_e_pmt] )
    tgammalst_aer = np.array( [get_av_transmission_aer(e,Hgamma_LST,costheta,aod,H,AA)                        for e in bh.xi_e_pmt] )    
    tgammalst_mol = np.array( [get_av_transmission_mol(e,Hgamma_LST,costheta)                                 for e in bh.xi_e_pmt] )    

    tmumst_aer    = np.array( [get_av_transmission_phi_aer(e,R_MST,inner_R_MST,thetac,rhoR,costheta,aod,H,AA) for e in bh.xi_e_pmt] )    
    tmumst_mol    = np.array( [get_av_transmission_phi_mol(e,R_MST,inner_R_MST,thetac,rhoR,costheta)          for e in bh.xi_e_pmt] )    
    tgammamst_aer = np.array( [get_av_transmission_aer(e,Hgamma_MST,costheta,aod,H,AA)                        for e in bh.xi_e_pmt] )    
    tgammamst_mol = np.array( [get_av_transmission_mol(e,Hgamma_MST,costheta)                                 for e in bh.xi_e_pmt] )    

    tmusst_aer    = np.array( [get_av_transmission_phi_aer(e,R_SST,inner_R_SST,thetac,rhoR,costheta,aod,H,AA) for e in bh.xi_e_sipm] )    
    tmusst_mol    = np.array( [get_av_transmission_phi_mol(e,R_SST,inner_R_SST,thetac,rhoR,costheta)          for e in bh.xi_e_sipm] )    
    tgammasst_aer = np.array( [get_av_transmission_aer(e,Hgamma_SST,costheta,aod,H,AA)                        for e in bh.xi_e_sipm] )    
    tgammasst_mol = np.array( [get_av_transmission_mol(e,Hgamma_SST,costheta)                                 for e in bh.xi_e_sipm] )    

    tmulst_aer_nocam    = np.array( [get_av_transmission_phi_aer(e,R_LST,inner_R_LST,thetac,rhoR,costheta,aod,H,AA) for e in bh.xi_e_pmt_nocam] )
    tmulst_mol_nocam    = np.array( [get_av_transmission_phi_mol(e,R_LST,inner_R_LST,thetac,rhoR,costheta)          for e in bh.xi_e_pmt_nocam] )
    tgammalst_aer_nocam = np.array( [get_av_transmission_aer(e,Hgamma_LST,costheta,aod,H,AA)                        for e in bh.xi_e_pmt_nocam] )    
    tgammalst_mol_nocam = np.array( [get_av_transmission_mol(e,Hgamma_LST,costheta)                                 for e in bh.xi_e_pmt_nocam] )    

    tmumst_aer_nocam    = np.array( [get_av_transmission_phi_aer(e,R_MST,inner_R_MST,thetac,rhoR,costheta,aod,H,AA) for e in bh.xi_e_pmt_nocam] )    
    tmumst_mol_nocam    = np.array( [get_av_transmission_phi_mol(e,R_MST,inner_R_MST,thetac,rhoR,costheta)          for e in bh.xi_e_pmt_nocam] )    
    tgammamst_aer_nocam = np.array( [get_av_transmission_aer(e,Hgamma_MST,costheta,aod,H,AA)                        for e in bh.xi_e_pmt_nocam] )    
    tgammamst_mol_nocam = np.array( [get_av_transmission_mol(e,Hgamma_MST,costheta)                                 for e in bh.xi_e_pmt_nocam] )    

    tmusst_aer_nocam    = np.array( [get_av_transmission_phi_aer(e,R_SST,inner_R_SST,thetac,rhoR,costheta,aod,H,AA) for e in bh.xi_e_sipm_nocam] )    
    tmusst_mol_nocam    = np.array( [get_av_transmission_phi_mol(e,R_SST,inner_R_SST,thetac,rhoR,costheta)          for e in bh.xi_e_sipm_nocam] )    
    tgammasst_aer_nocam = np.array( [get_av_transmission_aer(e,Hgamma_SST,costheta,aod,H,AA)                        for e in bh.xi_e_sipm_nocam] )    
    tgammasst_mol_nocam = np.array( [get_av_transmission_mol(e,Hgamma_SST,costheta)                                 for e in bh.xi_e_sipm_nocam] )    

    tmulst_atm        = tmulst_mol          * tmulst_aer
    tmulst_atm_nocam  = tmulst_mol_nocam    * tmulst_aer_nocam
    tmulst_comb       = bh.xi_det_pmt       * tmulst_atm
    tmulst_comb_nocam = bh.xi_det_pmt_nocam * tmulst_atm_nocam
    tgammalst_atm       = trans_lst_int(bh.xi_wl_pmt)
    tgammalst_atm_nocam = trans_lst_int(bh.xi_wl_pmt_nocam)
#    print (tgammalst_atm)
#    print ('\n')
#    tgammalst_atm = tgammalst_mol * tgammalst_aer
#    print (tgammalst_atm)    
    tgammalst_comb       = bh.xi_det_pmt       * tgammalst_atm
    tgammalst_comb_nocam = bh.xi_det_pmt_nocam * tgammalst_atm_nocam

    
    tmumst_atm           = tmumst_mol          * tmumst_aer
    tmumst_atm_nocam     = tmumst_mol_nocam    * tmumst_aer_nocam
    tmumst_comb          = bh.xi_det_pmt       * tmumst_atm
    tmumst_comb_nocam    = bh.xi_det_pmt_nocam * tmumst_atm_nocam
    tgammamst_atm        = trans_mst_int(bh.xi_wl_pmt)             # tgammamst_mol * tgammamst_aer
    tgammamst_atm_nocam  = trans_mst_int(bh.xi_wl_pmt_nocam)             # tgammamst_mol * tgammamst_aer        
    tgammamst_comb       = bh.xi_det_pmt       * tgammamst_atm
    tgammamst_comb_nocam = bh.xi_det_pmt_nocam * tgammamst_atm_nocam

    tmusst_atm           = tmusst_mol           * tmusst_aer
    tmusst_atm_nocam     = tmusst_mol_nocam     * tmusst_aer_nocam
    tmusst_comb          = bh.xi_det_sipm       * tmusst_atm
    tmusst_comb_nocam    = bh.xi_det_sipm_nocam * tmusst_atm_nocam
    tgammasst_atm        = trans_sst_int(bh.xi_wl_sipm)            # tgammasst_mol  * tgammasst_aer
    tgammasst_atm_nocam  = trans_sst_int(bh.xi_wl_sipm_nocam)      # tgammasst_mol  * tgammasst_aer        
    tgammasst_comb       = bh.xi_det_sipm       * tgammasst_atm
    tgammasst_comb_nocam = bh.xi_det_sipm_nocam * tgammasst_atm_nocam

    min_pmt        = bh.get_low_idx(bh.xi_det_pmt, 2e-2)
    min_sipm       = bh.get_low_idx(bh.xi_det_sipm,2e-2)   
    min_pmt_nocam  = bh.get_low_idx(bh.xi_det_pmt_nocam, 2e-2)
    min_sipm_nocam = bh.get_low_idx(bh.xi_det_sipm_nocam,2e-2)   
    max_pmt        = bh.get_high_idx(bh.xi_det_pmt, 2e-2)
    max_sipm       = bh.get_high_idx(bh.xi_det_sipm,2e-2)   
#    max_pmt_nocam  = bh.get_high_idx(bh.xi_det_pmt_nocam, 2e-2)
    max_pmt_nocam  = bh.get_high_idx(tgammalst_comb_nocam,5e-4)
    max_sipm_nocam = bh.get_high_idx(tgammasst_comb_nocam,5e-4)
    
    print ('min PMT: ', min_pmt, ' at: ',bh.xi_wl_pmt[min_pmt], 'min PMT (nocam): ', min_pmt_nocam, ' at: ',bh.xi_wl_pmt_nocam[min_pmt_nocam])
    print ('max PMT: ', max_pmt, ' at: ',bh.xi_wl_pmt[max_pmt], 'max PMT (nocam): ', max_pmt_nocam, ' at: ',bh.xi_wl_pmt_nocam[max_pmt_nocam])

    tmulst_tot       = (np.cumsum(tmulst_atm)[max_pmt] -np.cumsum(tmulst_atm)[min_pmt] ) *bh.xi_steps
    tmumst_tot       = (np.cumsum(tmumst_atm)[max_pmt] -np.cumsum(tmumst_atm)[min_pmt] ) *bh.xi_steps
    tmusst_tot       = (np.cumsum(tmusst_atm)[max_sipm]-np.cumsum(tmusst_atm)[min_sipm]) *bh.xi_steps

    tmulst_tot_nocam = ( np.cumsum(tmulst_atm_nocam)[max_pmt_nocam] -np.cumsum(tmulst_atm_nocam)[min_pmt_nocam] ) *bh.xi_steps
    tmumst_tot_nocam = ( np.cumsum(tmumst_atm_nocam)[max_pmt_nocam] -np.cumsum(tmumst_atm_nocam)[min_pmt_nocam] ) *bh.xi_steps
    tmusst_tot_nocam = ( np.cumsum(tmusst_atm_nocam)[max_sipm_nocam]-np.cumsum(tmusst_atm_nocam)[min_sipm_nocam]) *bh.xi_steps

    print ('tmulst tot: ', tmulst_tot, ' no cam: ',tmulst_tot_nocam)

#    tgammalst_tot  = np.cumsum(tgammalst_atm)[max_pmt] -np.cumsum(tgammalst_atm)[min_pmt] 
#    tgammamst_tot  = np.cumsum(tgammamst_atm)[max_pmt] -np.cumsum(tgammamst_atm)[min_pmt] 
#    tgammasst_tot  = np.cumsum(tgammasst_atm)[max_sipm]-np.cumsum(tgammasst_atm)[min_sipm]

    tgammalst_tot  = (np.cumsum(tgammalst_atm)[max_pmt] -np.cumsum(tgammalst_atm)[min_pmt] ) *bh.xi_steps
    tgammamst_tot  = (np.cumsum(tgammamst_atm)[max_pmt] -np.cumsum(tgammamst_atm)[min_pmt] ) *bh.xi_steps
    tgammasst_tot  = (np.cumsum(tgammasst_atm)[max_sipm]-np.cumsum(tgammasst_atm)[min_sipm]) *bh.xi_steps

    tgammalst_tot_nocam  = (np.cumsum(tgammalst_atm_nocam)[max_pmt_nocam] -np.cumsum(tgammalst_atm_nocam)[min_pmt_nocam] ) *bh.xi_steps
    tgammamst_tot_nocam  = (np.cumsum(tgammamst_atm_nocam)[max_pmt_nocam] -np.cumsum(tgammamst_atm_nocam)[min_pmt_nocam] ) *bh.xi_steps
    tgammasst_tot_nocam  = (np.cumsum(tgammasst_atm_nocam)[max_sipm_nocam]-np.cumsum(tgammasst_atm_nocam)[min_sipm_nocam]) *bh.xi_steps

    print ('tgammalst tot: ',    tgammalst_tot,                     ' no cam: ',tgammalst_tot_nocam)
    print ('tgammalst(min PMT)', tgammalst_atm[min_pmt],            ' no cam: ',tgammalst_atm_nocam[min_pmt_nocam])
    print ('tgammalst(max PMT)', tgammalst_atm[max_pmt],            ' no cam: ',tgammalst_atm_nocam[max_pmt_nocam])
    print ('tgammalst(min PMT)', np.cumsum(tgammalst_atm)[min_pmt], ' no cam: ',np.cumsum(tgammalst_atm_nocam)[min_pmt_nocam])
    print ('tgammalst(max PMT)', np.cumsum(tgammalst_atm)[max_pmt], ' no cam: ',np.cumsum(tgammalst_atm_nocam)[max_pmt_nocam])

    print (np.cumsum(bh.xi_det_pmt_nocam)*bh.xi_steps)
    print (np.cumsum(bh.xi_det_pmt)*bh.xi_steps)
    print (np.cumsum(tgammalst_atm_nocam)*bh.xi_steps)
    print (np.cumsum(tgammalst_atm)*bh.xi_steps)
    print (np.cumsum(tgammalst_comb_nocam)*bh.xi_steps)
    print (np.cumsum(tgammalst_comb)*bh.xi_steps)
    print (np.cumsum(tmulst_comb_nocam)*bh.xi_steps)
    print (np.cumsum(tmulst_comb)*bh.xi_steps)
    
    
#    print (tgammalst_tot, tgammamst_tot, tgammasst_tot)
    
#    ax.plot(bh.xi_e_pmt, np.cumsum(tgammalst_comb)/np.cumsum(tmulst_comb)/tgammalst_tot*tmulst_tot, '--',label=r'LST (h=10 km)')
#    ax.plot(bh.xi_e_pmt, np.cumsum(tgammamst_comb)/np.cumsum(tmumst_comb)/tgammamst_tot*tmumst_tot, '--',label=r'MST (h=8 km)')
#    ax.plot(bh.xi_e_sipm,np.cumsum(tgammasst_comb)/np.cumsum(tmusst_comb)/tgammasst_tot*tmusst_tot, '--',label=r'SST (h=6.5 km)')

    ax.plot(bh.xi_e_pmt,       (np.cumsum(tgammalst_comb)      *bh.xi_steps - np.cumsum(tmulst_comb)*bh.xi_steps      *tgammalst_tot       /tmulst_tot)     /np.cumsum(tgammalst_comb)/bh.xi_steps,       'b-',label=r'LST (with window)',lw=1)
    ax.plot(bh.xi_e_pmt_nocam, (np.cumsum(tgammalst_comb_nocam)*bh.xi_steps - np.cumsum(tmulst_comb_nocam)*bh.xi_steps*tgammalst_tot_nocam/tmulst_tot_nocam)/np.cumsum(tgammalst_comb_nocam)/bh.xi_steps, '--',color='steelblue',label=r'LST (no window)')    
    ax.plot(bh.xi_e_pmt,       (np.cumsum(tgammamst_comb)      *bh.xi_steps - np.cumsum(tmumst_comb)*bh.xi_steps      *tgammamst_tot       /tmumst_tot)     /np.cumsum(tgammamst_comb)/bh.xi_steps,       'g-',label=r'MST (with window)',lw=1)
    ax.plot(bh.xi_e_pmt_nocam, (np.cumsum(tgammamst_comb_nocam)*bh.xi_steps - np.cumsum(tmumst_comb_nocam)*bh.xi_steps*tgammamst_tot_nocam/tmumst_tot_nocam)/np.cumsum(tgammamst_comb_nocam)/bh.xi_steps, '--',color='darkseagreen',label=r'MST (no window)')
    ax.plot(bh.xi_e_sipm,      (np.cumsum(tgammasst_comb)      *bh.xi_steps - np.cumsum(tmusst_comb)*bh.xi_steps      *tgammasst_tot       /tmusst_tot)     /np.cumsum(tgammasst_comb)/bh.xi_steps,       'r-',label=r'SST',lw=1)
#    ax.plot(bh.xi_e_sipm_nocam,(np.cumsum(tgammasst_comb_nocam)*bh.xi_steps - np.cumsum(tmusst_comb_nocam)*bh.xi_steps*tgammasst_tot_nocam/tmusst_tot_nocam)/np.cumsum(tgammasst_comb_nocam)/bh.xi_steps, 'r-.',label=r'SST (no window)')

    ax.legend(bbox_to_anchor=(0.99, 0.99),loc=1)

    ax.set_xlabel('Start of sudden detector blindness $\epsilon_{blind}$ (eV)')
    ax.set_ylabel(r'$\Delta B_{\gamma} / B_{\gamma}$')
#    ax.set_ylim([0.,0.165])
    ax.set_ylim([0.,0.25])

    ax_c = ax.twiny()
    wls  = np.array([600,500,400,350,300,250,200])
    #wls  = np.array([800,700,600,500,400,300,250,200])
    wlticks = [bh.nm2ev(wl) for wl in wls]
    ax_c.set_xticks(wlticks);
    ax_c.set_xticklabels(['{:g}'.format(wl) for wl in wls]);
    ax_c.set_xlabel('Wavelength (nm)')
    
    xmin, xmax = 2.0, 5.25
    ax.set_xlim(xmin, xmax)
    ax_c.set_xlim(xmin, xmax);
    
    plt.savefig('output/Bandwidth_Ratio.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()


if (plot_pmt):

    fig_size[1] = fig_size[1]*2

    #    aod = 0.1
    #    H   = 1300
    #    AA  = 2.5
    
    #    es_corr = bh.nm2ev(atm_tab['wl'])
    #    tgammalst_aer_corr = np.array( [get_av_transmission_aer(e,Hgamma_LST,costheta,aod,H,AA)                   for e in es_corr] )    
    #    tgammasst_aer_corr = np.array( [get_av_transmission_aer(e,Hgamma_SST,costheta,aod,H,AA)                   for e in es_corr] )
    
    aod = 0.03
    H   = 600
    AA  = 1.2

    #    tgammalst_aer_corr2 = np.array( [get_av_transmission_aer(e,Hgamma_LST,costheta,aod,H,AA)                  for e in es_corr] )
    #    tgammasst_aer_corr2 = np.array( [get_av_transmission_aer(e,Hgamma_SST,costheta,aod,H,AA)                  for e in es_corr] )    
    
    es = np.arange(1.2,5.5,0.1)
    
    tmulst_aer    = np.array( [get_av_transmission_phi_aer(e,R_LST,inner_R_LST,thetac,rhoR,costheta,aod,H,AA) for e in es] )
    tmulst_mol    = np.array( [get_av_transmission_phi_mol(e,R_LST,inner_R_LST,thetac,rhoR,costheta)          for e in es] )
    tmulst_atm    = tmulst_mol    * tmulst_aer

    tmusst_aer    = np.array( [get_av_transmission_phi_aer(e,R_SST,inner_R_SST,thetac,rhoR,costheta,aod,H,AA) for e in es] )    
    tmusst_mol    = np.array( [get_av_transmission_phi_mol(e,R_SST,inner_R_SST,thetac,rhoR,costheta)          for e in es] )    
    tmusst_atm     = tmusst_mol   * tmusst_aer
    
#    tgammalst_aer = np.array( [get_av_transmission_aer(e,Hgamma_LST,costheta,aod,H,AA)                        for e in es] )    
#    tgammalst_mol = np.array( [get_av_transmission_mol(e,Hgamma_LST,costheta)                                 for e in es] )    
#    tgammalst_atm  = tgammalst_mol * tgammalst_aer
    
#    tgammasst_aer = np.array( [get_av_transmission_aer(e,Hgamma_SST,costheta,aod,H,AA)                        for e in es] )    
#    tgammasst_mol = np.array( [get_av_transmission_mol(e,Hgamma_SST,costheta)                                 for e in es] )    
#    tgammasst_atm = tgammasst_mol  * tgammasst_aer

    trans_lst = get_trans_from_trans_file(Hgamma_LST)
    trans_sst = get_trans_from_trans_file(Hgamma_SST)
#    ods_lst = np.array( atm_tab['10.000']) - np.array( atm_tab['2.197'])  + np.log(tgammalst_aer_corr) - np.log(tgammalst_aer_corr2)
#    ods_sst = np.array( atm_tab['6.000'])  - np.array( atm_tab['2.197'])  + np.log(tgammasst_aer_corr) - np.log(tgammasst_aer_corr2)

    f, ax = plt.subplots(2,1)
    ax[1].plot(bh.qe_e,bh.qe_ham, '--', color='darkmagenta',label=r'$\xi_{pde}$ (PMT$_{1}$)')
    ax[1].plot(bh.ete_e,bh.qe_ete, '--', color='teal', label=r'$\xi_{pde}$ (PMT$_{2}$)')
    ax[1].plot(bh.si_e,bh.qe_si,  '-', color='green',label=r'$\xi_{pde}$ (SiPM)')
    # empty fake line for the legend
    #    ax.plot(np.NaN, np.NaN, '-', color='none', label='')    
    x = [3.1,3.101]
    y = [0.,0.1]
    # ax[1].plot(x,y, '--', color='w',label='.')
    ax[1].legend(bbox_to_anchor=(0.8, 0.92),loc=2)#,ncol=2)

    ax[0].plot(es,tmulst_atm,  '-', color='b',  label=r'$t_{\mu}$ (LST)')
    ax[0].plot(bh.nm2ev(np.array( atm_tab['wl'] )), trans_lst,'--',color='cornflowerblue',label=r'$t_{\gamma}$ (LST)')
    ax[0].plot(es,tmusst_atm,  '-', color='r', label=r'$t_{\mu}$ (SST)')
#    ax.plot(es,tgammalst_atm,'-', label=r'$t_{\gamma}$ (LST, $h_{\gamma}$=10 km)')
#    ax.plot(es,tgammasst_atm,'-', label=r'$t_{\gamma}$ (SST, $h_{\gamma}$=6 km)')
    ax[0].plot(bh.nm2ev(np.array( atm_tab['wl'] )), trans_sst,'--',color='darkorange',label=r'$t_{\gamma}$ (SST)')
    ax[0].legend(bbox_to_anchor=(0.8, 0.82),loc=2)#,ncol=2)

    ax[1].set_xlabel('Photon energy $\epsilon$ (eV)')
    ax[0].set_ylabel(r'atmospheric transparency')
    ax[1].set_ylabel(r'photon detection efficiency')

    ax[0].set_ylim([0.,1.05])
    ax[1].set_ylim([0.,0.5])

    ax_c = ax[0].twiny()
    wls  = np.array([700,600,500,400,300,250])
    wlticks = [bh.nm2ev(wl) for wl in wls]
    ax_c.set_xticks(wlticks);
    ax_c.set_xticklabels(['{:g}'.format(wl) for wl in wls]);
    ax_c.set_xlabel('Photon wavelength (nm)')
    
    xmin, xmax = 1.3, 5.8
    ax[0].set_xlim(xmin, xmax)
    ax[1].set_xlim(xmin, xmax)
    ax_c.set_xlim(xmin, xmax);
    
    plt.savefig('output/Bandwidth_PMT.pdf',bbox_inches='tight')
    plt.show()
    plt.close()

'''
