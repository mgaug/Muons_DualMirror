import matplotlib
import matplotlib
from numpy import sqrt

twocolumn = False
#print "HELLO\n"
if twocolumn == True:
    fig_width_pt = 255.76535
    fontsize     = 15
else:
    fig_width_pt = 426.79134
    fontsize     = 10

#fontsize = 20

inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5.)+1.0)/2.0        # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width/golden_mean      # height in inches
fig_size =  [fig_width*1.35,fig_height*1.35] # make figures a bit larger than default

    
params = {'backend': 'pdf',
          'axes.labelsize': fontsize+2,
          'font.size': fontsize,
          'legend.fontsize': fontsize+1,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'axes.linewidth': 0.5,
          'lines.linewidth': 0.7,
          'text.usetex': False,
          'ps.usedistiller': False,
          'figure.figsize': fig_size,
          'font.family': 'Arial',
#          'axes.labelpad' : 2, 
          'font.serif': ['Bitstream Vera Serif'],
#          'image.cmap'  : 'Parula'
  }
matplotlib.rcParams.update(params)

#matplotlib.style.use('v2.0')

# colors suitable for deuteranopia
CB_colors = ['#377eb8', '#ff7f00', '#4daf4a',
             '#f781bf', '#a65628', '#984ea3',
             '#999999', '#e41a1c', '#dede00']

