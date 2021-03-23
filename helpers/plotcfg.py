import numpy as np
import matplotlib as mpl
from matplotlib import ticker,pyplot
from matplotlib.colors import LinearSegmentedColormap

def figsize(scale):
    fig_width_pt = 455.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "axes.labelsize": 14,               # LaTeX default is 10pt font.
    "font.size": 14,
    "legend.fontsize": 14,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.figsize"   : (10, 6)    # figure size in inches
    }
mpl.rcParams.update(pgf_with_latex)

tableau = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),
             (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236),
(255, 188, 121), (207, 207, 207)]
for i in range(len(tableau)):  
    r, g, b = tableau[i]  
    tableau[i] = (r / 255., g / 255., b / 255.)

def center_spines(ax=None, centerx=0, centery=0):
    """Centers the axis spines at <centerx, centery> on the axis "ax", and
    places arrows at the end of the axis spines."""
    if ax is None:
        ax = pyplot.gca()

    # Set the axis's spines to be centered at the given point
    # (Setting all 4 spines so that the tick marks go in both directions)
    ax.spines['left'].set_position(('data', centerx))
    ax.spines['bottom'].set_position(('data', centery))
    ax.spines['right'].set_position(('data', centerx - 1))
    ax.spines['top'].set_position(('data', centery - 1))

      

    # Hide the line (but not ticks) for "extra" spines
    for side in ['right', 'top']:
        ax.spines[side].set_color('none')

    # On both the x and y axes...
    for axis, center in zip([ax.xaxis, ax.yaxis], [centerx, centery]):
        # Turn on minor and major gridlines and ticks
        #axis.set_ticks_position('both')
#        axis.grid(True, 'major', ls='solid', lw=0.5, color='gray')
#        axis.grid(True, 'minor', ls='solid', lw=0.1, color='gray')
        #axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        # Hide the ticklabels at <centerx, centery>
        formatter = CenteredFormatter()
        formatter.center = center
        axis.set_major_formatter(formatter)

    # Add offset ticklabels at <centerx, centery> using annotation
    # (Should probably make these update when the plot is redrawn...)
    xlabel, ylabel = map(formatter.format_data, [centerx, centery])
    

# Note: I'm implementing the arrows as a path effect rather than a custom 
#       Spines class. In the long run, a custom Spines class would be a better
#       way to go. One of the side effects of this is that the arrows aren't
#       reversed when the axes are reversed!


class CenteredFormatter(ticker.ScalarFormatter):
    """Acts exactly like the default Scalar Formatter, but yields an empty
    label for ticks at "center"."""
    center = 0
    def __call__(self, value, pos=None):
        if value == self.center:
            return ''
        else:
            return ticker.ScalarFormatter.__call__(self, value, pos)


def hide_spines():
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""

    # Retrieve a list of all current figures.
    figures = [x for x in mpl._pylab_helpers.Gcf.get_all_fig_managers()]
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            #ax.spines['left'].set_color('none')
            #ax.spines['bottom'].set_position('center')
            #ax.spines['left'].set_position('left')
            # Disable ticks.
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            

pyplot.register_cmap(cmap=LinearSegmentedColormap(name='tableau', segmentdata=tableau))
