plt.rcParams['figure.figsize'] = [6,6]
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight']= 'normal'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits']=(-6, 6)
mpl.rcParams['axes.formatter.use_mathtext']=True
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True ;

_base = float(mpl.rcParams.get("font.size", 12.0))
if _base > 0:
    mpl.rcParams["axes.titlesize"] = _base * 0.9
    mpl.rcParams["axes.labelsize"] = _base * 0.8
    mpl.rcParams["xtick.labelsize"] = _base * 0.7
    mpl.rcParams["ytick.labelsize"] = _base * 0.7
    mpl.rcParams["legend.fontsize"] = _base * 0.6
    mpl.rcParams["figure.titlesize"] = _base * 0.9