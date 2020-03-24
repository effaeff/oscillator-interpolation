import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import ExponentialModel, GaussianModel


x_range = (200, 3200, 1000)
xx_frf = np.loadtxt('../data/01_raw/128_Heller_XX_A_B-120_0__0__Signal 2_Signal 1.txt', delimiter=' ')
xx_frf = xx_frf[(np.where((xx_frf[:, 0] > x_range[0]) & (xx_frf[:, 0] < x_range[1])))]


x = xx_frf[:, 0]
y = xx_frf[:, 1]

#exp_mod = ExponentialModel(prefix='exp_')
#pars = exp_mod.guess(y, x=x)

gauss1 = GaussianModel(prefix='g1_')
#pars.update(gauss1.make_params())
pars = gauss1.make_params()

pars['g1_center'].set(value=1076)
pars['g1_sigma'].set(value=719)
pars['g1_amplitude'].set(value=410)

gauss2 = GaussianModel(prefix='g2_')
pars.update(gauss2.make_params())

pars['g2_center'].set(value=1509)
pars['g2_sigma'].set(value=353)
pars['g2_amplitude'].set(value=240)

gauss3 = GaussianModel(prefix='g3_')
pars.update(gauss3.make_params())

pars['g3_center'].set(value=2297)
pars['g3_sigma'].set(value=1219)
pars['g3_amplitude'].set(value=170)

#gauss4 = GaussianModel(prefix='g4_')
#pars.update(gauss4.make_params())

# pars['g4_center'].set(value=400)
# pars['g4_sigma'].set(value=100)
# pars['g4_amplitude'].set(value=80)

mod = gauss1 + gauss2 + gauss3 #+gauss4#+ exp_mod

init = mod.eval(pars, x=x)
out = mod.fit(y, pars, x=x)

print(out.fit_report(min_correl=0.5))

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
axes[0].plot(x, y, 'b')
axes[0].plot(x, init, 'k--', label='initial fit')
axes[0].plot(x, out.best_fit, 'r-', label='best fit')
axes[0].legend(loc='best')

comps = out.eval_components(x=x)
axes[1].plot(x, y, 'b')
axes[1].plot(x, comps['g1_'], 'g--', label='Gaussian component 1')
axes[1].plot(x, comps['g2_'], 'm--', label='Gaussian component 2')
axes[1].plot(x, comps['g3_'], 'r--', label='Gaussian component 3')
#axes[1].plot(x, comps['exp_'], 'k--', label='Exponential component')
axes[1].legend(loc='best')

plt.show()

osci1 = [out.best_values['g1_center'], out.best_values['g1_amplitude'], out.best_values['g1_sigma']]
osci2 = [out.best_values['g2_center'], out.best_values['g2_amplitude'], out.best_values['g2_sigma']]
osci3 = [out.best_values['g3_center'], out.best_values['g3_amplitude'], out.best_values['g3_sigma']]
#osci4 = [out.best_values['g4_center'], out.best_values['g4_amplitude'], out.best_values['g4_sigma']]


