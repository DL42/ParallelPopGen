import time

import numpy
from numpy import array
import dadi

def prior_onegrow_mig((nu1F, nu2B, nu2F, m, Tp, T), (n1,n2), pts):
    """
    Model with growth, split, bottleneck in pop2, exp recovery, migration

    nu1F: The ancestral population size after growth. (Its initial size is
          defined to be 1.)
    nu2B: The bottleneck size for pop2
    nu2F: The final size for pop2
    m: The scaled migration rate
    Tp: The scaled time between ancestral population growth and the split.
    T: The time between the split and present

    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = yy = dadi.Numerics.default_grid(pts)
    gamma_model = -2;
    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx, gamma=gamma_model)
    # Now do the population growth event.
    phi = dadi.Integration.one_pop(phi, xx, Tp, nu=nu1F, gamma=gamma_model)

    # The divergence
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We need to define a function to describe the non-constant population 2
    # size. lambda is a convenient way to do so.
    nu2_func = lambda t: nu2B*(nu2F/nu2B)**(t/T)
    phi = dadi.Integration.two_pops(phi, xx, T, nu1=nu1F, nu2=nu2_func, m12=m, m21=m, gamma1=gamma_model, gamma2=gamma_model)

    # Finally, calculate the spectrum.
    sfs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,yy))
    return sfs

def runModel(nu1F, nu2B, nu2F, m, Tp, T):

    ns = (0,1001)
    print 'sample sizes:', ns

    # These are the grid point settings will use for extrapolation.
    pts_l = [110,120,130]

    func = prior_onegrow_mig
    params = array([nu1F, nu2B, nu2F, m, Tp, T])

    # Make the extrapolating version of the demographic model function.
    func_ex = dadi.Numerics.make_extrap_func(func)

    # Calculate the model AFS.
    model = func_ex(params, ns, pts_l)
    model = model.marginalize((0,))
    return model
	

time1 = time.time()

model = runModel(2, 0.05, 5, 1, 0.005, 0.045)

time2 = time.time()
print "total runtime (s): " + str(time2-time1)
model = model/model.S()

for i in range(0,len(model)):
	print model[i]