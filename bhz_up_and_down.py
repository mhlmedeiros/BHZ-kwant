import kwant

import tinyarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 22}

mpl.rc('font', **font)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sigma_0 = tinyarray.array([[1,0],[0,1]])
sigma_x = tinyarray.array([[0,1],[1,0]])
sigma_y = tinyarray.array([[0,-1j],[1j,0]])
sigma_z = tinyarray.array([[1,0],[0,-1]])


def qsh_system(a=2, L=500, W=200):
    hamiltonian_syst = """
       + C(x,y) * identity(4) + M * kron(sigma_0, sigma_z)
       - B * (k_x**2 + k_y**2) * kron(sigma_0, sigma_z)
       - D * (k_x**2 + k_y**2) * kron(sigma_0, sigma_0)
       + A * k_x * kron(sigma_z, sigma_x)
       - A * k_y * kron(sigma_0, sigma_y)
    """

    hamiltonian_lead = """
       + C_const * identity(4) + M * kron(sigma_0, sigma_z)
       - B * (k_x**2 + k_y**2) * kron(sigma_0, sigma_z)
       - D * (k_x**2 + k_y**2) * kron(sigma_0, sigma_0)
       + A * k_x * kron(sigma_z, sigma_x)
       - A * k_y * kron(sigma_0, sigma_y)
    """

    template_syst = kwant.continuum.discretize(hamiltonian_syst, grid_spacing=a)
    template_lead = kwant.continuum.discretize(hamiltonian_lead, grid_spacing=a)
    print(template_lead)


    # def shape(site):
    #     (x, y) = site.pos
    #     return (-W/2 < y < W/2 and -L/2 < x < L/2)
    #
    # def lead_shape(site):
    #     (x, y) = site.pos
    #     return (-W/2 < y < W/2)
    #
    # syst = kwant.Builder()
    # syst.fill(template_syst, shape, (0, 0))
    #
    # lead = kwant.Builder(kwant.TranslationalSymmetry([-a, 0]))
    # lead.fill(template_lead, lead_shape, (0, 0))
    #
    # syst.attach_lead(lead)
    # syst.attach_lead(lead.reversed())
    #
    # syst = syst.finalized()
    # return syst


qsh_system()




def band_structure(pot=0,shift=0):

    def potential_barrier(x,y):
        if abs(x) < L_barrier/2 :
            return -(pot-shift)
        else:
            return 0

    params = dict(A=364.5, B=-686.0, D=-512.0,
        M=-10.0, C=potential_barrier,C_const=-shift)

    syst = qsh_system()

    kx_max = 0.25
    kx_min = -kx_max
    kwant.plotter.bands(syst.leads[0], params=params,
                        momenta=np.linspace(kx_min, kx_max, 201),
                        fig_size=(6,6), show=False)

    plt.grid()
    plt.xlim(kx_min, kx_max)
    plt.ylim(-40, 40)
    plt.xlabel(r'momentum [nm$^{-1}$]')
    plt.ylabel(r'Energy [meV]')
    plt.tight_layout()
    plt.show()

band_structure()


def analyze_bhz(pot=0, L_barrier=100, shift=0, lead_index=0):

    def potential_barrier(x,y):
        if abs(x) < L_barrier/2 :
            return -(pot-shift)
        else:
            return 0

    params = dict(A=364.5, B=-686.0, D=-512.0,
        M=-10.0, C=potential_barrier,C_const=-shift)

    syst = qsh_system()


    # get scattering wave functions at E=0
    wf = kwant.wave_function(syst, energy=-10, params=params)

    # prepare density operators
    # sigma_z = np.array([[1, 0], [0, -1]])
    prob_density = kwant.operator.Density(syst, np.kron(sigma_z, np.eye(2)))
    # J_0 = kwant.operator.Current(syst)

    # calculate expectation values and plot them
    wf_sqr = sum(prob_density(psi) for psi in wf(lead_index))
    print(max(wf_sqr))
    # current = sum(J_0(psi,params=params) for psi in wf(lead_index))
    fig, (ax1) = plt.subplots(1, 1, figsize=(16, 4))
    ax1 = plt.gca()
    ax1.set_title(r'$\Psi_{\uparrow}$')
    ax1.set_xlabel(r'$x$ [nm]')
    ax1.set_ylabel(r'$y$ [nm]')
    kwant.plotter.map(syst, (1/max(wf_sqr)) * wf_sqr,fig_size=(9,2),ax=ax1,cmap='seismic')
    ax = ax1
    im = [obj for obj in ax.get_children()
          if isinstance(obj, mpl.image.AxesImage)][0]
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    # ax2 = plt.gca()
    # ax2.set_title(r'$J_{\uparrow}$')
    # ax2.set_xlabel(r'$x$ [nm]')
    # ax2.set_ylabel(r'$y$ [nm]')
    # kwant.plotter.current(syst, current,colorbar=False,fig_size=(14,2),ax=ax2)
    # plt.tight_layout()
    # plt.show()

analyze_bhz()


def plot_conductance(syst, energies,L_barrier=100, pot=0, shift=0):

    def potential_barrier(x,y):
        if abs(x) < L_barrier/2 :
            return -(pot-shift)
        else:
            return 0

    params = dict(A=364.5, B=-686.0, D=-512.0,
        M=-10.0, C=potential_barrier,C_const=-shift)


    # Compute conductance
    data = []
    for energy in energies:
        smatrix = kwant.smatrix(syst, energy,params=params)
        data.append(smatrix.transmission(1,0))

    plt.figure()
    plt.plot(energies,data,linestyle='',marker='o',color='red')
    plt.xlabel(r"E$_F$ [meV]")
    plt.ylabel(r"conductance $[e^2/h]$")
    plt.show()

syst = qsh_system()
plot_conductance(syst, energies=np.linspace(-40,40,100))
