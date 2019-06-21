import kwant
import kwant.continuum
import tinyarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sigma_0 = tinyarray.array([[1,0],[0,1]])
sigma_x = tinyarray.array([[0,1],[1,0]])
sigma_y = tinyarray.array([[0,-1j],[1j,0]])
sigma_z = tinyarray.array([[1,0],[0,-1]])




def bhz_up(a=2, L=1000, W=200):
    hamiltonian_syst = """
       + (C(x,y) + C_const) * identity(2) + M * sigma_z
       - B * (k_x**2 + k_y**2) *  sigma_z
       - D * (k_x**2 + k_y**2) *  sigma_0
       + A * k_x * sigma_x
       - A * k_y * sigma_y
    """

    hamiltonian_lead = """
       + C_const * identity(2) + M * sigma_z
       - B * (k_x**2 + k_y**2) * sigma_z
       - D * (k_x**2 + k_y**2) * sigma_0
       + A * k_x * sigma_x
       - A * k_y * sigma_y
       """

    template_syst = kwant.continuum.discretize(hamiltonian_syst, grid=a)
    template_lead = kwant.continuum.discretize(hamiltonian_lead, grid=a)


    def shape(site):
        (x, y) = site.pos
        return (-W/2 < y < W/2 and -L/2 < x < L/2)

    def lead_shape(site):
        (x, y) = site.pos
        return (-W/2 < y < W/2)

    syst = kwant.Builder()
    syst.fill(template_syst, shape, (0, 0))

    lead = kwant.Builder(kwant.TranslationalSymmetry([-a, 0]))
    lead.fill(template_lead, lead_shape, (0, 0))

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    syst = syst.finalized()
    return syst

def bhz_down(a=2, L=1000, W=200):
    hamiltonian_syst = """
       + (C(x,y) + C_const) * identity(2) + M * sigma_z
       - B * (k_x**2 + k_y**2) *  sigma_z
       - D * (k_x**2 + k_y**2) *  sigma_0
       - A * k_x * sigma_x
       - A * k_y * sigma_y
    """

    hamiltonian_lead = """
       + C_const * identity(2) + M * sigma_z
       - B * (k_x**2 + k_y**2) * sigma_z
       - D * (k_x**2 + k_y**2) * sigma_0
       - A * k_x * sigma_x
       - A * k_y * sigma_y
       """

    template_syst = kwant.continuum.discretize(hamiltonian_syst, grid=a)
    template_lead = kwant.continuum.discretize(hamiltonian_lead, grid=a)


    def shape(site):
        (x, y) = site.pos
        return (-W/2 < y < W/2 and -L/2 < x < L/2)

    def lead_shape(site):
        (x, y) = site.pos
        return (-W/2 < y < W/2)

    syst = kwant.Builder()
    syst.fill(template_syst, shape, (0, 0))

    lead = kwant.Builder(kwant.TranslationalSymmetry([-a, 0]))
    lead.fill(template_lead, lead_shape, (0, 0))

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    syst = syst.finalized()
    return syst


def analyze_bhz_up(pot=0, L_barrier=500, shift=0, lead_index=0):

    '''
    Only for "up" component:

    This function returns a mapping of absolute squared wave function over the
    system shape, and, likewise, the current density mapping.

    lead_index = lead identifier: 0 for left lead and 1 for right lead
    L_barrier = length of potential scatterer (a barrier or a well)
    pot =  potential level for leads (only appear in the definition of the dictionary "params")
    shift = difference between scatterer potential level and leads potential (only at "potential_barrier")

    If shift is negative the scatterer will be a well, otherwise it'll be a barrier.
    Note that the signs of "pot" and "shift" are both changed bellow, this is because
    we want to think in terms of a tunable Fermi level, but what the code actually
    do is to move all the band structure with a fixed Fermi level at E = 0.

    '''

    def potential_barrier(x,y):
        """
        This function defines the potential scatterer in center of the system.
        It is passed as the C-parameter in the dictionary 'params'.
        x, y = coordinates in the plane of system

        """
        if abs(x) < L_barrier/2 :
            """
            The potential has the width of the system and is y-independent
            The shape of the potential depend on the sign of the value "shift";
            if shift > 0 we have a barrier, otherwise we get a well.
            """
            return pot-shift
        else:
            return 0


    params = dict(A=364.5, B=-686.0, D=-512.0,
        M=-10.0, C=potential_barrier,C_const=-pot)
    syst = bhz_up()
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


    # get scattering wave functions at E=0
    wf = kwant.wave_function(syst, energy=0, params=params)

    # prepare density operators
    sigma_z = np.array([[1, 0], [0, -1]])
    prob_density = kwant.operator.Density(syst, np.eye(2))
    J_0 = kwant.operator.Current(syst)

    # calculate expectation values and plot them
    wf_sqr = sum(prob_density(psi) for psi in wf(lead_index))
    current = sum(J_0(psi,params=params) for psi in wf(lead_index))

    ax1 = plt.gca()
    ax1.set_title(r'$\Psi_{\uparrow}$')
    ax1.set_xlabel(r'$x$ [nm]')
    ax1.set_ylabel(r'$y$ [nm]')
    kwant.plotter.map(syst, wf_sqr,colorbar=False,fig_size=(9,2),ax=ax1)
    plt.tight_layout()
    plt.show()

    ax2 = plt.gca()
    ax2.set_title(r'$J_{\uparrow}$')
    ax2.set_xlabel(r'$x$ [nm]')
    ax2.set_ylabel(r'$y$ [nm]')
    kwant.plotter.current(syst, current,colorbar=False,fig_size=(14,2),ax=ax2)
    plt.tight_layout()
    plt.show()

def analyze_bhz_down(pot=0, L_barrier=500, shift=0, lead_index=0):

    '''
    Only for "down" component:

    This function returns a mapping of absolute squared wave function over the
    system shape, and, likewise, the current density mapping.

    lead_index = lead identifier: 0 for left lead and 1 for right lead
    L_barrier = length of potential scatterer (a barrier or a well)
    pot =  potential level for leads (only appear in the definition of the dictionary "params")
    shift = difference between scatterer potential level and leads potential (only at "potential_barrier")

    If shift is negative the scatterer will be a well, otherwise it'll be a barrier.
    Note that the signs of "pot" and "shift" are both changed bellow, this is because
    we want to think in terms of a tunable Fermi level, but what the code actually
    do is to move all the band structure with a fixed Fermi level at E = 0.

    '''

    def potential_barrier(x,y):
        """
        This function defines the potential scatterer in center of the system.
        It is passed as the C-parameter in the dictionary 'params'.
        x, y = coordinates in the plane of the system

        """
        if abs(x) < L_barrier/2 :
            """
            The potential has the same width of the system and is y-independent
            The shape of the potential depend on the sign of the value "shift";
            if shift > 0 we have a barrier, otherwise we get a well.
            """
            return pot-shift
        else:
            return 0

    # params = dict(A=3.65, B=-68.6, D=-51.1,
    #     M=-0.01, C=potential_barrier,C_const=-shift)
    params = dict(A=364.5, B=-686.0, D=-512.0,
        M=-10.0, C=potential_barrier,C_const=-pot)
    syst = bhz_down()
    # kx_max = 0.25
    # kx_min = -kx_max
    # kwant.plotter.bands(syst.leads[0], params=params,
    #                     momenta=np.linspace(kx_min, kx_max, 201),
    #                     fig_size=(6,6), show=False)
    #
    # plt.grid()
    # plt.xlim(kx_min, kx_max)
    # plt.ylim(-40, 40)
    # plt.xlabel(r'momentum [nm$^{-1}$]')
    # plt.ylabel(r'Energy [meV]')
    # plt.tight_layout()
    # plt.show()


    # get scattering wave functions at E=0
    wf = kwant.wave_function(syst, energy=0, params=params)

    # prepare density operators
    sigma_z = np.array([[1, 0], [0, -1]])
    prob_density = kwant.operator.Density(syst)
    J_0 = kwant.operator.Current(syst)

    # calculate expectation values and plot them
    # wf_sqr = sum(prob_density(psi) for psi in wf(lead_index))
    current = sum(J_0(psi,params=params) for psi in wf(lead_index))

    # ax1 = plt.gca()
    # ax1.set_title(r'$\Psi_{\downarrow}$')
    # ax1.set_xlabel(r'$x$ [nm]')
    # ax1.set_ylabel(r'$y$ [nm]')
    # kwant.plotter.map(syst, wf_sqr,colorbar=False,fig_size=(9,2),ax=ax1)
    # plt.tight_layout()
    # plt.show()

    ax2 = plt.gca()
    ax2.set_title(r'$J_{\downarrow}$')
    ax2.set_xlabel(r'$x$ [nm]')
    ax2.set_ylabel(r'$y$ [nm]')
    kwant.plotter.current(syst, current,colorbar=False,fig_size=(9,2),ax=ax2)
    plt.tight_layout()
    plt.show()


def plot_conductance(syst, energies,L_barrier=100, pot=0, shift=0, choosed_color='red'):

    '''
    Essa função gera o gráfico da condutância do sistema dependendo da energia de Fermi
    imposto ao sistema + leads.

    syst = 'sistema' finalizado - kwant
    energies = array com os valores adotados para a energia de Fermi
    L_barrier = comprimento da barreira de potencial presente no sistema
    pot =
    shift =
    chossed_color = cor adotada para os 'dots' que compõem o gráfico

    '''
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
    plt.grid()
    plt.plot(energies, data, linestyle='', marker='o', color=choosed_color)
    plt.xlabel(r"$\varepsilon_F$ [meV]")
    plt.ylabel(r"$G_{01}~[e^2/h]$")
    plt.ylim(-1,10)
    plt.tight_layout()
    plt.show()





# analyze_bhz_up(pot=30,shift=0)
analyze_bhz_down(pot=30,shift=-30)

# syst = bhz_up(L=200)
# plot_conductance(syst, energies=np.linspace(-40,40,100))



# syst=bhz_down(L=200)
# plot_conductance(syst, energies=np.linspace(-40,40,100),choosed_color='blue')
