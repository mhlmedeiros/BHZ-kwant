import kwant

import tinyarray
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla


font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 22}

mpl.rc('font', **font)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



ge = 22.7
gh = -1.21

sigma_0 = tinyarray.array([[1,0],[0,1]])
sigma_x = tinyarray.array([[0,1],[1,0]])
sigma_y = tinyarray.array([[0,-1j],[1j,0]])
sigma_z = tinyarray.array([[1,0],[0,-1]])
sigma_g = tinyarray.array([[ge,0],[0,gh]])

def qsh_system_with_mag(a=2, L=500, W=200):

       hamiltonian = """
              + C * identity(4) + M * kron(sigma_0, sigma_z)
              - B * ((k_x - y * B_0/(25)**2)**2 + k_y**2) * kron(sigma_0, sigma_z)
              - D * ((k_x - y * B_0/(25)**2)**2 + k_y**2) * kron(sigma_0, sigma_0)
              + A * (k_x - y * B_0/(25)**2) * kron(sigma_z, sigma_x)
              - A * k_y * kron(sigma_0, sigma_y)
              + mu_B * B_0/2 * kron(sigma_z, Matrix([[22.7,0],[0,-1.21]]))
       """

       # hamiltonian_syst = """
       #        + C(x,y) * identity(4) + M * kron(sigma_0, sigma_z)
       #        - B * (k_x**2 + k_y**2) * kron(sigma_0, sigma_z)
       #        - D * (k_x**2 + k_y**2) * kron(sigma_0, sigma_0)
       #        + A * k_x * kron(sigma_z, sigma_x)
       #        - A * k_y * kron(sigma_0, sigma_y)
       # """
       #
       # hamiltonian_lead = """
       #        + C_const * identity(4) + M * kron(sigma_0, sigma_z)
       #        - B * (k_x**2 + k_y**2) * kron(sigma_0, sigma_z)
       #        - D * (k_x**2 + k_y**2) * kron(sigma_0, sigma_0)
       #        + A * k_x * kron(sigma_z, sigma_x)
       #        - A * k_y * kron(sigma_0, sigma_y)
       # """

       template_syst = kwant.continuum.discretize(hamiltonian, grid_spacing = a)
       # template_syst = kwant.continuum.discretize(hamiltonian_syst, grid_spacing = a)
       # template_lead = kwant.continuum.discretize(hamiltonian_lead, grid_spacing = a)

       def shape(site):
           (x, y) = site.pos
           return (-W/2 < y < W/2 and -L/2 < x < L/2)

       def lead_shape(site):
           (x, y) = site.pos
           return (-W/2 < y < W/2)

       syst = kwant.Builder()
       syst.fill(template_syst, shape, (0, 0))

       lead = kwant.Builder(kwant.TranslationalSymmetry([-a, 0]))
       lead.fill(template_syst, lead_shape, (0, 0))

       syst.attach_lead(lead)
       syst.attach_lead(lead.reversed())

       syst = syst.finalized()
       return syst

def plot_by_plotter():

       kx_max = 0.5
       kx_min = -kx_max

       kwant.plotter.bands(syst.leads[0], params=params,
                          momenta=np.linspace(kx_min, kx_max, 201),
                          fig_size=(6,6), show=False)

       plt.grid()
       plt.xlim(kx_min, kx_max)
       plt.ylim(-100, 100)
       plt.xlabel(r'momentum [nm$^{-1}$]')
       plt.ylabel(r'Energy [meV]')
       plt.tight_layout()
       plt.show()

def bandas_data(syst, a = 2, B_0 = 0, k_pcent = 0.25):

       params = dict( A = 364.5, B = -686.0,
                     D = -512.0, M = -10.0,
                     C = 0, B_0 = B_0, mu_B = 57.84E-3 )

       # syst = qsh_system_with_mag()
       bands = kwant.physics.Bands(syst.leads[0], params = params)
       momenta = np.linspace(-np.pi/a, np.pi/a, 201) * k_pcent
       energies = [bands(k) for k in momenta]

       plt.plot(momenta,energies,marker=",")
       # plt.xlim(kx_min,kx_max)
       plt.ylim(-40, 40)
       plt.xlabel(r'momentum [nm$^{-1}$]')
       plt.ylabel(r'Energy [meV]')
       plt.tight_layout()
       plt.show()

def bhz_up(a=2, L=1000, W=200):
    hamiltonian_syst = """
    + C * identity(2) + M * sigma_z
    - B * ((k_x - y * B_0/(25)**2)**2 + k_y**2) *  sigma_z
    - D * ((k_x - y * B_0/(25)**2)**2 + k_y**2) *  sigma_0
    + A * (k_x - y * B_0/(25)**2) * sigma_x
    - A * k_y * sigma_y
    + mu_B * B_0/2 * Matrix([[22.7,0],[0,-1.21]])
    """

    template_syst = kwant.continuum.discretize(hamiltonian_syst, grid_spacing=a)
    template_lead = kwant.continuum.discretize(hamiltonian_syst, grid_spacing=a)

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
    + C * identity(2) + M * sigma_z
    - B * ((k_x - y * B_0/(25)**2)**2 + k_y**2) * sigma_z
    - D * ((k_x - y * B_0/(25)**2)**2 + k_y**2) * sigma_0
    - A * (k_x - y * B_0/(25)**2) * sigma_x
    - A * k_y * sigma_y
    - mu_B * B_0/2 * Matrix([[22.7,0],[0,-1.21]])
    """

    template_syst = kwant.continuum.discretize(hamiltonian_syst, grid_spacing=a)
    template_lead = kwant.continuum.discretize(hamiltonian_syst, grid_spacing=a)

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

def bandas_up_and_down(syst1, syst2, line1="-", line2="--", a = 2, B_0 = 0, k_pcent = 0.55):

    params = dict( A = 364.5, B = -686.0,
                 D = -512.0, M = -10.0,
                 C = 0, B_0 = B_0, mu_B = 57.84E-3 )


    bands1 = kwant.physics.Bands(syst1.leads[0], params = params)
    bands2 = kwant.physics.Bands(syst2.leads[0], params = params)

    momenta = np.linspace(-np.pi/a, np.pi/a, 201) * k_pcent

    energies1 = [bands1(k) for k in momenta]
    energies2 = [bands2(k) for k in momenta]

    plt.plot(momenta, energies1, linestyle = line1,color="blue")
    plt.plot(momenta, energies2, linestyle = line2,color="red")
    plt.xlim(momenta[0],momenta[200])
    plt.ylim(-20, 20)
    plt.xlabel(r'momentum [nm$^{-1}$]')
    plt.ylabel(r'Energy [meV]')
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.tight_layout()
    plt.show()

def band_k_B(syst, k_0 = 0, B_0 = 0):
    params = dict( A = 364.5, B = -686.0,
    D = -512.0, M = -10.0,
    C = 0, B_0 = B_0, mu_B = 57.84E-3 )


    bands = kwant.physics.Bands(syst.leads[0], params = params)
    return bands(k_0)

def collect_energies(syst, B_list, k_0 = 0):
       energies = []
       for B in B_list:
           energies.append(band_k_B(syst, B_0 = B))
       return energies

def plot_energy_vs_B(sistema_up, sistema_down, line1 = "-", line2 = "--", k_0 = 0):
    B_array = np.linspace(0,10,201)

    bands1 = collect_energies(sistema_up, B_array)
    bands2 = collect_energies(sistema_down, B_array)

    plt.plot(B_array, bands1, linestyle = line1,color="blue",label="oi")
    plt.plot(B_array, bands2, linestyle = line2,color="red",label="tchau")
    # plt.xlim(momenta[0],momenta[200])
    # plt.legend()
    plt.ylim(-100, 100)
    plt.xlabel(r'B [T]')
    plt.ylabel(r'Energy [meV]')
    plt.grid()
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.tight_layout()
    plt.show()


def main():

       # sistema = qsh_system_with_mag()
       # bandas_data(sistema, B_0 = 0.1, k_pcent=0.1, a=1)

       sistema_up = bhz_up()
       # bandas_data(sistema_up, B_0 = 0.11)

       sistema_down = bhz_down()
       # bandas_data(sistema_down, B_0 = 0.1)

       bandas_up_and_down(sistema_up, sistema_down, B_0 = 0.1, a=0.5, k_pcent=0.01)

       # plot_energy_vs_B(sistema_up, sistema_down)


if __name__ == '__main__':
    main()
