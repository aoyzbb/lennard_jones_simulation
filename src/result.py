from LJ_MC_New import *
from LJ_MD import *
import matplotlib.pyplot as plt
import h5py

def cal_energy_with_g(flag):
    fig,ax = plt.subplots(1,1)
    if flag == 0:
        distance = np.loadtxt('./distance_par1.txt')
        ax.set_title('The distribution with par1, N = 84')
    if flag == 1:
        distance = np.loadtxt('./distance_par2.txt')
        ax.set_title('The distribution with par2, N = 90')
    if flag == 2:
        distance = np.loadtxt('./distance_par3.txt')
        ax.set_title('The distribution with par3, N = 110')
    energy = 0
    n,bins,_ = ax.hist(distance,range = (0,2.5),bins = 200)
    for i in range(1,np.size(bins)):
        if bins[i] < 2.5 and bins[i] != 0:
            r2i = bins[i] **(-2)
            r6i = r2i ** 3
            energy += n[i] * 4 * r6i * ( r6i -1 )

    plt.show()
    return energy

class LJ_MC_Periodic_Pressure(LJ_MC_Periodic_random):
    """class with method to calculate the pressure"""

    def pressure_calculate(self):
        """method to calculate,tail correction is considered """
        vir = 0
        for i in range(self.N):
            for j in range(i+1,self.N):
                distance = self.distance_calculate(i,j)
                if distance < 2.5:
                    vir += 16 * distance**(-1) *(distance **(-12) - 0.5 * distance ** (-6))

        pressure_trun = self.rho * self.T + self.L ** (-2) * vir /3
        tail_cre = 48 *  math.pi * (self.rho**2) * ((1/2.5)** 11 * (1/11) -0.1*(1/2.5)**5)
        return pressure_trun + tail_cre

def pressure_calculation(rho , N = 500 , T = 0.9):
    system = LJ_MC_Periodic_Pressure(N,rho , T)
    system.equilibrium_simulation(300)
    system.simulation(10000,flag = 0)
    pressure = np.zeros(10)
    for i in range(10):
        pressure[i] = system.pressure_calculate()
        system.simulation(1000,flag = 0)

    print('debug:',pressure.mean())
    return pressure.mean()

def equation_of_state(T):
    rho_step = np.linspace(0.1,1,10)
    pressure = np.zeros(10)
    for i in range(10):
        pressure[i] =pressure_calculation(rho_step[i],100,T = 0.3)

    fig ,ax= plt.subplots(1,1)
    ax.plot(rho_step,pressure , label = 'T = 0.3')
    ax.set_title('Equation of state')
    ax.set_xlabel('rho')
    ax.set_ylabel('pressure')

    pressure_2 = np.zeros(10)
    for i in range(10):
        pressure_2[i] =pressure_calculation(rho_step[i],100,T = 0.4)


    ax.plot(rho_step,pressure_2,label = 'T =0.4')

    pressure_3 = np.zeros(10)
    for i in range(10):
        pressure_2[i] =pressure_calculation(rho_step[i],100,T = 0.2)
    ax.plot(rho_step,pressure_3,label = 'T =0.2')
    ax.legend()
    plt.show()

def initial_potision():
    """Plot the track of systems with different initial condition"""
    crystal = LJ_MC_Periodic(84,0.84,0.72)
    random_sys = LJ_MC_Periodic_random(84,0.84,0.72)
    fig ,ax = plt.subplots(1,1)
    crystal.simulation(1500)
    energy_cry = crystal.simulation(8000,flag =1)
    random_sys.simulation(1500)
    energy_ran = random_sys.simulation(8000,flag = 1)
    step = np.linspace(0,7999,8000)
    ax.plot(step,energy_cry, label = 'energy of crystal initial position')
    ax.plot(step,energy_ran, label = 'energy of random initial position')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    fig ,ax= plt.subplots(1,1)
    rho_step = np.linspace(0.1,1,10)
    pressure = np.array([0.028213591175048208,0.05339473634030384,0.07654063288428475,
                         0.07654063288428475,0.1149260544005575, 0.10402902063554106
                         ,0.11848404593341415,0.207861720474347, 0.43822387885844655,
                          1.016505104832428])
    pressure_2 = np.array([0.03811191242038421,0.06875364768881714,0.11607941410269396,
                           0.15070416184900293,0.18119213977843035,0.17638039084990026,
                           0.3139456118559291,0.3183276277177708,0.7295290140678845,
                           1.2157466493381102])
    pressure_3 = np.array([0.018229123346988202,0.03482893748645091, 0.04094904528802272
                           ,0.04343589329459675,0.04674071346955698,0.05803220260472498,
                           0.1239527648208826,0.11606370161300941,0.5033077045641929,
                           0.7112054438988362])
    #x.plot(rho_step,pressure_3,label = 'T =0.2')
    #ax.plot(rho_step,pressure , label = 'T = 0.3')
    ax.set_title('Equation of state,produced by Ao')
    ax.set_xlabel('rho')
    ax.set_ylabel('pressure')
    ax.plot(rho_step,pressure_2,label = 'T =0.4')
    ax.legend()
    plt.show()

