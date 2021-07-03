from LJ_MC_New import *
from LJ_MD import *
import matplotlib.pyplot as plt
import h5py

def plotU():
    """Generate a plot about MC and MD

    Plot the energy of the MD and MC system at the same plot.
    The beginning 200 steps are discarded since the energy is too big
    """
    fig,ax = plt.subplots(1,1)
    period = LJ_MC_Periodic(90, 0.1 , 1)
    hard = LJ_MC_Hard(90,0.1,1)
    period.simulation(300 , flag = 0)
    hard.simulation(300, flag =0)

    step = np.linspace(0,4999,5000)
    potential_hard = hard.simulation(5000, flag = 1)
    potential_periodic = period.simulation(5000, flag =1)

    ax.plot(step,potential_hard , label = 'hard boundary condition,potential')
    ax.plot(step,potential_periodic , label = 'periodic boundary condition,potential')

    ax.set_xlabel("time")
    ax.set_ylabel("potential")
    ax.legend()

    ax.set_title("MC simulation. U(t),second parameter set")


    plt.show()

def equilibrium_plot():
    """Plot the energy(MC) or potential and kinetic energy(MD) after equilibrium"""
    fig,(ax1,ax2) = plt.subplots(2,1)
    period = LJ_MC_Periodic(84, 0.84 , 0.72)
    hard = LJ_MC_Hard(84,0.84,0.72)
    period.equilibrium_simulation(400)
    hard.equilibrium_simulation(400)
    hard_potential = hard.simulation(300,flag = 1)
    period_potential= period.simulation(300,flag = 1)
    hard_MA = np.zeros(270)
    period_MA = np.zeros(270)
    for i in range(270):
        hard_MA[i] = hard_potential[i:i+30].mean()
        period_MA[i] = period_potential[i:i+30].mean()

    step = np.linspace(30,300,270)
    ax1.plot(np.linspace(0,299,300),hard_potential,label = 'potential for hard')
    ax1.plot(step,hard_MA,label = 'MA30 for hard')
    ax2.plot(np.linspace(0,299,300),period_potential,label = 'potential for periodic')
    ax2.plot(step,period_MA,label = 'MA30 for periodic')
    ax1.set_title('the potential for MC after equilibrium with second set parameter')
    ax1.legend()
    ax2.legend()
    plt.show()

def devariance_calculate():
    """calculate the fluctuation of the energy"""
    periodic = LJ_MC_Periodic(84,0.84,0.728)
    periodic_potential = periodic.equilibrium_simulation(400)

    hard = LJ_MC_Hard(84,0.84,0.728)
    hard_potential = hard.equilibrium_simulation(400)
    print(periodic_potential.std())
    print(hard_potential.std())

def devariance_plot():
    """Plot delta_U(t) against N"""
    energy_d = np.zeros(20)
    energy = np.zeros(20)
    filename = './position.h5'
    h5f = h5py.File(filename,'w')

    for i in range(50,150 ,20):
        A = LJ_MC_Periodic(i,0.84,0.728)
        periodic_potential = A.equilibrium_simulation(600)
        np.save('./configuration',A.position)
        j = int((i-50)/5)
        energy_d[j] = periodic_potential.std()
        energy[j] = periodic_potential.mean()
        h5f.create_dataset('position' + str(i), data=A.position)

        print('the N',i)

    np.save('./energy_detail',energy)
    np.save('./variance_detail',energy_d)
    step = 1 / np.linspace(50,145,5)
    h5f.close()
    fig, (ax1,ax2) = plt.subplots(2,1)
    ax1.plot(step,energy_d)
    ax1.set_title('delta_U against 1/N')
    ax1.set_xlabel('1/N')
    ax1.set_ylabel('energy variance')
    ax2.plot(step,energy)
    ax2.set_xlabel('1/N')
    ax2.set_ylabel('energy')
    plt.show()


def long_simu():
    """Use very long step to simulate

    Choose step_num = 1e5
    """
    energy = np.zeros(10)
    variance = np .zeros(10)
    for i in range(100,300,20):
        A = LJ_MC_Hard(i,0.84,0.728)
        A.equilibrium_simulation(400)
        A.simulation(int(10000),flag = 0)
        period_energy = A.simulation(500,flag = 1)
        j = int((i-100)/20)
        print(j) 
        energy[j] = period_energy.mean()
        variance[j] = period_energy.std()

    np.savetxt('./energy_detail_hard.txt',energy)
    np.savetxt('./variance_detail_hard.txt',variance)
    fig, (ax1,ax2) = plt.subplots(2,1)
    step = np.linspace(100,280,10)
    ax1.plot(step,variance)
    ax1.set_title('delta_U against N with hard condition')
    ax1.set_xlabel('N')
    ax1.set_ylabel('energy variance')
    ax2.plot(step,energy)
    ax2.set_xlabel('N')
    ax2.set_ylabel('energy')
    plt.show()


def radial_distribution(flag = 0):
    """
    Calculate the radical distribution of the system
    """
    distance = np.array([])
    if flag == 0 :
        A = LJ_MC_Periodic(84,0.84,0.72)
    elif flag == 1 :
        A = LJ_MC_Periodic(90,0.1,1)
    else:
        A = LJ_MC_Periodic(N = 110, rho = 1.1 , T = 0.9)
    A.equilibrium_simulation(500)
    A.simulation(10000,flag = 0)
    for i in range(A.N):
        for j in range(i+1,A.N):
            distance = np.append(distance,A.distance_calculate(i,j))

    print(A.energy_for_all(),A.N)
    fig,ax = plt.subplots(1,1)
    if flag == 0:
        np.savetxt('./distance_par1.txt',distance)
        ax.set_title('The distribution with par1, N = 84')
    if flag == 1:
        np.savetxt('./distance_par2.txt',distance)
        ax.set_title('The distribution with par2, N = 90')
    if flag == 2:
        np.savetxt('./distance_par3.txt',distance)
        ax.set_title('The distribution with par3, N = 110')

    ax.hist(distance,bins = 'auto')

    plt.show()


def plot_variance_and_energy():
    """Help function, shit code"""
    energy = np.log2(-np.loadtxt('./energy_detail.txt'))
    energy_for_h = np.log2(-np.loadtxt('./energy_detail_hard.txt'))

    variance = np.log2(np.loadtxt('./variance_detail.txt'))
    variance_for_h = np.log2(np.loadtxt('./variance_detail_hard.txt'))
    fig, (ax1,ax2) = plt.subplots(2,1)
    step = np.log2(1/np.linspace(100,220,6))
    ax1.plot(step,variance[0:6], label = 'period condition')
    ax1.plot(step[0:5],variance_for_h[0:5], label = 'hard condition')
    ax1.set_title('delta_U against N with period condition,par1')
    ax1.set_xlabel('ln(1/N)')
    ax1.set_ylabel('energy variance')
    ax2.plot(step,energy[0:6],label = 'period condition')
    ax2.plot(step[0:5],energy_for_h[0:5],label = 'hard condition')
    ax2.set_xlabel('ln(1/N)')
    ax2.set_ylabel('ln(-energy)')
    ax1.legend()
    ax2.legend()
    plt.show()

def plot_distribution(flag):
    """plot the radical distribution, shit code"""
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

    _,n,_ = ax.hist(distance,range = (0,2.5),bins = 70)
    print(n)
    print(np.shape(n))
    plt.show()



if __name__ == '__main__':
    plot_distribution(2)

