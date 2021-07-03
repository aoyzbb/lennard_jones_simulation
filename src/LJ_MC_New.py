import numpy as np
import matplotlib.pyplot as plt
import math
import random
from abc import *


class LJ_MC(ABC):
    def __init__(self, N, rho, T): pass

    def position_initial(self): pass

    def distance_calculate(self, i, j): pass

    def boundary_condition_check(self,n):pass

    def energy_for_all(self): pass

    def energy_for_one(self, n): pass

    def position_update(self, energy): pass

    def simulation(self,step):pass

    def equilibrium_simulation(self, step_num): pass



class LJ_MC_Periodic(LJ_MC):
    """The simulation model for Lennard Jones potential

    Use periodic boundary condition

    Attributes:
        N: number of particle
        T: temperature
        L: size of box
        rho: density of the box
        position: position of particle at present time
        energy: energy of the system at present time
    """

    def __init__(self, N, rho, T):
        self.N = N
        self.rho = rho
        self.T = T
        self.L = (N / rho) ** 0.5
        self.position = np.zeros((N, 2))
        self.position_initial()

    def position_initial(self):
        """Initial the particle position with lattice position

        Divide the box into lattice. Then place particle on it to
        avoid overlap.
        """
        pos_step = self.L * 3 / 5
        bias = self.L / 5
        num = int(self.N ** 0.5) + 1
        for i in range(self.N):
            self.position[i, 0] = (i // num) * pos_step / num + bias
            self.position[i, 1] = (i % num) * pos_step / num + bias

    def distance_calculate(self, i, j):
        """Calculate the distance between the two particle (or their images)

        Find the shortest distance among particle's images if the distance is larger than 2.5(cut-off distance).

        Args:
            i: the first particle
            j: the second particle (serve as origin)

        Returns:
            relative_dis : the distance between particle i and j (considering their images)
        """
        relative_pos = self.position[i] - self.position[j]
        relative_dis = np.linalg.norm(relative_pos)
        if relative_dis > self.L / 2:
            pos_candidate = np.array([relative_pos,
                                      relative_pos + [0, self.L],
                                      relative_pos - [0, self.L],
                                      relative_pos + [self.L, 0],
                                      relative_pos - [self.L, 0],
                                      relative_pos + [0, self.L] + [self.L, 0],
                                      relative_pos + [0, self.L] - [self.L, 0],
                                      relative_pos - [0, self.L] + [self.L, 0],
                                      relative_pos - [0, self.L] - [self.L, 0]])
            dis_candidate = np.linalg.norm(pos_candidate, axis=1)
            arg = np.argsort(-dis_candidate)
            relative_pos = pos_candidate[arg[0]]
            relative_dis = np.linalg.norm(relative_pos)

        return relative_dis

    def energy_for_all(self):
        """Calculate the energy for the system

        Returns:
            energy: the energy for the system
        """
        energy = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                relative_dis = self.distance_calculate(i,j)

                if relative_dis < 2.5:
                    if relative_dis < 1e-5:
                        print(i, j)
                        print(self.position[i], self.position[j])
                        raise ValueError("distance between particles is too small")

                    r2i = relative_dis ** (-2)
                    r6i = r2i ** 3
                    energy += 4 * r6i * (r6i - 1)

        return energy

    def boundary_condition_check(self,n):
        """Use periodic boundary condition to check and update the position

        Args:
            n: the index of particle to be considered

        Returns:
            pos: the new position for particle n when considering the boundary
        """
        pos = self.position[n] % self.L
        return pos


    def energy_for_one(self, n):
        """Calculate the energy associated with particle n

        return the energy of particle n with other particles

        Args:
            n: the index of particle

        Returns:
            energy: the energy associated with particle n
        """

        energy = 0
        position_n = self.position[n]
        for i in range(self.N):
            if i != n:
                relative_dis = self.distance_calculate(i,n)

                if relative_dis < 2.5:
                    if relative_dis < 1e-5:
                        print(i, n)
                        print(self.position[i], position_n)
                        raise ValueError("distance between particles is too small")

                    r2i = relative_dis ** (-2)
                    r6i = r2i ** 3
                    energy += 4 * r6i * (r6i - 1)

        return energy

    def position_update(self, energy, ctrl_step=0.031):
        """Update the position of the system
        
        Use Metropolis Algorithm and periodic boundary condition
        
        Args:
            energy: the energy at previous step
            ctrl_step: the relative change of position this step will make

        Returns:
            energy: the energy at current step
        """
        step = ctrl_step * self.L * (np.random.rand(2) - 0.5)
        particle_change = random.randint(0, self.N - 1)
        energy_old_n = self.energy_for_one(particle_change)
        old_pos = self.position[particle_change].copy()
        self.position[particle_change] = self.position[particle_change] + step

        self.position[particle_change] = self.boundary_condition_check(particle_change)

        energy_new_n = self.energy_for_one(particle_change)
        energy_change = energy_new_n - energy_old_n

        if energy_change >= 0:
            if random.random() < math.exp(-energy_change / self.T):
                energy += energy_change

            else:
                self.position[particle_change] = old_pos

        else:
            energy += energy_change

        return energy


    def simulation(self,step,flag=0):
        """Simulation the system for specific steps

        Args:
            step: the number of step to be simulated
            flag: the form of energy to return. flag = 0: the energy at the final state ; flag = 1 ,return energy of each step

        Returns:
            energy: the energy during the process
        """

        if flag == 1:
            energy = np.zeros(step)
            energy[0] = self.energy_for_all()
            for i in range(1,step):
                energy[i] = self.position_update(energy[i - 1])
                if i % 100 == 0:
                    energy[i] = self.energy_for_all()
                    print('step:', i)

        if flag == 0:
            energy = self.energy_for_all()
            for i in range(1,step):
                energy = self.position_update(energy)
                if i % 100 == 0:
                    energy = self.energy_for_all()
                    if i %1000 == 0:
                        print('simu, step:', i)


        return energy

    def equilibrium_simulation(self, step_num):
        """begin simulation and  bring the system into to equilibrium

        The equilibrium might take some time. So use a stack with the length of step_num
        The equilibrium condition is that the relative fluctuation is smaller than 1.
        Since each step only calculate the change of energy at this step.
        The error might accumulate. Update the energy at every 100 step with
        grand energy calculation algorithm

        Args:
            step_num: the number of step per cycle
        Returns:
            energy: the energy at each step in the last cycle
        """
        energy = np.zeros(step_num)
        energy[0] = self.energy_for_all()
        step_count = 0
        threshold = 0.03 * self.T
        while abs(energy.std()/energy.mean())> threshold or energy.mean()>0:
            self.simulation(10 * step_num)
            energy[0] = self.energy_for_all()
            for i in range(1, step_num):
                energy[i] = self.position_update(energy[i - 1])

                if i % 1000 == 1:
                    energy[i] = self.energy_for_all()
                    #print('step:', i)

            step_count +=  (11 *step_num)
            if step_count > 1e6:
                print('stop now')
                break

        #print(self.position)
        return energy


class LJ_MC_Hard(LJ_MC_Periodic):
    """The model for hard boundary

    The difference with periodic boundary is the method to calculate distance and
    the method to treat particles hit the wall

    """

    def distance_calculate(self, i, j):
        """Calculate the distance with hard boundary"""

        return np.linalg.norm(self.position[i] - self.position[j])



    def boundary_condition_check(self,n):
        """check the position of particle n with hard boundary"""

        pos = self.position[n].copy()
        for i in range(2):
            if self.position[n, i] <= 0:
                pos[i] = - self.position[n, i]

            if self.position[n, i] >= self.L:
                pos[i] = 2 * self.L - self.position[n, i]

            assert (pos[i] >= 0 and pos[i] <=self.L)

        return pos

class LJ_MC_Periodic_random(LJ_MC_Periodic):
    """system initialized with random position"""

    def position_initial(self):
        self.position = np.random.random((self.N,2)) * self.L




def test():
    """A test function for LJ_MC_Periodic

    Test the model with rho = 0.84 ,N =84 , T = 0.78 . A figure illustrate the potential
    at equilibrium will be created.
    """
    A = LJ_MC_Periodic(84, 0.84, 0.78)
    plt.figure()
    step_num = 300
    potential = A.equilibrium_simulation(step_num)
    plt.plot(np.arange(step_num), potential, 'g-')
    plt.show()


if __name__ == '__main__':
    test()
