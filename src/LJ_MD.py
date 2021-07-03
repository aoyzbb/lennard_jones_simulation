import numpy as np
import matplotlib.pyplot as plt
import random
from abc import *


class LJ_MD(ABC):
    def __init__(self, rho, N, T, dt): pass

    def Position_and_Velocity_init(self): pass

    def distance_calculate(self, i, j): pass

    def boundary_condition_check(self):pass

    def Force_calculate(self): pass

    def Position_and_velocity_update(self, nudt): pass

    def equilibrium_simulation(self, step_num): pass

    def simulation(self,step , flag = 0):pass


class LJ_MD_Periodic(LJ_MD):
    """Molecular Dynamics with Lennard_Jones Potential

    The cut-off distance is rc = 2.5 sigma, use periodic boundary condition

    Attributes:
        rho: the density of the particle
        N: the number of particle
        T: the temperature of the model
        dt: the simulation step
        position: the position of particle at present time
        velocity: the velocity of particle at present time
    """

    def __init__(self, N,rho, T, dt=0.001):
        self.rho = rho
        self.N = N
        self.L = (N / rho) ** 0.5
        self.T = T
        self.dt = dt
        self.position = np.zeros((N, 2))
        self.velocity = np.zeros((N, 2))
        self.Position_and_Velocity_init()

    def Position_and_Velocity_init(self):
        """Initialize position and velocity

        Position is initialized with lattice position
        Velocity is initialized with Maxwell-Boltzmann distribution

        """
        pos_step = self.L * 3 / 5
        bias = self.L / 5
        num = int(self.N ** 0.5) + 1
        for i in range(self.N):
            self.position[i, 0] = (i // num) * pos_step / num + bias
            self.position[i, 1] = (i % num) * pos_step / num + bias

        self.velocity = np.random.normal(0, self.T ** 0.5, (self.N, 2))
        _, potential = self.Force_calculate()
        kinetic = np.linalg.norm(self.velocity) ** 2
        print("initial condition:", potential, kinetic)

    def distance_calculate(self, i, j):
        """calculate the distance between particle i and j

        Use periodic boundary condition,calculate the distance between i and j

        Args:
            i: the first particle
            j: the second particle (serve as origin)

        Returns:
            relative_pos : the relative position between particle i and j (considering their image)
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
            arg = np.argsort(dis_candidate)
            relative_pos = pos_candidate[arg[0]]
            relative_dis = np.linalg.norm(relative_pos)

        return relative_pos , relative_dis

    def boundary_condition_check(self):
        """check the boundary condition"""
        self.position = self.position % self.L

    def Force_calculate(self):
        """Force Calculation for current time

        The boundary is periodic boundary

        Returns:
            force: ndarray (N,2), the force on the particle
            potential: potential of the system
        """
        force = np.zeros((self.N, 2))
        potential = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                relative_pos, relative_dis = self.distance_calculate(i,j)

                if relative_dis < 2.5:
                    if relative_dis < 1e-5:
                        print(i, j)
                        print(self.position[i], self.position[j])
                        print(self.velocity[i], self.velocity[j])
                        raise ValueError("distance between particles is too small")
                    r2i = relative_dis ** (-2)
                    r6i = r2i ** 3
                    ff = 48 * r2i * r6i * (r6i - 0.5)

                    force[i] += (ff * relative_pos)
                    force[j] -= (ff * relative_pos)
                    potential += 4 * r6i * (r6i - 1)


        return force, potential

    def Position_and_velocity_update(self, nudt=0.01):
        """Update position and velocity at the present time

        Use Verlet-Like algorithm, See Understanding Molecular Simulation  Page144
        Then, Anderson Thermostat is used. The collision frequency is initially set to 0.5

        Args:
            nudt: probability for each particle to collide with the thermostat at each step
            flag: whether to use thermostat at this step

        Returns:
            potential: potential of current step
            kinetic: kinetic energy of current step

        """

        force_old, _ = self.Force_calculate()
        self.position = self.position + self.velocity * self.dt + force_old * (self.dt ** 2) / 2

        force_new, potential = self.Force_calculate()
        self.velocity = self.velocity + self.dt * (force_old + force_new) / 2

        self.boundary_condition_check()

        threshold = nudt
        for i in range(self.N):
            if random.random() < threshold:
                self.velocity[i] = np.random.normal(0, self.T ** 0.5, (2))

        kinetic = np.linalg.norm(self.velocity) ** 2
        return potential, kinetic

    def equilibrium_simulation(self, step_num):
        """Simulation of the model

        time will be changed to reach equilibrium

        Args:
            step_num: the number of step
        Returns:
            potential: the potential at each step
            kinetic: the kinetic energy at each step

        """
        potential = np.zeros(step_num)
        kinetic = np.zeros(step_num)
        potential[0], kinetic[0] = self.Position_and_velocity_update()
        threshold = 0.05 * self.T
        while abs(potential.std()/potential.mean()) > threshold or potential.mean()>0:
            for i in range(0, step_num):
                potential[i], kinetic[i] = self.Position_and_velocity_update(nudt=0.1)
                if i % 10 == 0:
                    print("step", i)
            print(potential.mean())
            print(kinetic.mean())

        return potential, kinetic

    def simulation(self, step, flag=0):
        """Simulate the system with giving steps

        Args:
            step: number of steps to be simulated
            flag: the return type. flag = 0 , the energy and potential for the final step; flag = 1, the energy and potential for every step
        """
        if flag == 0:
            for i in range(0,step-1):
                self.Position_and_velocity_update(nudt = 0.1)
                if i%10 == 0:
                    print("step:",i)
            potential,kinetic =  self.Position_and_velocity_update(nudt = 0.1)

        if flag == 1:
            potential = np.zeros(step)
            kinetic = np.zeros(step)
            for i in range(step):
                potential[i], kinetic[i] = self.Position_and_velocity_update(nudt=0.1)
                if i % 100 == 0:
                    print("step", i)

        return potential,kinetic



class LJ_MD_Hard(LJ_MD_Periodic):
    """Molecular Dynamics with Lennard_Jones Potential

    The cut-off distance is rc = 2.5 sigma, use hard boundary condition

    """

    def distance_calculate(self, i, j):
        relative_pos = self.position[i] - self.position[j]
        relative_dis = np.linalg.norm(relative_pos)
        return relative_pos,relative_dis

    def boundary_condition_check(self):
        for i in range(self.N):
            for j in range(2):
                if self.position[i,j] < 0 :
                    self.position[i,j] = -self.position[i,j]
                    self.velocity[i,j] = -self.velocity[i,j]

                if self.position[i,j] > self.L:
                    self.position[i,j] = 2 * self.L - self.position[i,j]
                    self.velocity[i,j] = - self.velocity[i,j]


def test():
    A = LJ_MD_Periodic(84, 0.84, 0.78, dt=0.001)

    plt.figure()
    step_num = 100
    potential, kinetic = A.equilibrium_simulation(step_num)
    plt.plot(np.arange(step_num), potential, 'g-')
    plt.show()




if __name__ == '__main__':
    test()
