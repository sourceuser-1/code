# Author: Kishansingh Rajput
# Script: CEBAF cavities digital twin
# Org: Thomas Jefferson National Accelerator Facility

import numpy as np
# from CEBAF_opt.utils import format
import pandas as pd
import math

from gym import spaces
import gym

class cavity():
    """


    """
    def __init__(self, data, cavity_id):
        """

        :param pathToCavityData:
        """
        # data = pd.read_pickle(path_cavity_data)
        self.cavity_id = cavity_id
        try:
            row = data[data["cavity_id"] == cavity_id]
        except(e):
            print(e)

        if len(row) < 1:
            print("Cavity-id ", self.cavity_id, " not found in the dataset...")
        self.length = float(row["length"])
        self.type = str(row['type'])
        self.Q0 = float(row["Q0"])
        self.trip_slope = float(row['trip_slope'])
        self.trip_offset = float(row['trip_offset'])
        self.shunt = float(row['shunt_impedance'])
        self.max_gset = float(row['max_gset'])
        self.ops_gset_max = float(row['ops_gset_max'])
        if pd.isna(self.ops_gset_max):
            self.max_gset_to_use = self.max_gset
        else:
            self.max_gset_to_use = self.ops_gset_max
        self.min_gset = 3.0
        ## IF C100 set min gset to 5.0
        ## If C75 ? (Ask Jay)
        self.min_gset = 3.0
        self.gradient = self.max_gset_to_use
        if self.type in ["C75", "C100"]:
            self.min_gset = 5.0
            def computeHeat():
                return ((self.gradient**2) * self.length * 1e12) / (self.shunt * self.Q0)
        else:
            def computeHeat():
                return ((self.gradient**2) * self.length * 1e12) / (self.shunt * self.Q0)

        self.RFheat = computeHeat
        self.chargeFraction = self.getTripRate()

    def describe(self):
        """

        :return:
        """
        print("Cavity type: ", self.cavity_id)
        print("Cavity current gradient: ", self.gradient)
        print("Cavity length: ", self.length)
        print("Cavity Q0: ", self.Q0)
        print("Cavity trip slope: ", self.trip_slope)
        print("Cavity trip offset: ", self.trip_offset)
        print("Cavity shunt: ", self.shunt)
        print("Cavity max_gset: ", self.max_gset)
        print("Cavity ops gset max: ", self.ops_gset_max)
        print("max gset to use: ", self.max_gset_to_use)



    def setGradient(self, grad):
        """

        :param gradArray:
        :return:
        """
        if type(grad) in [int, float, np.float32, np.float64]:
            if grad < self.min_gset:
                # print("Error: ", self.cavity_id, " requested gradient is lower than minimum safe gradient. Setting to min...")
                self.gradient = self.min_gset
            # elif grad > self.max_gset_to_use:
            elif grad > self.max_gset:
                # print("Error: ", self.cavity_id, " requested gradient is higher than maximum safe gradient. Setting to max...")
                self.gradient = self.max_gset_to_use
            else:
                self.gradient = grad
        else:
            print("Error: ", self.cavity_id, " gradient must be a float or integer and not ", type(grad))

        self.chargeFraction += 60*self.getTripRate()

    def getRFHeat(self):
        """

        :return:
        """
        return self.RFheat()


    def getTripRate(self):
        """

        :return:
        """
        if pd.isna(self.trip_slope) or pd.isna(self.trip_offset):
            return 0
        return math.exp(-10.268+self.trip_slope*(self.gradient - self.trip_offset))

    def getGradient(self):
        return self.gradient

    def getEnergy(self):
        return self.length * self.gradient

    def reset(self):
        self.gradient = self.max_gset_to_use - 0.02
        # self.gradient = self.min_gset
        # self.gradient = 10.48
        self.chargeFraction = self.getTripRate()

class digitalTwin():
    """

    """
    def __init__(self, path_cavity_data, linac="North"):
        """

        :param path_cavity_data:
        """
        data = pd.read_pickle(path_cavity_data)
        cavity_ids = data["cavity_id"]
        self.cavities = []
        self.cavity_order = []
        for i in range(len(cavity_ids)):
            if linac.lower() == "north" or linac.lower() == "n":
                if cavity_ids.iloc[i][0] == '1':
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            elif linac.lower() == "south" or linac.lower() == "s":
                if cavity_ids.iloc[i][0] == '2':
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            elif linac.lower() == "test":
                if cavity_ids.iloc[i][0] == '2' and cavity_ids.iloc[i][2] == '0' and cavity_ids.iloc[i][3] == '6':
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            else:
                if cavity_ids.iloc[i][0] == '0':
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
        self.name = linac

        if linac.lower() in ["north", "n", "south", "s"]:
            self.energyConstraint = 1150
            self.energyMargin = 2
        elif linac.lower() == "test":
            self.energyConstraint = 31.85
            self.energyMargin = 0.1
        else:
            self.energyConstraint = 126.5
            self.energyMargin = 0.22

    def list_cavities(self):
        """

        :return:
        """
        return self.cavity_order

    def describeCavity(self, cavity_id):
        """

        :param cavity_id:
        :return:
        """
        indx = self.cavity_order.index(cavity_id)
        self.cavities[indx].describe()

    def setGradients(self, grad_array):
        """

        :param grad_array:
        :return:
        """
        for i in range(len(self.cavities)):
            self.cavities[i].setGradient(grad_array[i])

    def getGradients(self):
        """

        :return:
        """
        gradients = []
        for cavity in self.cavities:
            gradients.append(cavity.getGradient())
        return np.array(gradients)

    def getRFHeat(self):
        """

        :return:
        """
        heat = 0.0
        for cavity in self.cavities:
            heat += cavity.getRFHeat()
        return heat

    def getTripRates(self):
        """

        :return:
        """
        tr = 0.0
        for cavity in self.cavities:
            tr += cavity.getTripRate()
        return 3600*tr

    def getEnergyGain(self):
        """

        :return:
        """
        e = 0.0
        for cavity in self.cavities:
            e += cavity.getEnergy()
        return e

    def getMinGradients(self):
        """

        """
        min_grads = []
        for cavity in self.cavities:
            min_grads.append(cavity.min_gset)
        return np.array(min_grads)

    def getMaxGradients(self):
        """

        """
        max_grads = []
        for cavity in self.cavities:
            max_grads.append(cavity.max_gset_to_use)
        return np.array(max_grads)

    def reset(self):
        for cavity in self.cavities:
            cavity.reset()

    def getEnergyConstraint(self):
        return self.energyConstraint

    def getEnergyMargin(self):
        return self.energyMargin

    def updateGradients(self, delta):
        """

        """
        # print("Updating gradients with: ", delta)
        new_grads = self.getGradients() + delta
        self.setGradients(new_grads)

    def isTrip(self):
        for cavity in self.cavities:
            if cavity.chargeFraction >= 1:
                return True
        return False

    def printChargeFraction(self):
        c = []
        for cavity in self.cavities:
            c.append(cavity.chargeFraction)
        print(c)

class cebaf_env_v0(gym.Env):
    def __init__(self, path_cavity_data, linac="North", trackTime=False, max_steps=100):
        self.linac = digitalTwin(path_cavity_data, linac)
        self.nCavities = len(self.linac.list_cavities())
        self.trackTime = trackTime
        self.action_space = spaces.Box(
            low=-0.1,
            high=0.1,
            shape=(self.nCavities,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=self.linac.getMinGradients(),
            high=self.linac.getMaxGradients(),
            shape=(self.nCavities,),
            dtype=np.float32
        )

        self.states = self.linac.getGradients()
        self.beta, self.gamma = 0.1,50.0
        self.max_steps = max_steps
        self.counter = 0

    def _computeReward(self):
        return - self.beta * self.linac.getRFHeat() \
               - self.gamma * self.linac.getTripRates() \
               # - self.alpha * max((abs(self.linac.getEnergyGain() - self.linac.getEnergyConstraint()) - self.linac.getEnergyMargin()), 0.) \
            # + self.counter * 10

    def _takeAction(self, action):
        self.linac.updateGradients(action)

    def step(self, action):
        """

        """
        self._takeAction(action)
        reward = self._computeReward()
        next_state = self.linac.getGradients()
        done = False
        if self.trackTime == True:
            done = self.linac.isTrip()
            if done == True:
                self.linac.printChargeFraction()

        if abs(self.linac.getEnergyGain() - self.linac.getEnergyConstraint()) >= self.linac.getEnergyMargin():
            done = True
            # print("Energy Gain: ", self.linac.getEnergyGain())
            # print("Number of Steps: ", self.counter+1)
            # print("Gradients: ", self.linac.getGradients())
            reward = -10000

        self.counter += 1
        if self.counter >= self.max_steps:
            done = True


        return next_state, reward, done, {}

    def reset(self):
        self.linac.reset()
        self.counter = 0
        return self.linac.getGradients()

    def getTripRates(self):
        return self.linac.getTripRates()
    def getRFHeat(self):
        return self.linac.getRFHeat()

# dt = digitalTwin("~/LDRD/cavity_table.pkl")
# # dt.list_cavities()
# print("RF heat: ", dt.getRFHeat())
# print("Trip Rate: ", dt.getTripRates())
# print("Energy gain: ", dt.getEnergyGain(), " MeV")
# dt.setGradients([13.]*200)
# print("RF heat: ", dt.getRFHeat())
# print("Trip Rate: ", dt.getTripRates())
# print("Energy gain: ", dt.getEnergyGain(), " MeV")
# tmp = np.array([2]*418).astype(np.float32)
# dt.setGradients(tmp)
# print("RF heat: ", dt.getRFHeat())
# print("Trip Rate: ", 3600*dt.getTripRates()*1e2)
# tmp = np.array([3]*418).astype(np.float32)
# dt.setGradients(tmp)
# print("RF heat: ", dt.getRFHeat())
# print("Trip Rate: ", 3600*dt.getTripRates()*1e2)[kishan@ifarm1901 models]$