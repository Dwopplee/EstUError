#!/usr/bin/env python3

# Avoid needing display if plots aren't being shown
import sys

if "--noninteractive" in sys.argv:
    import matplotlib as mpl

    mpl.use("svg")

import frccontrol as frccnt
import matplotlib.pyplot as plt
import numpy as np


class Elevator(frccnt.System):

    def __init__(self, dt):
        """Elevator subsystem.

        Keyword arguments:
        dt -- time between model/controller updates
        """
        state_labels = [("Position", "m"), ("Velocity", "m/s")]
        u_labels = [("Voltage", "V")]
        self.set_plot_labels(state_labels, u_labels)

        # Number of motors
        self.num_motors = 2.0
        # Elevator carriage mass in kg
        self.m = 6.803886
        # Radius of pulley in meters
        self.r = 0.02762679089
        # Gear ratio
        self.G = 42.0 / 12.0 * 40.0 / 14.0

        self.model = frccnt.models.elevator(
            frccnt.models.MOTOR_CIM, self.num_motors, self.m, self.r, self.G)
        frccnt.System.__init__(self, self.model, -12.0, 12.0, dt)

        q = [0.02, 0.4]
        r = [12.0]
        self.design_dlqr_controller(q, r)
        self.design_two_state_feedforward(q, r)

        self.sysd.A = np.concatenate(
            (
                np.concatenate((self.sysd.A, self.sysd.B), axis=1),
                np.zeros((1, (self.sysd.A.shape[1] + self.sysd.B.shape[1]))),
            ),
            axis=0,
        )

        self.sysd.B = np.concatenate(
            (
                self.sysd.B,
                np.zeros((1, self.sysd.B.shape[1])),
            ),
            axis=0,
        )

        self.sysd.C = np.concatenate(
            (
                self.sysd.C,
                np.zeros((self.sysd.C.shape[0], 1)),
            ),
            axis=1,
        )

        self.K = np.concatenate(
            (
                self.K,
                np.ones((self.K.shape[0], 1)),
            ),
            axis=1,
        )

        self.r = np.concatenate(
            (
                self.r,
                np.zeros((1, self.r.shape[1])),
            ),
            axis=0,
        )

        self.Kff = np.concatenate(
            (
                self.Kff,
                np.zeros((self.Kff.shape[0], 1)),
            ),
            axis=1,
        )

        print(self.sysd.A)
        print(self.sysd.B)
        print(self.sysd.C)
        print(self.K)
        print(self.r)
        print(self.Kff)

        q_pos = 0.05
        q_vel = 1.0
        q_voltage = 10.0
        r_pos = 0.0001
        self.design_kalman_filter([q_pos, q_vel, q_voltage], [r_pos])


def main():
    dt = 0.00505
    elevator = Elevator(dt)
    elevator.export_cpp_coeffs("Elevator", "control/")

    if "--save-plots" in sys.argv or "--noninteractive" not in sys.argv:
        try:
            import slycot

            plt.figure(1)
            elevator.plot_pzmaps()
        except ImportError:  # Slycot unavailable. Can't show pzmaps.
            pass
    if "--save-plots" in sys.argv:
        plt.savefig("elevator_pzmaps.svg")

    # Set up graphing
    l0 = 0.1
    l1 = l0 + 5.0
    l2 = l1 + 0.1
    t = np.arange(0, l2 + 5.0, dt)

    refs = []

    # Generate references for simulation
    for i in range(len(t)):
        if t[i] < l0:
            r = np.matrix([[0.0], [0.0]])
        elif t[i] < l1:
            r = np.matrix([[1.524], [0.0]])
        else:
            r = np.matrix([[0.0], [0.0]])
        refs.append(r)

    if "--save-plots" in sys.argv or "--noninteractive" not in sys.argv:
        plt.figure(2)
        state_rec, ref_rec, u_rec = elevator.generate_time_responses(t, refs)
        elevator.plot_time_responses(t, state_rec, ref_rec, u_rec)
    if "--save-plots" in sys.argv:
        plt.savefig("elevator_response.svg")
    if "--noninteractive" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
