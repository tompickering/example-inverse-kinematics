#!/usr/bin/env python3

# example_inverse_kinematics.py

# Dependencies:
# pysdl2
# numpy

# This program implements Jacobian-based inverse kinematics for the case of a chain of rotating arms in 2D space.

# One end of the first arm is fixed in space, and the arm is free to pivot about this point. Each next arm pivots
# about the end of the previous one. The tip of the final arm is referred to as the 'effector'. Our goal is to
# adjust the angles of each arm in an attempt to move the effector to a target location - in our case, the location
# of the mouse cursor.

# We implement this using the Jacobian technique. This is a clever mathematical framework which allows us to determine
# the ideal way to update the entire system of arms - represented as a vector of thetas - as a whole. This carries
# a strict advantage over a simple gradient descent approach, which in its simplest form can only really consider
# one input angle at a time - which is not helpful in our case, because it is often the case that one arm needs to
# swing *away* from the target to allow another arm to reach it more effectively. The Jacobian approach does not
# suffer this issue because it takes the desired direction of our effector and transforms this into a desired direction
# in our entire input space - that is, the very space in which we express our vector of arm angles. As such, it
# considers all of those input dimensions in concert.


import sys
import sdl2.ext

import time
import ctypes

import numpy as np
from math import sin, cos


WIN_SIZE = (600, 400)

STEP_RATE = 0.01


# An arm is just a length (which remains constant) and an angle (which we update)
class Arm:
    def __init__(self, l):
        self.l = l
        self.theta = 0.0

    # "Given that your pivot end is currently here and at this angle - where
    # is your other end?"
    #
    # N.B. The first arm's angle is relative to the world axes - after that,
    # each arm's angle is relative to the previous arm.
    def get_end(self, base, base_theta):
        return (base[0] + self.l * cos(base_theta + self.theta),
                base[1] + self.l * sin(base_theta + self.theta))


# This function simply takes a configuration of arms and tells us the 2D position of
# the effector - i.e. the tip of the final arm.
def forward_kinematics(arms):
    x = y = 0.0
    theta = 0.0
    for arm in arms:
        (x, y) = arm.get_end((x, y), theta)
        theta += arm.theta
    return (x, y)


# Old version hard-coded for the 2D case
#def jacobian(arms):
#    t0 = arms[0].theta
#    t1 = arms[1].theta
#    l0 = arms[0].l
#    l1 = arms[1].l
#    j11 = -l0 * np.sin(t0) - l1 * np.sin(t0 + t1)
#    j12 = -l1 * np.sin(t0 + t1)
#    j21 = l0 * np.cos(t0) + l1 * np.cos(t0 + t1)
#    j22 = l1 * np.cos(t0 + t1)
#    return np.array([[j11, j12], [j21, j22]])


# The Jacobian matrix of a function f tells us the linear stretching of space in the infinitesimal region
# around a particular point in the output space - which can be considered as a linaer approximation of
# space in the immediate locale of that output point.
#
# The function f need not itself be linear - however the Jacobian gives us a way to treat it as if it were,
# for a small region of the output space.
#
# Our specific function transforms a pose vector (a vector of thetas) into a location in 2D space. The arm
# lengths are constants which are baked into this matrix.
#
# It tells us the linear transformation of space which a given combination of thetas create around the effector.
def jacobian(arms):
    n = len(arms)

    # The 2 rows correspond to the 2 dimensions of the output space (i.e. the effector location)
    # The n columns correspond to the n dimentsions of our input pose vector (i.e. the number of thetas)
    J = np.zeros((2, n))

    lengths = [a.l for a in arms]
    thetas = [a.theta for a in arms]

    # Calculate each column of the jacobian - one for each dimension of the input vector
    for j in range(n):
        angle = np.cumsum(thetas)
        sum_sin = sum([lengths[i] * np.sin(np.sum(thetas[:i+1])) for i in range(j, n)])
        sum_cos = sum([lengths[i] * np.cos(np.sum(thetas[:i+1])) for i in range(j, n)])
        J[0, j] = -sum_sin
        J[1, j] = sum_cos

    return J


sdl2.ext.init()

window = sdl2.ext.Window('IK', size=WIN_SIZE)
window.show()

surf = window.get_surface()
ORIGIN = tuple(map(lambda x: int(x/2), WIN_SIZE))

# Arbitrary configuration of arms
arms = [Arm(100), Arm(80), Arm(20), Arm(30), Arm(90)]

mouse = (0, 0)
event = sdl2.SDL_Event()

while True:
    while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
        if event.type == sdl2.SDL_KEYDOWN:
            if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                sdl2.ext.quit()
                sys.exit()
        if event.type == sdl2.SDL_QUIT:
            sdl2.ext.quit()
            sys.exit()
        if event.type == sdl2.SDL_MOUSEMOTION:
            mouse = (event.motion.x, event.motion.y)

    try:
        mouse = sdl2.ext.mouse.mouse_coords()
    except AttributeError as e:
        pass

    # The target position relative to the origin
    t = (mouse[0] - ORIGIN[0],
         mouse[1] - ORIGIN[1])


    # This is an iterative process - at each step, we are reasoning about an infinitesimal region
    # of space and interpreting it as a simple linear transformation of the inputs. However, this
    # is only correct for the current position of the effector - we assume it applies to a small
    # region of space around the effector, however, this is an approximation which becomes less
    # accurate with distance. As such, we apply it only in order to make a *small* update to the
    # arm thetas, and then repeat the process and refresh our linear approximation of space around
    # the *new* position of the effector under our adjusted thetas.
    for _ in range(20):

        # First, we calculate where our effector is
        pos = forward_kinematics(arms)

        # We find the error - that is, vector that would move our effector to our target location
        err = (t[0]-pos[0], t[1]-pos[1])

        # We find the Jacobian - that is, the matrix that tells us how to consider space around the
        # effector as a linear transformation of the input theta values
        J = jacobian(arms)

        # We find the inverse (rather, pseudo-inverse - but it can be thought of the same way!) of the matrix.
        # This allows us to invert the relationship, and transform the linear space around the effector back
        # into input space.
        J_inv = np.linalg.pinv(J)

        # Now we use the dot product to project the error vector - that is, the desired motion of the
        # effector - backwards into input space. That is to say, we transform the desired update of the
        # effector's position in an assumed linear output space backwards into terms of the corresponding
        # desired change in input space.
        #
        # As this is only valid over small distances, we scale the result down via a heuristic STEP_RATE.
        d_theta = STEP_RATE * J_inv.dot(err)

        # Now we simply apply that desired change in input space to our thetas
        for i in range(len(arms)):
            arms[i].theta += d_theta[i]


    sdl2.ext.draw.fill(surf, sdl2.ext.Color(0, 0, 0))

    # Draw each arm in turn
    start = ORIGIN
    base_theta = 0.0
    for arm in arms:
        end = arm.get_end(start, base_theta)
        sdl2.ext.draw.line(surf, sdl2.ext.Color(255, 255, 255), tuple(map(int, (*start, *end))))
        start = end
        base_theta += arm.theta

    window.refresh()

    time.sleep(1.0/60.0)
