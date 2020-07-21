import sys
import torch
import pygame

from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.constraints import Joint, YConstraint, XConstraint, RotConstraint, TotalConstraint
from lcp_physics.physics.constraints import FixedJoint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse, MDP
from lcp_physics.physics.utils import Defaults, Recorder
from lcp_physics.physics.world import World, run_world_traj, Trajectory
import numpy as np
from numpy import loadtxt

vel = np.array([[1.0,1.0,1.0,1.0,1.0],[-0.0,-0.0,-0.0,-0.0,-0.0]])/10
traj1 = vel.copy() 
traj1[0,:] = 2500*vel[0,:]
traj1[1,:] = -2500*vel[1,:]

traj2 = vel.copy() 
traj2[0,:] = 2500*vel[0,:]
traj2[1,:] = 2500*vel[1,:] 

pygame.init()
screen = pygame.display.set_mode((1000, 1000), pygame.DOUBLEBUF)
screen.set_alpha(None)

polygon = np.array((loadtxt("../DynamicAffordances/data/polygons_1_2f_sq.csv", delimiter=',')))

bodies = []
joints = []
restitution = 0.00 # no impacts in quasi-dynamics
fric_coeff = 0.01
n_pol = int(polygon[0,0])

xr = 500
yr = 500

# adds body based on triangulation
r0 = Hull([xr, yr], [[1, 1], [-1, 1], [-1, -1], [1, -1]],
         restitution=0.00, fric_coeff=0.00, mass = 0.01, name="obj")
bodies.append(r0)

for i in range(n_pol):
	x2 = [polygon[0,1+8*i], -polygon[0,2+8*i]]
	x1 = [polygon[0,3+8*i], -polygon[0,4+8*i]]
	x0 = [polygon[0,5+8*i], -polygon[0,6+8*i]]
	verts = 250*np.array([x0, x1, x2])
	print(verts)
	p0 = np.array([xr + polygon[0,7+8*i], yr - polygon[0,8+8*i]])
	r1 = Hull(p0, verts, restitution=restitution, mass = 0.0001, fric_coeff=1, name="obj_"+str(i))
	print('disp1')
	r1.add_force(MDP(g=100))
	# r1.add_force(Gravity(g=100))
	bodies.append(r1)
	joints += [FixedJoint(r1, r0)]
	r1.add_no_contact(r0)
	r0 = r1

# Manipulators
c = Circle([200, 550], 5, mass = 100000000, vel=(0, 0, 0), restitution=restitution,
            fric_coeff=1, name = "f1")
bodies.append(c)

c = Circle([200, 450], 5, mass = 100000000, vel=(0, 0, 0), restitution=restitution,
            fric_coeff=1, name = "f2")
bodies.append(c)

vel = torch.tensor(traj1)
traj_f = [] 
traj_f.append(Trajectory(vel = vel, name = "f1"))

vel = torch.tensor(traj2)
traj_f.append(Trajectory(vel = vel, name = "f2"))

# Environment
# r = Rect([0, 500, 505], [900, 10],
#          restitution=restitution, fric_coeff=1)
# bodies.append(r)
# joints.append(TotalConstraint(r))

recorder = None
world = World(bodies, joints, dt=0.1)
run_world_traj(world, run_time=5, screen=screen, recorder=recorder, traj=traj_f)
print('\n')
print(world.states)
print('\n')