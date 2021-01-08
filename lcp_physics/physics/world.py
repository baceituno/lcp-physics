import time
# from functools import lru_cache
from argparse import Namespace

import ode
import torch
import pdb
import math

from . import engines as engines_module
from . import contacts as contacts_module
from .utils import Indices, Defaults, cross_2d, get_instance, left_orthogonal
import numpy as np

X, Y = Indices.X, Indices.Y
DIM = Defaults.DIM

class Trajectory(object):
	"""Fingers velocity trajectory"""
	def __init__(self, vel=np.zeros((2,5)), name='TrajNo'):
		# super(Trajectory, self).__init__()
		self.vel = vel
		self.name = name

class Reference(object):
	"""Fingers pose trajectory"""
	def __init__(self, pos=np.zeros((3,5)), name='RefNo'):
		# super(Trajectory, self).__init__()
		self.ref = pos
		self.name = name
		

class World:
	"""A physics simulation world, with bodies and constraints.
	"""
	def __init__(self, bodies, constraints=[], dt=Defaults.DT, engine=Defaults.ENGINE,
				 contact_callback=Defaults.CONTACT, eps=Defaults.EPSILON,
				 tol=Defaults.TOL, fric_dirs=Defaults.FRIC_DIRS,
				 post_stab=Defaults.POST_STABILIZATION, strict_no_penetration=True, facets = []):
		self.contacts_debug = False  # XXX

		# Load classes from string name defined in utils
		self.engine = get_instance(engines_module, engine)
		self.contact_callback = get_instance(contacts_module, contact_callback)
		self.states = []
		self.fingers1 = []
		self.fingers2 = []
		self.times = []
		self.t = 0
		self.t_prev = -1
		self.idx = 0
		self.dt = dt
		self.traj = []
		self.ref = []
		self.eps = eps
		self.tol = tol
		self.fric_dirs = fric_dirs
		self.post_stab = post_stab
		self.gamma = 0.01
		self.applied = True
		self.facets = facets

		self.bodies = bodies
		self.vec_len = len(self.bodies[0].v)

		# XXX Using ODE for broadphase for now
		# self.space = ode.HashSpace()
		# for i, b in enumerate(bodies):
		# 	 b.geom.body = i
		# 	 self.space.add(b.geom)

		self.static_inverse = True
		self.num_constraints = 0
		self.joints = []
		for j in constraints:
			b1, b2 = j.body1, j.body2
			i1 = bodies.index(b1)
			i2 = bodies.index(b2) if b2 else None
			self.joints.append((j, i1, i2))
			self.num_constraints += j.num_constraints
			if not j.static:
				self.static_inverse = False

		M_size = bodies[0].M.size(0)
		self._M = bodies[0].M.new_zeros(M_size * len(bodies), M_size * len(bodies))
		# XXX Better way for diagonal block matrix?
		for i, b in enumerate(bodies):
			self._M[i * M_size:(i + 1) * M_size, i * M_size:(i + 1) * M_size] = b.M

		self.set_v(torch.cat([b.v for b in bodies]))

		self.contacts = None
		self.find_contacts()
		self.strict_no_pen = strict_no_penetration
		# if self.strict_no_pen:
		# 	for b in self.bodies:
		# 		print(f'{b.__class__}: {vars(b)}\n')
		# 	assert all([c[0][3].item() <= self.tol for c in self.contacts]),'Interpenetration at start:\n{}'.format(self.contacts)

	def step(self, fixed_dt=True):
		dt = self.dt
		if fixed_dt:
			end_t = self.t + self.dt
			while self.t < end_t:
				dt = end_t - self.t
				self.step_dt(dt)
		else:
			self.step_dt(dt)

	# @profile
	def step_dt(self, dt):

		# PI contrrol weights
		w1 = 1	# P-term
		w2 = 1 - w1 # I-term

		# gets velocities
		if self.idx >= 0 and self.applied:
			for body in self.bodies:
				for tr in self.traj:
					for ref in self.ref:
						if body.name == tr.name and body.name == ref.name:
							if self.idx < np.shape(tr.vel)[1]:
								vel = w1*tr.vel[:,self.idx]
								vel = torch.cat([vel.new_zeros(1), vel])
								dt_ = math.ceil((self.t+1e-6)*10)/10 - self.t
								if self.idx == np.shape(tr.vel)[1]-1:
									vel += w2*(ref.ref[:,self.idx] - body.p)/dt_
								else:
									vel += w2*(ref.ref[:,self.idx+1] - body.p)/dt_
								body.v = vel

			# # updates velocities
			self.set_v(torch.cat([b.v for b in self.bodies]))
			self.applied = True


		start_p = torch.cat([b.p for b in self.bodies])
		start_rot_joints = [(j[0].rot1, j[0].rot2) for j in self.joints]
		new_v = self.engine.solve_dynamics(self, dt)
		self.set_v(new_v)
		# print('orig')
		# print(new_v)

		if self.idx >= 0 and self.applied:
			for body in self.bodies:
				for tr in self.traj:
					for ref in self.ref:
						if body.name == tr.name and body.name == ref.name:
							if self.idx < np.shape(tr.vel)[1]:
								vel = w1*tr.vel[:,self.idx]
								vel = torch.cat([vel.new_zeros(1), vel])
								dt_ = math.ceil((self.t+1e-6)*10)/10 - self.t
								if self.idx == np.shape(tr.vel)[1]-1:
									vel += w2*(ref.ref[:,self.idx] - body.p)/dt_
								else:
									vel += w2*(ref.ref[:,self.idx+1] - body.p)/dt_
								body.v = vel

			# # updates velocities
			self.set_v(torch.cat([b.v for b in self.bodies]))
			self.applied = True

		while True:
			# try step with current dt
			for body in self.bodies:
				body.move(dt)
			for joint in self.joints:
				joint[0].move(dt)
				joint[0].stabilize()
			self.find_contacts()
			
			if all([c[0][3].item() <= self.tol for c in self.contacts]):
				break
			else:
				break
				# print('refining')
				if not self.strict_no_pen and dt < self.dt / 8:
					# if step becomes too small, just continue
					break
				dt /= 2
				# reset positions to beginning of step
				# XXX Clones necessary?
				self.set_p(start_p.clone())
				for j, c in zip(self.joints, start_rot_joints):
					j[0].rot1 = c[0].clone()
					j[0].update_pos()

		self.correct_penetrations()

		if self.post_stab:
			tmp_v = self.v
			dp = self.engine.post_stabilization(self).squeeze(0)
			dp /= 2 # XXX Why 1/2 factor?
			# XXX Clean up / Simplify this update?
			self.set_v(dp)
			for body in self.bodies:
				body.move(dt)
			for joint in self.joints:
				joint[0].move(dt)
			# print('s2')
			self.set_v(tmp_v)

			# self.find_contacts()  # XXX Necessary to recheck contacts?
			self.correct_penetrations()
		self.times.append(self.t)

		self.t += dt

	def get_v(self):
		return self.v

	def set_v(self, new_v):
		if np.isnan(new_v).sum().item() > 0:
			new_v[:] = 0.0
		self.v = new_v
		for i, b in enumerate(self.bodies):
			b.v = self.v[i * len(b.v):(i + 1) * len(b.v)]

	def set_p(self, new_p):
		for i, b in enumerate(self.bodies):
			b.set_p(new_p[i * self.vec_len:(i + 1) * self.vec_len])

	def apply_forces(self, t):
		return torch.cat([b.apply_forces(t) for b in self.bodies])


	def correct_penetrations(self):
		# corrects 1
		for c in self.contacts:
			# corrects for penetration in the fingers
			if self.bodies[c[1]].name[0] == 'f':
				if c[0][3].item() > 11:
					pass
					# norm = (c[0][1])/torch.norm(c[0][1])
					# pen = c[0][3].item()

					# corrected_p = self.bodies[c[1]].p + torch.cat([torch.tensor([0.0]).double(), c[0][1]])
					# self.bodies[c[1]].set_p(corrected_p)

			# corrects for penetration with the environment
			if self.bodies[c[1]].name == 'env':
				if c[0][3].item() > self.tol:
					for b in self.bodies:
						if b.name[0] == 'o':
							corrected_p = b.p + torch.cat([torch.tensor([0.0]).double(), torch.tensor([0.0]).double(), -torch.tensor([c[0][3].item()]).double()])
							b.set_p(corrected_p)

		# object frame
		irot = torch.tensor([[math.cos(self.bodies[0].p[0].item()), math.sin(self.bodies[0].p[0].item())],[-math.sin(self.bodies[0].p[0].item()),math.cos(self.bodies[0].p[0].item())]]).view(2,2)
		pos = self.bodies[0].pos.view(2,1).float()

		# finds the fingers
		for b in self.bodies:
			if b.name[0] == 'f':
				inside = False
				# finger in objec frame					
				ray = torch.matmul(irot,torch.tensor([b.p[1],b.p[2]]).view(2,1).float()-pos).view(2,1)
				for f in self.facets:
					# checks is the ray in y+ intercepts the facet (Jordan Polygon Theorem)
					if (ray[0] <= max(f[0][0],f[1][0])) and (ray[0] >= min(f[0][0],f[1][0])):
						if f[0][0] == f[1][0]:
							if (ray[1] <= f[0][1]):
								inside = True - inside
							if (ray[1] <= f[1][1]):
								inside = True - inside
						else:
							if (ray[1] <= ((ray[0]-f[1][0])*(f[0][1]-f[1][1])/(f[0][0]-f[1][0]) + f[1][1])):
								inside = True - inside
				if inside:
					# finds closest facet
					dist = 1e6
					for f in self.facets:
						v1 = torch.tensor(f[1].copy()).view(2,1)-torch.tensor(f[0].copy()).view(2,1) # AB
						v2 = ray - torch.tensor(f[0].copy()).view(2,1)
						d1 = torch.matmul(v1.view(1,2),v2)/torch.norm(v1)

						if d1.item() <= 0:
							p1 = torch.tensor(f[0].copy()).view(2,1)
						elif d1.item() >= torch.norm(torch.tensor(f[1].copy()-f[0].copy()).view(2,1)).item():
							p1 = torch.tensor(f[1].copy()).view(2,1)
						else:
							p1 = torch.tensor(f[0].copy()).view(2,1) + d1.view(1,1)*v1/torch.norm(v1)

						if torch.norm(p1 - ray).item() < dist:
							dist = torch.norm(p1 - ray).item()
							correction = (p1 - ray).view(2,1)

					# projects to closes facet and assigns
					b.set_p(b.p + torch.cat([torch.tensor([0.0]), torch.matmul(irot.transpose(0, 1), correction.float()).view(2)]))

	def find_contacts(self):
		# import time
		# start_c1 = time.time()
		self.contacts = []
		# ODE contact detection
		# self.space.collide([self], self.contact_callback)
		# pdb.set_trace()
		for i, b1 in enumerate(self.bodies):
			g1 = Namespace()
			g1.no_contact = b1.no_contact
			g1.body_ref = b1
			g1.body = i
			for j, b2 in enumerate(self.bodies[:i]):
				g2 = Namespace()
				g2.no_contact = b2.no_contact
				g2.body_ref = b2
				g2.body = j
				self.contact_callback([self], g1, g2, self.gamma)

		# end_c1 = time.time()
		# print("time per finding contacts: ")
		# print(end_c1 - start_c1)


	def restitutions(self):
		restitutions = self._M.new_empty(len(self.contacts))
		for i, c in enumerate(self.contacts):
			r1 = self.bodies[c[1]].restitution
			r2 = self.bodies[c[2]].restitution
			restitutions[i] = (r1 + r2) / 2
			# restitutions[i] = math.sqrt(r1 * r2)
		return restitutions

	def M(self):
		return self._M

	def Je(self):
		Je = self._M.new_zeros(self.num_constraints,
							   self.vec_len * len(self.bodies))
		row = 0
		for joint in self.joints:
			J1, J2 = joint[0].J()
			i1 = joint[1]
			i2 = joint[2]
			Je[row:row + J1.size(0),
			i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
			if J2 is not None:
				Je[row:row + J2.size(0),
				i2 * self.vec_len:(i2 + 1) * self.vec_len] = J2
			row += J1.size(0)
		return Je

	def Jc(self):
		Jc = self._M.new_zeros(len(self.contacts), self.vec_len * len(self.bodies))
		for i, contact in enumerate(self.contacts):
			c = contact[0]  # c = (normal, contact_pt_1, contact_pt_2, penetration_dist)
			i1 = contact[1]
			i2 = contact[2]
			J1 = torch.cat([cross_2d(c[1], c[0]).reshape(1, 1),
							c[0].unsqueeze(0)], dim=1)
			J2 = -torch.cat([cross_2d(c[2], c[0]).reshape(1, 1),
							 c[0].unsqueeze(0)], dim=1)
			Jc[i, i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
			Jc[i, i2 * self.vec_len:(i2 + 1) * self.vec_len] = J2
		return Jc

	def Jf(self):
		Jf = self._M.new_zeros(len(self.contacts) * self.fric_dirs,
							   self.vec_len * len(self.bodies))
		for i, contact in enumerate(self.contacts):
			c = contact[0]  # c = (normal, contact_pt_1, contact_pt_2)
			dir1 = left_orthogonal(c[0])
			dir2 = -dir1
			i1 = contact[1]  # body 1 index
			i2 = contact[2]  # body 2 index
			J1 = torch.cat([
				torch.cat([cross_2d(c[1], dir1).reshape(1, 1),
						   dir1.unsqueeze(0)], dim=1),
				torch.cat([cross_2d(c[1], dir2).reshape(1, 1),
						   dir2.unsqueeze(0)], dim=1),
			], dim=0)
			J2 = torch.cat([
				torch.cat([cross_2d(c[2], dir1).reshape(1, 1),
						   dir1.unsqueeze(0)], dim=1),
				torch.cat([cross_2d(c[2], dir2).reshape(1, 1),
						   dir2.unsqueeze(0)], dim=1),
			], dim=0)
			Jf[i * self.fric_dirs:(i + 1) * self.fric_dirs,
			i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
			Jf[i * self.fric_dirs:(i + 1) * self.fric_dirs,
			i2 * self.vec_len:(i2 + 1) * self.vec_len] = -J2
		return Jf

	def mu(self):
		return self._memoized_mu(*[(c[1], c[2]) for c in self.contacts])

	def _memoized_mu(self, *contacts):
		# contacts is argument so that cacheing can be implemented at some point
		mu = self._M.new_zeros(len(self.contacts))
		for i, contacts in enumerate(self.contacts):
			i1 = contacts[1]
			i2 = contacts[2]
			# mu[i] = torch.sqrt(self.bodies[i1].fric_coeff * self.bodies[i2].fric_coeff)
			mu[i] = 0.5 * (self.bodies[i1].fric_coeff + self.bodies[i2].fric_coeff)
		return torch.diag(mu)

	def E(self):
		return self._memoized_E(len(self.contacts))

	def _memoized_E(self, num_contacts):
		n = self.fric_dirs * num_contacts
		E = self._M.new_zeros(n, num_contacts)
		for i in range(num_contacts):
			E[i * self.fric_dirs: (i + 1) * self.fric_dirs, i] += 1
		return E

	def save_state(self):
		raise NotImplementedError

	def load_state(self, state_dict):
		raise NotImplementedError

	def reset_engine(self):
		raise NotImplementedError



def run_world(world, animation_dt=None, run_time=10, print_time=True,
			  screen=None, recorder=None, pixels_per_meter=1, traj = [Trajectory()], pos_f = [Reference()]):
	"""Helper function to run a simulation forward once a world is created.
	"""
	import math
	# If in batched mode don't display simulation
	if hasattr(world, 'worlds'):
		screen = None

	if screen is not None:
		import pygame
		background = pygame.Surface(screen.get_size())
		background = background.convert()
		background.fill((255, 255, 255))

	if animation_dt is None:
		animation_dt = float(world.dt)
	elapsed_time = 0.
	prev_frame_time = -animation_dt
	start_time = time.time()

	world.idx = 0
	world.traj = traj
	world.ref = pos_f
	world.t_prev = -10.0
	# world.engine = get_instance(engines_module,'LemkeEngine')
	while world.t < run_time:
		if world.t - world.t_prev >= 0.099:
			# print(world.t)
			for body in world.bodies:
				if body.name == "obj":
					world.states.append(body.p)
				if body.name == "f0":
					world.fingers1.append(body.p)
				if body.name == "f1":
					world.fingers2.append(body.p)
			world.idx += 1
			world.t_prev = round(world.t,2)
			world.applied = True

		world.step()
		# pdb.set_trace()
		
		if screen is not None:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					return

			if elapsed_time - prev_frame_time >= animation_dt or recorder:
				prev_frame_time = elapsed_time

				screen.blit(background, (0, 0))
				update_list = []
				for body in world.bodies:
					update_list += body.draw(screen, pixels_per_meter=pixels_per_meter)
				for joint in world.joints:
					update_list += joint[0].draw(screen, pixels_per_meter=pixels_per_meter)

				# Visualize contact points and normal for debug
				# (Uncomment contacts_debug line in contacts handler):
				if world.contacts_debug:
					for c in world.contacts_debug:
						(normal, p1, p2, penetration), b1, b2 = c
						b1_pos = world.bodies[b1].pos
						b2_pos = world.bodies[b2].pos
						p1 = p1 + b1_pos
						p2 = p2 + b2_pos
						pygame.draw.circle(screen, (0, 255, 0), p1.data.numpy().astype(int), 5)
						pygame.draw.circle(screen, (0, 0, 255), p2.data.numpy().astype(int), 5)
						pygame.draw.line(screen, (0, 255, 0), p1.data.numpy().astype(int),
										 (p1.data.numpy() + normal.data.numpy() * 100).astype(int), 3)

				if not recorder:
					pass
					# Don't refresh screen if recording
					pygame.display.update(update_list)
					pygame.display.flip()  # XXX
				else:	
					recorder.record(world.t)

			elapsed_time = time.time() - start_time
			if not recorder:
				# Adjust frame rate dynamically to keep real time
				wait_time = world.t - elapsed_time
				if wait_time >= 0 and not recorder:
					wait_time += animation_dt  # XXX
					time.sleep(max(wait_time - animation_dt, 0))
				#	animation_dt -= 0.005 * wait_time
				# elif wait_time < 0:
				#	animation_dt += 0.005 * -wait_time
				# elapsed_time = time.time() - start_time

		elapsed_time = time.time() - start_time
		if print_time:
			print('\r ', '{} / {}  {} '.format(float(world.t), float(elapsed_time),
											   1 / animation_dt), end='')
