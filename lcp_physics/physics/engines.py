import torch

from lcp_physics.lcp.lcp import LCPFunction
from lcp_physics.lcp.lemkelcp import LemkeLCP


class Engine:
    """Base class for stepping engine."""
    def solve_dynamics(self, world, dt):
        raise NotImplementedError


class PdipmEngine(Engine):
    """Engine that uses the primal dual interior point method LCP solver.
    """
    def __init__(self, max_iter=10):
        self.lcp_solver = LCPFunction
        self.cached_inverse = None
        self.max_iter = max_iter

    # @profile
    def solve_dynamics(self, world, dt):
        t = world.t
        Je = world.Je()
        neq = Je.size(0) if Je.ndimension() > 0 else 0

        f = world.apply_forces(t)
        u = torch.matmul(world.M(), world.get_v()) + dt * f
        if neq > 0:
            u = torch.cat([u, u.new_zeros(neq)])
        if not world.contacts:
            # No contact constraints, no complementarity conditions
            if neq > 0:
                P = torch.cat([torch.cat([world.M(), -Je.t()], dim=1),
                               torch.cat([Je, Je.new_zeros(neq, neq)],
                                         dim=1)])
            else:
                P = world.M()
            if self.cached_inverse is None:
                inv = torch.inverse(P)
                if world.static_inverse:
                    self.cached_inverse = inv
            else:
                inv = self.cached_inverse
            x = torch.matmul(inv, u)  # Kline Eq. 2.41
        else:
            # Solve Mixed LCP (Kline 2.7.2)
            Jc = world.Jc()
            v = torch.matmul(Jc, world.get_v()) * 0*world.restitutions()
            M = world.M().unsqueeze(0)
            if neq > 0:
                b = Je.new_zeros(Je.size(0)).unsqueeze(0)
                Je = Je.unsqueeze(0)
            else:
                b = torch.tensor([])
                Je = torch.tensor([])
            Jc = Jc.unsqueeze(0)
            u = u[:world.M().size(0)].unsqueeze(0)
            v = v.unsqueeze(0)
            E = world.E().unsqueeze(0)
            mu = world.mu().unsqueeze(0)
            Jf = world.Jf().unsqueeze(0)
            G = torch.cat([Jc, Jf,
                           Jf.new_zeros(Jf.size(0), mu.size(1), Jf.size(2))], dim=1)
            F = G.new_zeros(G.size(1), G.size(1)).unsqueeze(0)
            F[:, Jc.size(1):-E.size(2), -E.size(2):] = E
            F[:, -mu.size(1):, :mu.size(2)] = mu
            F[:, -mu.size(1):, mu.size(2):mu.size(2) + E.size(1)] = \
                -E.transpose(1, 2)
            h = torch.cat([v, v.new_zeros(v.size(0), Jf.size(1) + mu.size(1))], 1)
            solver = self.lcp_solver() 
            x = -solver.apply(M, u, G, h, Je, b, F)
        new_v = x[:world.vec_len * len(world.bodies)].squeeze(0)
        return new_v

    def post_stabilization(self, world):
        v = world.get_v()
        M = world.M()
        Je = world.Je()
        Jc = None
        if world.contacts:
            Jc = world.Jc()
        ge = torch.matmul(Je, v)
        gc = None
        if Jc is not None:
            gc = torch.matmul(Jc, v) + torch.matmul(Jc, v) * -0*world.restitutions()

        u = torch.cat([Je.new_zeros(Je.size(1)), ge])
        if Jc is None:
            neq = Je.size(0) if Je.ndimension() > 0 else 0
            if neq > 0:
                P = torch.cat([torch.cat([M, -Je.t()], dim=1),
                               torch.cat([Je, Je.new_zeros(neq, neq)], dim=1)])
            else:
                P = M
            if self.cached_inverse is None:
                inv = torch.inverse(P)
            else:
                inv = self.cached_inverse
            x = torch.matmul(inv, u)
        else:
            v = gc
            Je = Je.unsqueeze(0)
            Jc = Jc.unsqueeze(0)
            h = u[:M.size(0)].unsqueeze(0)
            b = u[M.size(0):].unsqueeze(0)
            M = M.unsqueeze(0)
            v = v.unsqueeze(0)
            F = Jc.new_zeros(Jc.size(1), Jc.size(1)).unsqueeze(0)
            
            solver = self.lcp_solver() 
            x = solver.apply(M, h, Jc, v, Je, b, F)
        dp = -x[:M.size(0)]
        return dp


class LemkeEngine(Engine):
    """Engine that uses the primal dual interior point method LCP solver.
    """
    def __init__(self, max_iter=10):
        # self.lcp_solver = LCPFunction
        self.lcp_solver = LemkeLCP()
        self.cached_inverse = None
        self.max_iter = max_iter

    # @profile
    def solve_dynamics(self, world, dt):
        t = world.t
        Je = world.Je()
        neq = Je.size(0) if Je.ndimension() > 0 else 0

        f = world.apply_forces(t)
        u = torch.matmul(world.M(), world.get_v()) + dt * f
        if neq > 0:
            u = torch.cat([u, u.new_zeros(neq)])
        if not world.contacts:
            # No contact constraints, no complementarity conditions
            if neq > 0:
                P = torch.cat([torch.cat([world.M(), -Je.t()], dim=1),
                               torch.cat([Je, Je.new_zeros(neq, neq)],
                                         dim=1)])
            else:
                P = world.M()
            if self.cached_inverse is None:
                inv = torch.inverse(P)
                if world.static_inverse:
                    self.cached_inverse = inv
            else:
                inv = self.cached_inverse
            x = torch.matmul(inv, u)  # Kline Eq. 2.41
        else:
            M = world.M()
            Jc = -world.Jc()
            Jf = world.Jf()
            Je = world.Je()
            E = world.E()
            mu = world.mu()

            u = torch.cat(((M.float() @ world.get_v().float()) + dt * f.float(), torch.zeros(Je.shape[0])))
            P = torch.cat([torch.cat([M.float(), -Je.t().float()], dim=1),
                           torch.cat([Je.float(), torch.zeros((Je.shape[0], Je.shape[0]))], dim=1)])

            Q = torch.cat((-Jc.t().float(), -Jf.t().float(), torch.zeros((Jc.shape[1], E.shape[1]))), dim=1)
            Q = torch.cat((Q, torch.zeros((Je.shape[0], Q.shape[1]))))

            R = torch.cat((Jc.float(), Jf.float(), torch.zeros((mu.shape[0], Jc.shape[1]))))
            R = torch.cat((R, torch.zeros((R.shape[0], Je.shape[0]))), dim=1)

            S = torch.zeros((Jc.shape[0], Q.shape[1]))
            S = torch.cat(
                (S, torch.cat((torch.zeros((Jf.shape[0], mu.shape[1])), torch.zeros((Jf.shape[0], Jf.shape[0])), E.float()), dim=1)))
            S = torch.cat((S, torch.cat((mu.float(), -E.t().float(), torch.zeros((mu.shape[0], E.shape[1]))), dim=1)))

            v = torch.cat(((Jc.float() @ world.get_v().float()) * -world.restitutions().float(),
                           torch.zeros(Jf.shape[0] + mu.shape[0])))

            _, _, x = self.lcp_solver(P, Q, R, S, u, v)
            x = -x.double()

        new_v = x[:world.vec_len * len(world.bodies)].squeeze(0)
        return new_v

    def post_stabilization(self, world):
        M = world.M()
        Je = world.Je()
        Jc = None
        if world.contacts:
            Jc = world.Jc()
        ge = torch.matmul(Je, world.get_v())
        gc = None
        if Jc is not None:
            gc = torch.matmul(Jc, world.get_v()) + torch.matmul(Jc, world.get_v()) * -world.restitutions()

        u = torch.cat([Je.new_zeros(Je.size(1)), ge])
        if Jc is None:
            neq = Je.size(0) if Je.ndimension() > 0 else 0
            if neq > 0:
                P = torch.cat([torch.cat([M, -Je.t()], dim=1),
                               torch.cat([Je, Je.new_zeros(neq, neq)], dim=1)])
            else:
                P = M
            if self.cached_inverse is None:
                inv = torch.inverse(P)
            else:
                inv = self.cached_inverse
            x = torch.matmul(inv, u)
        else:
            neq = Je.size(0) if Je.ndimension() > 0 else 0
            if neq > 0:
                P = torch.cat([torch.cat([M, -Je.t()], dim=1),
                               torch.cat([Je, Je.new_zeros(neq, neq)], dim=1)])
            else:
                P = M

            Q = torch.cat((-Jc.t(), torch.zeros((Je.shape[0], Jc.shape[0]))))
            R = torch.cat((Jc, torch.zeros((Jc.shape[0], Je.shape[0]))), dim=1)
            S = torch.zeros((R.shape[0], Q.shape[1]))
            v = gc
            _, _, x = self.lcp_solver(P, Q, R, S, u, v)
            x = -x.double()

        dp = -x[:M.size(0)]
        return dp