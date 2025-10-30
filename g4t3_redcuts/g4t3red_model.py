# Thu.Sep.25.193800.2025
import numpy as np


class GenericRedModel:
    '''Model for Red's objective function. Stores Red settings and Blue weights.
    '''
    def __init__(self, log, loud, blueweights, budget_local, budget_global, eps_local, eps_global, mu_local, mu_global,
                 project_const, boundary_tol, mask, tmp):
        '''
        Parameters:
            log (danoLogger): Log to record objective values.
            loud (int): Whether to log record the objective values.
            blueweights (array): Blue solution defining the Red objective.
            budget_local (float): Local Red budget.
            budget_global (float): Global Red budget.
            eps_local (float): Scalar for local barrier term.
            eps_global (float): Scalar for global barrier term.
            mu_local (float): Scalar inside local log barrier.
            mu_global (float): Scalar inside global log barrier.
            project_const (float): Constant used for projecting points into feasible region.
            boundary_tol (float): Tolerance for a points proximity to boundary.
            mask (ndarray): Memory for filter masks.
            tmp1 (ndarray): Memory for temporary calculations.
        '''
        self.log = log
        self.loud = loud
        self.dim = len(blueweights)
        self.blueweights = blueweights
        self.set_redsettings(budget_local, budget_global, eps_local, eps_global, mu_local, mu_global, project_const, boundary_tol)
        self.mask = mask
        self.tmp1 = tmp
        self.ones = np.ones(self.dim, dtype=np.float32)  # Array of ones.
        self.reset_statistics()

    def set_blueweights(self, blueweights):
        '''Setter for blueweights.
        '''
        np.copyto(self.blueweights, blueweights, casting='unsafe')

    def set_redsettings(self, budget_local, budget_global, eps_local, eps_global, mu_local, mu_global, project_const, boundary_tol):
        '''Sets Red settings to the given values.
        '''
        self.budget_local = budget_local
        self.budget_global = budget_global
        self.eps_local = eps_local
        self.eps_global = eps_global
        self.mu_local = mu_local
        self.mu_global = mu_global
        self.project_const = project_const
        self.boundary_tol = boundary_tol
        #self.softlog_const = np.exp(budget_local / 2.0)
        #self.pwlogconst = 2 * budget_local
        
    def reset_statistics(self):
        '''Resets the objective and gradient evaluation counters.
        '''
        self.n_objevals = 0
        self.n_gradevals = 0
        self.projection_count = 0

    def report_statistics(self):
        '''Returns the objective and gradient evaluation counters.
        '''
        return self.n_objevals, self.n_gradevals, self.projection_count
    
    def eval_objective(self, x):
        '''Evaluate the objective function at the given point.

        Parameters:
            x (ndarray): The point where the objective function is to be evaluated.
        
        Returns:
            fval (float): The value of the objective function for the given weights at the point x.
        '''
        raise NotImplementedError

    def eval_gradient(self, x, xfilter, g):
        '''Compute the gradient of the objective function at the given point.

        Parameters:
            x (ndarray): Point where the gradient of the objective function is to be computed.
            xfilter (ndarray): Filter for which coordinates of the gradient to ignore.
            g (ndarray): Memory location to store the computed gradient in.
        '''
        raise NotImplementedError
        
    def isfeasible_barrier(self, x):
        '''Determines whether the given point x is outside the barriers.
        '''
        # Check if x exceeds local-barrier (budget_local / mu_local).
        np.greater_equal(x, self.budget_local / self.mu_local, out=self.mask)
        if self.mask.any(): 
            return False
        
        # Check if x exceeds global-barrier (budget_global / mu_global).
        if self.budget_global <= self.mu_global * np.sum(x):
            return False
        
        return True

    def isinfeasiblebarrier_detailed(self, x):
        '''Determines which of the barriers (local or global), if any, the given point x is outside.
        '''
        local_flag = 0
        global_flag = 0

        # Check if x exceeds local-barrier (budget_local / mu_local).
        np.greater_equal(x, self.budget_local / self.mu_local, out=self.mask)
        if self.mask.any():
            local_flag = 1

        # Check if x exceeds global-barrier (budget_global / mu_global).
        if self.budget_global <= self.mu_global * np.sum(x):
            global_flag = 2

        return local_flag + global_flag
    
    def isnonnegative(self, x):
        '''Determines if the given point x is nonnegative.
        '''
        np.less(x, 0, out=self.mask)
        if self.mask.any():
            return False
        return True
    
    def isfeasible(self, x):
        '''Determines if the given point x is feasible, i.e. is nonnegative and does not violate barriers.
        '''
        return self.isfeasible_barrier(x) and self.isnonnegative(x)

    def project_feasible(self, x, gfilter):
        '''Project the given point x into the feasible region, i.e. the nonnegative orthant and within the barriers.
        '''
        self.projection_count += 1

        # Determine coordinates where x exceeds local-barrier (budget_local / mu_local).
        np.greater_equal(x, self.budget_local / self.mu_local, out=self.mask)
        # Set those coordinates to within the feasible region (local-barrier - boundary_tol).
        x[self.mask] = self.budget_local / self.mu_local - self.boundary_tol
        
        # Determine coordinates where x is negative (< boundary_tol).
        np.less(x, self.boundary_tol, self.mask)
        # Set those coordinates to zero.
        x[self.mask] = 0.0
        
        # Check if x violates the global barrier (budget_global / mu_global).
        if self.budget_global <= self.mu_global * np.sum(x):
            # Determine coordinates of x where the gradient filter isn't being applied.
            np.greater(gfilter, 0, out=self.mask)
            # Loop while x is within project_const of the global-barrier.
            while self.budget_global <= self.mu_global * np.sum(x) + self.project_const:
                # Scale the unfiltered coordinates of x down by (1 - project_const).
                x[self.mask] = (1 - self.project_const) * x[self.mask]


##########################################################################
# Log Barrier
##########################################################################
class ExpModel_LogBarrier(GenericRedModel):
    '''
    '''
    def __init__(self, log, loud, blueweights, budget_local, budget_global, eps_local, eps_global, 
                 mu_local, mu_global, project_const, boundary_tol, mask, tmp):
        '''
        '''
        super().__init__(log, loud, blueweights, budget_local, budget_global, eps_local, eps_global, 
                         mu_local, mu_global, project_const, boundary_tol, mask, tmp)
        self.tmp2 = np.zeros(self.dim, dtype=np.float32)  # Memory for storing temporary calculations.

    def eval_objective(self, x):
        '''Evaluate the log-barrier objective function at x, i.e.,
        SUM(blueweights * (e^x - 1)) + eps_local * SUM(log(budget_local - mu_local * x)) + eps_global * log(budget_global - mu_global * SUM(x)).
        '''
        self.n_objevals += 1

        # SUM(blueweights * (e^x - 1))
        np.subtract(np.exp(x, out=self.tmp1), 1.0, out=self.tmp1)
        attacked_weights = np.dot(self.blueweights, self.tmp1)

        # eps_local * SUM(log(budget_local - mu_local * x))
        np.log(np.subtract(self.budget_local, np.multiply(self.mu_local, x, out=self.tmp1), out=self.tmp1), out=self.tmp1)
        local_barrier = self.eps_local * np.sum(self.tmp1)

        # eps_global * log(budget_global - mu_global * SUM(x))
        global_barrier = self.eps_global * np.log(self.budget_global - (self.mu_global * np.sum(x)))

        if self.loud:
            self.log.joint("  func eval #%d:  expterm=%.6f  localbarr=%.6f  globalbarr=%.6f\n"
                           %(self.n_objevals, attacked_weights, local_barrier, global_barrier))
            
        return attacked_weights + local_barrier + global_barrier

    def eval_gradient(self, x, gfilter, g):
        '''Compute the gradient of log-barrier objective function at x, i.e.,
        (blueweights * e^x) - (eps_local*mu_local / (budget_local - mu_local * x)) - (eps_global*mu_global / (budget_global - mu_global * SUM(x)) * ones).
        '''
        self.n_gradevals += 1

        # blueweights * e^x  (element-wise)
        np.multiply(self.blueweights, np.exp(x, out=self.tmp1), out=self.tmp1)

        # -eps_local*mu_local / (budget_local - mu_local * x)  (element-wise)
        np.subtract(self.budget_local, np.multiply(self.mu_local, x, out=self.tmp2), out=self.tmp2)
        np.divide(-self.eps_local * self.mu_local, self.tmp2, out=self.tmp2)

        np.add(self.tmp1, self.tmp2, out=self.tmp1)

        # -eps_global*mu_global / (budget_global - mu_global * SUM(x)) * ones
        global_barrier = -self.eps_global * self.mu_local / (self.budget_global - self.mu_global * np.sum(x))
        np.multiply(global_barrier, self.ones, out=self.tmp2)

        np.add(self.tmp1, self.tmp2, out=self.tmp1)

        # Ignore filtered coordinates.
        return np.multiply(self.tmp1, gfilter, out=g)


''' UNIMPLEMENTED MODELS BELOW

    ##########################################################################
    # Soft Barrier
            
    def objective_softbarrier(self, x):
        Evaluate the soft-barrier objective function at x, i.e.,
        SUM(blueweights * (e^x - 1)) - eps_local * SUM(1 / (U - x)^2) - eps_global / (K - SUM(x))^2.
    
        # SUM(blueweights * (e^x - 1))
        np.subtract(np.exp(x, out=self.tmp1), 1.0, out=self.tmp1)
        attacked_weights = np.dot(self.blueweights, self.tmp1)

        # -eps_local * SUM(1 / (U - x)^2)
        np.power(np.subtract(self.U, x, out=self.tmp1), 2, out=self.tmp1)
        np.divide(1.0, self.tmp1, out=self.tmp1)
        local_barrier = -self.eps_local * np.sum(self.tmp1)

        # -eps_global / (K - SUM(x))^2
        global_barrier = -self.eps_global / (self.K - np.sum(x))**2

        
        return attacked_weights + local_barrier + global_barrier

    def gradient_softbarrier(self, x, xfilter, g):
        #Compute gradient of soft-barrier objective function at x, i.e.,
        #(blueweights * e^x) - (2*eps_local / (U - x)^3) - (2*eps_global / (K - SUM(x))^3 * ones).

        # blueweights * e^x  (element-wise)
        np.multiply(self.blueweights, np.exp(x, out=self.tmp1), out=self.tmp1)

        # -2 * eps_local / (U - x)^3  (element-wise)
        np.power(np.subtract(self.U, x, out=self.tmp2), 3, out=self.tmp2)
        np.divide(-2.0 * self.eps_local, self.tmp2, out=self.tmp2)

        np.add(self.tmp1, self.tmp2, out=self.tmp1)

        # -2 * eps_global / (K - SUM(x))^2 * ones
        global_barrier = -2.0 * self.eps_global / (self.K - np.sum(x))**3
        np.multiply(global_barrier, self.ones, out=self.tmp2)

        np.add(self.tmp1, self.tmp2, out=self.tmp1)

        # Ignore filtered coordinates.
        return np.multiply(self.tmp1, xfilter, out=g)
        
    ##########################################################################
    # Piecewise Log Barrier
        
    def objective_pwlogbarrier(self, x):
        # SUM(blueweights * (e^x - 1))
        np.subtract(np.exp(x, out=self.tmp1), 1.0, out=self.tmp1)
        attacked_weights = np.dot(self.blueweights, self.tmp1)

        # eps_local * SUM(log(U - x))
        np.subtract(self.U, x, out=self.tmp1) #tmp <- U - x
        np.less(self.tmp1, 1.0, out=self.mask) #mask <- U-x < 1
        # tmp3 = log(U-x) if U-x<1, else 0
        np.copyto(self.tmp3, np.where(self.mask, np.log(self.tmp1, out=self.tmp2), 0), casting='unsafe')
        local_barrier = self.eps_local * self.pwlogconst * np.sum(self.tmp3)

        # eps_global * log(K - SUM(x))
        global_barrier = 0
        KminusX = self.K - np.sum(x)
        if KminusX < 1:
            global_barrier = self.eps_global * self.pwlogconst * np.log(KminusX)

        if self.loud:
            self.log.joint("                                                      >>func eval:  ")
            self.log.joint("expterm=%.3f  localbarr=%.3f  globalbarr=%.3f\n"%(attacked_weights,
                                                                              local_barrier,
                                                                              global_barrier))

        return attacked_weights + local_barrier + global_barrier

    def gradient_pwlogbarrier(self, x, xfilter, g):
        # blueweights * e^x  (element-wise)
        np.multiply(self.blueweights, np.exp(x, out=self.tmp1), out=self.tmp1)

        # -eps_local / (U - x)  (element-wise)
        np.subtract(self.U, x, out=self.tmp2)
        np.less(self.tmp2, 1.0, out=self.mask)
        np.copyto(self.tmp4,
                  np.where(self.mask,
                           np.divide(-self.eps_local * self.pwlogconst, self.tmp2, out=self.tmp3),
                           0), casting='unsafe')

        np.add(self.tmp1, self.tmp4, out=self.tmp1)

        # -eps_global / (K - SUM(x)) * ones
        global_barrier = 0
        KminusX = self.K - np.sum(x)
        if KminusX < 1:
            global_barrier = -self.eps_global / KminusX
        np.multiply(global_barrier, self.ones, out=self.tmp2)

        np.add(self.tmp1, self.tmp2, out=self.tmp1)

        # Ignore filtered coordinates.
        return np.multiply(self.tmp1, xfilter, out=g)

    ##########################################################################
    # Softlog Barrier
        
    def objective_softlogbarrier(self, x):
        #Evaluate the soft-log barrier objective function at x, i.e.,
        #   SUM(blueweights * (e^x - 1))
        #   + eps_local * SUM(softlog_const * log(1 + e^(x-U)) * log(U - x))
        #   + eps_global * softlog_const * log(1 + e^(x-U)) * log(K - SUM(x)).
        # SUM(blueweights * (e^x - 1))
        np.subtract(np.exp(x, out=self.tmp1), 1.0, out=self.tmp1)
        attacked_weights = np.dot(self.blueweights, self.tmp1)

        # eps_local * softlog_const * SUM(log(1 + e^(x-U)) * log(U - x))
        np.subtract(x, self.U, out=self.tmp1) #tmp = x-U
        np.multiply(-1, self.tmp1, out=self.tmp2) #tmp2 = U-x
        np.exp(self.tmp1, out=self.tmp1)
        np.add(1, self.tmp1, out=self.tmp1)
        np.log(self.tmp1, out=self.tmp1)
        np.log(self.tmp2, out=self.tmp2)
        np.multiply(self.tmp1, self.tmp2, out=self.tmp1)
        local_barrier = self.eps_local * self.softlog_const * np.sum(self.tmp1)

        # eps_global * softlog_const * log(1 + e^(SUM(x)-K)) * log(K - SUM(x))
        sumx = np.sum(x)
        global_barrier = self.eps_global * self.softlog_const \
            * np.log(1 + np.exp(sumx - self.K)) * np.log(self.K - sumx)
        
        return attacked_weights + local_barrier + global_barrier

    def gradient_softlogbarrier(self, x, xfilter, g):
        #Compute gradient of log-barrier objective function at x, i.e.,
        #(blueweights * e^x) ...
        tmp1 = self.tmp1
        tmp2 = self.tmp2
        tmp3 = self.tmp3
        tmp4 = self.tmp4
        tmp5 = self.tmp5
        
        # blueweights * e^x  (element-wise)
        np.multiply(self.blueweights, np.exp(x, out=tmp1), out=tmp1)

        # Compute gradient of local barrier term.
        #tmp2 = U - x
        np.subtract(self.U, x, out=tmp2)
        #tmp3 = e^(x-U)
        np.exp(np.multiply(-1, tmp2, out=tmp3), out=tmp3)
        #tmp4 = 1 + e^(x-U)
        np.add(1, tmp3, out=tmp4)
        # tmp3 = e^(x-U) * log(U-x) * (U-x)
        np.multiply(np.multiply(tmp3,
                                np.log(tmp2, out=tmp5),
                                out=tmp5),
                    tmp2,
                    out=tmp3)
        # tmp2 = (1 + e^(x-U)) * (U-x)
        np.multiply(tmp4, tmp2, out=tmp2)
        # tmp4 = log(1 + e^(x-U)) * (1 + e^(x-U))
        np.multiply(np.log(tmp4, out=tmp5),
                    tmp4,
                    out=tmp4)
        # tmp5 = ([e^(x-U)log(U-x)(U-x)] - [log(1+e^(x-U))(1+e^(x-U))]) / [(1+e^(x-U))(x-U)] 
        np.divide(np.subtract(tmp3,
                              tmp4,
                              out=tmp5),
                  tmp2,
                  out=tmp5)
        # tmp2 = gradient of local barrier
        np.multiply(self.eps_local * self.softlog_const, tmp5, out=tmp2)

        np.add(tmp1, tmp2, out=tmp1)

        # Compute gradient of global barrier term.
        sumx = np.sum(x)
        KminusX = self.K - sumx
        e_XminusK = np.exp(sumx - self.K)
        global_barrier = (e_XminusK*np.log(KminusX)*KminusX - np.log(1 + e_XminusK)*(1 + e_XminusK)) \
            / (1 + e_XminusK)*KminusX
        np.multiply(global_barrier, self.ones, out=tmp2)

        np.add(tmp1, tmp2, out=tmp1)

        # Ignore filtered coordinates.
        return np.multiply(tmp1, xfilter, out=g)

    ##########################################################################
    # No Barrier
        
    def objective_nobarrier(self, x):
        #Evaluate the no-barrier objective function at x, i.e.,
        #SUM(blueweights * (e^x - 1)).
        
        # SUM(blueweights * (e^x - 1))
        np.subtract(np.exp(x, out=self.tmp1), 1.0, out=self.tmp1)
        return np.dot(self.blueweights, self.tmp1)

    def gradient_nobarrier(self, x, xfilter, g):
        #Compute gradient of no-barrier objective function at x, i.e.,
        #blueweights * e^x

        # blueweights * e^x  (element-wise)
        np.multiply(self.blueweights, np.exp(x, out=self.tmp1), out=self.tmp1)

        # Ignore filtered coordinates.
        return np.multiply(self.tmp1, xfilter, out=g)
'''