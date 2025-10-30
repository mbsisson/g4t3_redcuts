# Thu.Sep.25.202900.2025
from abc import ABC, abstractmethod
import numpy as np
import time

class GenericRedSolver(ABC):
    '''Solver for computing Red cuts for given Blue solutions.
    Encapsulates a variety of first-order algorithms to choose from.
    Stores algorithm parameters and auxiliary memory for array computations.
    '''
    def __init__(self, log, loud, redModel, initialpt, xfilter, gfilter, soln, mask, tmp):
        '''
        Parameters:
            log (danoLogger): Log for recording algorithm runtime details.
            loud (int): Whether to record runtime details to log file only.
            redModel (RedModel): The particular Red Model to solve.
            tol (float): Tolerance.
            max_iter (int): Maximum iterations.
            learn_rate (float): Rearning rate.
            momentum_param (float): Momentum parameter.
            backtrack_factor (float): The amount to contract by each iteration of backtracking.
            min_steplen (float): Minimum allowable step length.
            initialpt (ndarray):
            xfilter (ndarray):
            gfilter (ndarray):
            soln (ndarray):
            mask (ndarray):
            tmp (ndarray):
        '''
        self.log = log
        self.loud = loud
        self.redModel = redModel
        self.initialpt = initialpt
        self.xfilter = xfilter
        self.gfilter = gfilter
        self.dim = len(initialpt)
        self.soln = soln
        self.mask = mask
        self.tmp1 = tmp
        
    def set_algo(self, algo_type):
        '''Set the algorithm to use for given algo_type code.
        '''
        if algo_type == 0:
            self.algo = self.firstorder
        elif algo_type == 1:
            self.algo = self.firstorder_hueristic
        elif algo_type == 2:
            self.algo = self.AdaDelta
            
    def set_initialpt(self, initialpt):
        '''Setter for initialpt.
        '''
        np.copyto(self.initialpt, initialpt, casting='unsafe')

    def set_filters(self, xfilter, gfilter):
        '''Setter for the filters.
        '''
        np.copyto(self.xfilter, xfilter, casting='unsafe')
        np.copyto(self.gfilter, gfilter, casting='unsafe')
        
    @abstractmethod
    def set_params(self, params):
        """Implemented by subclasses only."""
        pass
        
    def solve(self):
        '''Run the algorithm.
        '''
        t_start = time.time()
        self.redModel.reset_statistics()
        
        # Memory addresses every algorithm will need.
        f = self.redModel.eval_objective
        gradf = self.redModel.eval_gradient
        x0 = self.initialpt
        xfilter = self.xfilter
        gfilter = self.gfilter
        soln = self.soln

        if self.loud:
            self.log.screen_off()  # Turn screen output off for algorithm.

        objval, niter, backtrack_count = self._run_algorithm(f, gradf, x0, xfilter, gfilter, soln)
        
        if self.loud:
            self.log.screen_on()

        t_end = time.time()
        runtime = t_end - t_start
        return objval, niter, backtrack_count, runtime
    
    @abstractmethod
    def _run_algorithm(self):
        """Implemented by subclasses only."""
        pass

######################################################################################################
# BACKTRACKING METHODS

    def _backtrack(self, x, p, x_cand, x_new, xfilter):
        '''
        Backtrack from point x_new along direction p towards point x until feasible w.r.t barrier.
        Store result (feasible point) in x_new. Return whether projection was needed.
        '''
        status = 0
        steplen = self.backtrack_factor
        
        # Compute candidate iterate: x_cand = x + steplen * p.
        np.add(x, np.multiply(steplen, p, out=self.tmp1), out=x_cand)

        # Repeat until candidate iterate is feasible.
        while not self.redModel.isfeasible_barrier(x_cand):
            # Reduce step length.
            steplen = self.backtrack_factor * steplen

            # If step length is too small, project current candidate.
            if steplen < self.min_steplen:
                self.redModel.project_feasible(x_cand, xfilter)
                status = 1
                break
            
            # Compute next candidate iterate.
            np.add(x, np.multiply(steplen, p, out=self.tmp1), out=x_cand)
            
        np.copyto(x_new, x, casting='unsafe')
        return status

    def _delicate_feasible_backtrack(self, f, gradf, x, p, x_cand, x_new, xfilter, gfilter):
        '''
        Conducts a delicate backtracking search from x_new back to x in the direction p to find a
        feasible point that, if possible, satisfys the Wolfe conditions. The point is stored in x_new.
        If unable to find a feasible point, projection is used. The method returns whether projection
        was required. If unable to satisfy Wolfe condiitons, the best point found so far is taken.
        
        It is assumed that the filters have pruned any negative coordinates of x_new and have been
        retroactively applied to x and p, so that there is no need to clip any x candidates.
        '''
        beta_low = 0 #largest beta s.t. corresponding x_cand is feasible and derphi(beta) > 0
        beta_upp = 'None' #smallest beta s.t. corresponding x_cand is feasible and derphi(beta) < 0
        beta_out = 1 #smallest beta s.t. corresponding x_cand is infeasible
        derphi_upp = 'None' #keep track of derphi(beta_upp) for debugging
        
        # Find beta_upp.
        feasinterval_flag = False
        while beta_upp == 'None':
            beta = beta_low + (beta_out - beta_low) / 2.0
            if self.loud: self.log.joint('    beta = %f\n'%(beta))

            np.add(x, np.multiply(beta, p, out=self.tmp1), out=x_cand)
            
            if self.redModel.isfeasible_barrier(x_cand):
                derphi = np.dot(p, gradf(x_cand, gfilter, self.tmp1))

                if derphi > 0:
                    feasinterval_flag = True
                    beta_low = beta
                else:
                    beta_upp = beta
                    derphi_upp = derphi
                    if self.loud:
                        self.log.joint('    beta_upp = %.12f,  derphi_upp = %f\n'%(beta_upp, derphi_upp))
            else:
                beta_out = beta

            if beta_out - beta_low < self.beta_interval_thresh:
                # If beta_low was increased, just used this stepsize.
                if feasinterval_flag:
                    if self.loud: self.log.joint('    Bisection srch failed: unable to find Wolfe interval.\n')
                    np.add(x, np.multiply(beta_low, p, out=self.tmp1), out=x_new)
                else:
                    if self.loud: self.log.joint('    Bisection srch failed: unable to find feasible interval, use projection.\n')
                    self.redModel.project_feasible(x_cand, gfilter)
                    np.copyto(x_new, x_cand, casting='unsafe')
                return 1

        if self.loud:
            self.log.joint('    Found Wolfe interval.\n')
        while beta_upp - beta_low >= self.beta_interval_thresh:
            beta = beta_low + (beta_upp - beta_low) / 2.0
            np.add(x, np.multiply(beta, p, out=self.tmp1), out=x_cand)
            derphi = np.dot(p, gradf(x_cand, gfilter, self.tmp1))

            if self.loud: self.log.joint('    beta = %.12f,  derphi = %f\n'%(beta, derphi))
            
            if abs(derphi) < self.derphi_zero_thresh:
                if self.loud: self.log.joint('    Backtrack Success.\n')
                break
            elif derphi > 0:
                beta_low = beta
            else:
                beta_upp = beta
                derphi_upp = derphi

        np.add(x, np.multiply(beta, p, out=self.tmp1), out=x_new)
        return 0


    def _delicate_backtrack(self, f, gradf, x, p, x_cand, x_new, xfilter, gfilter):
        '''
        Perform delicate backtrack from point x_new towards point x along direction p until a point
        satisfying Wolfe conditions is found. Store result in x_new. Returns 1 if unable to find Wolfe
        point. No attempts to satisfy feasibility.
        '''
        status = 0
        beta_lo = 0
        beta_hi = 1
        
        while beta_hi - beta_lo >= self.beta_interval_thresh:
            beta = beta_lo + (beta_hi - beta_lo) / 2.0
            np.add(x, np.multiply(beta, p, out=self.tmp1), out=x_cand)
            der_phi = np.dot(p, gradf(x_cand, xfilter, self.tmp1))

            if abs(der_phi) < self.derphi_zero_thresh:
                status = 1
                break
            elif der_phi > 0:
                beta_lo = beta
            else:
                beta_hi = beta

        np.multiply(beta, p, out=p)
        return status
        
    ### Logging methods #########################################

    def log_iteration(self, it, f_x, norm_grad_x, sum_x):
        self.log.joint('>%5d|  objval=%.5f   norm_grad=%.5f   sum_x=%.5f\n'%(it, f_x, norm_grad_x, sum_x))

    def log_step(self, x, grad_x, p):
        self.log.joint('  [indice] gradient, step direction, and iterate value:\n')
        self.displayvector(x, grad_x, p)

    def log_infeasible(self):
        self.log.joint('  Left feasible region. Backtrack.\n')

    def log_infeasible_detailed(self, code):
        self.log.joint('  Left feasible region:')
        if code == 1: self.log.joint(' violating local barrier.')
        elif code == 2: self.log.joint(' violating global barrier.')
        else: self.log.joint(' violating local and global barriers.')
        self.log.joint(' Backtrack.\n')
        
    def log_zigzag(self):
        self.log.joint('  zig-zagging detected: take delicate step\n')

    def log_jammed(self):
        self.log.joint('  Coordinates jamming near budget_global/budget_local, likely converged.\n')
        
    def log_fixcoordinate(self, active_idx):
        self.log.joint('  Fixing coordinate: %d\n'%(active_idx))

    def log_results(self, it):
        self.log.joint('Optimization stopped after %d iterations\n'%(it))
        
    def displayvector(self, x, g, p):
        '''Display the nonzero indices and coefficients of vector x
        '''
        thresh = 1e-10
        for i in range(len(x)):
            if abs(p[i]) > thresh:
                self.log.joint("    [{}]   {:.5f}   {:.5f}    {:.5f}\n".format(i, g[i], p[i], x[i]))

    def displayvector_single(self, p):
        '''Display the nonzero indices and coefficients of vector p
        '''
        thresh = 1e-10
        for i in range(len(p)):
            if abs(p[i]) > thresh:
                self.log.joint("    [{}]   {:.5f}\n".format(i, p[i]))
    

###################################################################################
# AdaDelta
###################################################################################
class AdaDelta(GenericRedSolver):
    '''
    '''
    def __init__(self, log, loud, redModel, params, initialpt, xfilter, gfilter, soln, mask, tmp):
        '''
        '''
        super().__init__(log, loud, redModel, initialpt, xfilter, gfilter, soln, mask, tmp)
        self.set_params(params)
        self.x = np.zeros(self.dim, dtype=np.float32)
        self.x_new = np.zeros(self.dim, dtype=np.float32)
        self.x_cand = np.zeros(self.dim, dtype=np.float32)
        self.x_old = np.zeros(self.dim, dtype=np.float32)
        self.grad_x = np.zeros(self.dim, dtype=np.float32)
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.p = np.zeros(self.dim, dtype=np.float32)
        self.p_old = np.zeros(self.dim, dtype=np.float32)
        self.accum_grad = np.zeros(self.dim, dtype=np.float32)
        self.accum_step = np.zeros(self.dim, dtype=np.float32)
        self.tmp2 = np.zeros(self.dim, dtype=np.float32)

    def set_params(self, params):
        self.tol = params['tol']
        self.max_iter = params['max_iter']
        self.learn_rate = params['learn_rate']
        self.momentum_param = params['momentum_param']
        self.backtrack_factor = params['backtrack_factor']
        self.min_steplen = params['min_steplen']
        self.decay_factor = params['decay_factor']
        self.decay_freq = params['decay_freq']
        self.adadelta_const = params['adadelta_const']
        self.adadelta_decayrate = params['adadelta_decayrate']
        self.iter_check_freq = params['iter_check_freq']
        self.cos_angle_thresh = params['cos_angle_thresh']
        self.beta_interval_thresh = params['beta_interval_thresh']
        self.derphi_zero_thresh = params['derphi_zero_thresh']

    def _run_algorithm(self, f, gradf, x0, xfilter, gfilter, soln):
        '''
        First-order ascent using AdaDelta steps.
        Includes backtracking for feasibility and step improvement.

        Sun.Sep.28.122600.2025
        '''
        # Get memory addresses for more readable code.
        x = self.x
        x_new = self.x_new
        grad_x = self.grad_x
        p = self.p # Step (update vector) at current iterate.
        p_old = self.p_old # Previous step.
        accum_grad = self.accum_grad
        accum_step = self.accum_step
        tmp1 = self.tmp1
        tmp2 = self.tmp2

        # Initialization.
        np.copyto(x, x0, casting='unsafe')
        f_x = f(x)
        gradf(x, gfilter, grad_x)
        norm_grad_x = np.linalg.norm(grad_x)
        np.copyto(p_old, grad_x, casting='unsafe')  # Set nonexistent previous to current gradient.
        accum_grad.fill(0)  # Accumulated gradient.
        accum_step.fill(0)  # Accumulated step.
        np.copyto(soln, x, casting='unsafe')  # Best iterate found so far.
        f_best = f_x

        # Store number of backtracks needed.
        backtrack_count = 0

        if self.loud: 
            self.log_iteration(0, f_x, norm_grad_x, np.sum(x))

        for it in range(1, self.max_iter + 1):
            # Accumulate gradient.
            np.add(np.multiply(self.adadelta_decayrate, accum_grad, out=tmp1),
                   np.multiply(1 - self.adadelta_decayrate, np.square(grad_x, out=tmp2), out=tmp2),
                   out=accum_grad)

            # Compute update step.
            np.multiply(np.divide(np.sqrt(np.add(accum_step, self.adadelta_const, out=tmp1), out=tmp1),
                                  np.sqrt(np.add(accum_grad, self.adadelta_const, out=tmp2), out=tmp2),
                                  out=tmp1),
                        grad_x,
                        out=p)
            
            # Zig-zag detection.
            if it % self.iter_check_freq == 0:
                '''
                self.log.joint('  Test for zig-zagging: current and previous update step angle = %f\n'
                               %(np.dot(p, p_old) / (np.linalg.norm(p) * np.linalg.norm(p_old))))
                self.log.joint('  p:\n')
                self.displayvector_single(p)
                self.log.joint('  p_old\n')
                self.displayvector_single(p_old)
                self.log.joint('  dot prod = %f\n'%(np.dot(p, p_old)))
                '''
                if np.dot(p, p_old) < np.linalg.norm(p) * np.linalg.norm(p_old) * self.cos_angle_thresh:
                    if self.loud: 
                        self.log_zigzag()
                    self._delicate_feasible_backtrack(f, gradf, x, p, self.x_cand, x_new, xfilter, gfilter)
                    backtrack_count += 1
                    
            # Log step direction.
            if self.loud:
                self.log_step(x, grad_x, p)
            
            # Compute new iterate: x_new = x + p.
            np.add(x, p, out=x_new)
            np.clip(x_new, 0, None, out=x_new)
            
            # Update filter to include new zero coordinates.
            np.multiply(xfilter, np.greater(x_new, 0, out=self.mask), out=xfilter)
            np.multiply(xfilter, gfilter, out=gfilter)

            # Backtrack if new iterate is infeasible (i.e., undefined for objective function).
            is_infeasible = self.redModel.isinfeasiblebarrier_detailed(x_new)
            if is_infeasible:
                if self.loud:
                    self.log_infeasible_detailed(is_infeasible)

                # Retro actively apply (possibly) new filter to current iterate and update step x and p, so no new zeros are lost.
                np.multiply(x, xfilter, out=x)
                np.multiply(p, xfilter, out=p)
                    
                # Record trouble making indice and fix it if backtracking is successful.
                active_idx = np.argmax(x_new)
                active_coefficient = x_new[active_idx]
                retcode = self._delicate_feasible_backtrack(f, gradf, x, p, self.x_cand, x_new, xfilter, gfilter)
                backtrack_count += 1
                if retcode == 0 and active_coefficient > self.redModel.budget_local / self.redModel.mu_local:
                    gfilter[active_idx] = 0
                    if self.loud: self.log_fixcoordinate(active_idx)

            # Accumulate update steps, using actual step taken (x_new - x as opposed to p).
            np.add(np.multiply(self.adadelta_decayrate, accum_step, out=tmp1),
                   np.multiply(1 - self.adadelta_decayrate,
                               np.square(np.subtract(x_new, x, out=tmp2), out=tmp2),
                               out=tmp2),
                   out=accum_step)

            # Update iterate.
            np.copyto(x, x_new, casting='unsafe')
            f_x = f(x)
            gradf(x, gfilter, grad_x)
            norm_grad_x = np.linalg.norm(grad_x)
            np.copyto(p_old, p, casting='unsafe')

            # Update running best solution if improved.
            if f_x > f_best:
                np.copyto(soln, x, casting='unsafe')
                f_best = f_x

            # Log progress.
            if self.loud:
                self.log_iteration(it, f_x, norm_grad_x, np.sum(x))

            # Convergence check.
            if norm_grad_x < self.tol:
                break

        self.log_results(it)
        return f_best, it, backtrack_count


###################################################################################
# Pure First-Order
###################################################################################
class PureFirstOrder(GenericRedSolver):
    '''
    '''
    def __init__(self, log, loud, redModel, params, initialpt, xfilter, gfilter, soln, mask, tmp):
        '''
        '''
        super().__init__(log, loud, redModel, initialpt, xfilter, gfilter, soln, mask, tmp)
        self.set_params(params)
        self.x = np.zeros(self.dim, dtype=np.float32)
        self.x_new = np.zeros(self.dim, dtype=np.float32)
        self.x_cand = np.zeros(self.dim, dtype=np.float32)
        self.x_old = np.zeros(self.dim, dtype=np.float32)
        self.grad_x = np.zeros(self.dim, dtype=np.float32)
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.p = np.zeros(self.dim, dtype=np.float32)
        self.p_old = np.zeros(self.dim, dtype=np.float32)
        self.tmp2 = np.zeros(self.dim, dtype=np.float32)

    def set_params(self, params):
        self.tol = params['tol']
        self.max_iter = params['max_iter']
        self.learn_rate = params['learn_rate']
        self.momentum_param = params['momentum_param']
        self.backtrack_factor = params['backtrack_factor']
        self.min_steplen = params['min_steplen']
        self.decay_factor = params['decay_factor']
        self.decay_freq = params['decay_freq']

    def _run_algorithm(self, f, gradf, x0, xfilter, gfilter, soln):
        '''
        First-order ascent using normalized gradient.
        Includes backtracking for feasibility.

        Thu.May.01.120900.2025
        '''
        # Get memory addresses for more readable code.
        x = self.x
        x_new = self.x_new
        grad_x = self.grad_x
        p = self.p # Step (update vector) at current iterate.
        p_old = self.p_old # Previous step.

        # Initialization.
        np.copyto(x, x0, casting='unsafe')
        f_x = f(x)
        gradf(x, xfilter, grad_x)
        norm_grad_x = np.linalg.norm(grad_x)
        np.copyto(soln, x, casting='unsafe')  # Best iterate found so far.
        f_best = f_x

        # Store number of backtracks needed.
        backtrack_count = 0

        if self.loud:
            self.log_iteration(0, f_x, norm_grad_x, np.sum(x))

        for it in range(1, self.max_iter + 1):
            # Compute step, p = learn_rate * grad_x/norm_grad_x.
            np.divide(grad_x, norm_grad_x, out=p)
            np.multiply(learn_rate, p, out=p)

            # log step direction.
            if self.loud: self.log_step(x, grad_x, p)

            # Compute new iterate: x_new = x + p.
            np.add(x, p, out=x_new)

            # Update filter to include any new zero coordinates.
            np.multiply(xfilter, np.greater(x_new, 0, out=self.mask), out=xfilter)
            np.multiply(xfilter, gfilter, out=gfilter)
            
            # Backtrack if new iterate is infeasible (i.e., undefined for objective function).
            if not self.redModel.isfeasible_barrier(x_new):
                if self.loud: self.log_infeasible()

                # Retro actively apply (possibly) new filter to current iterate and update step x and p, so new zeros are not lost.
                np.multiply(x, xfilter, out=x)
                np.multiply(p, xfilter, out=p)

                # Check if coordinated jammed.
                if np.count_nonzero(x_new >= self.redModel.budget_local / self.redModel.mu_local) >= self.redModel.budget_global // self.redModel.budget_local:
                    if self.loud: self.log_jammed()
                    break
                
                retcode = self._backtrack(f, gradf, x, p, self.x_cand, x_new, xfilter, gfilter)
                backtrack_count += 1

            # Update iterate.
            np.copyto(x, x_new, casting='unsafe')
            f_x = f(x)
            gradf(x, xfilter, grad_x)
            norm_grad_x = np.linalg.norm(grad_x)
            np.copyto(p_old, p, casting='unsafe')

            # Update running best solution if improved.
            if f_x > f_best:
                np.copyto(soln, x, casting='unsafe')
                f_best = f_x

            # Decay learning rate after set number of iterations.
            if it % self.decay_freq == 0:
                learn_rate = learn_rate * self.decay_factor

            # Log progress.
            if self.loud: self.log_iteration(it, f_x, norm_grad_x, np.sum(x))

            # Convergence check.
            if norm_grad_x < self.tol: break
                
        self.log_results(it)
        return f_best, it, backtrack_count  
    

###################################################################################
# First-Order with Hueristics
###################################################################################
class HueristicFirstOrder(GenericRedSolver):
    '''
    '''
    def __init__(self, log, loud, redModel, params, initialpt, xfilter, gfilter, soln, mask, tmp):
        '''
        '''
        super().__init__(log, loud, redModel, initialpt, xfilter, gfilter, soln, mask, tmp)
        self.set_params(params)
        self.x = np.zeros(self.dim, dtype=np.float32)
        self.x_new = np.zeros(self.dim, dtype=np.float32)
        self.x_cand = np.zeros(self.dim, dtype=np.float32)
        self.x_old = np.zeros(self.dim, dtype=np.float32)
        self.grad_x = np.zeros(self.dim, dtype=np.float32)
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.p = np.zeros(self.dim, dtype=np.float32)
        self.p_old = np.zeros(self.dim, dtype=np.float32)
        self.tmp2 = np.zeros(self.dim, dtype=np.float32)

    def set_params(self, params):
        self.tol = params['tol']
        self.max_iter = params['max_iter']
        self.learn_rate = params['learn_rate']
        self.momentum_param = params['momentum_param']
        self.backtrack_factor = params['backtrack_factor']
        self.min_steplen = params['min_steplen']
        self.iter_check_freq = params['iter_check_freq']
        self.cos_angle_thresh = params['cos_angle_thresh']
        self.beta_interval_thresh = params['beta_interval_thresh']
        self.derphi_zero_thresh = params['derphi_zero_thresh']

    def _run_algorithm(self, f, gradf, x0, xfilter, gfilter, soln):
        '''
        First-order ascent using normalized gradient steps and hurestic line seaching to prevent zigzagging.
        Includes backtracking for feasibility and step improvement.
        
        Wed.Apr.30.114000.2025
        '''
        # Get memory addresses for more readable code.
        x = self.x
        x_new = self.x_new
        grad_x = self.grad_x
        p = self.p  # Step (update vector) at current iterate.
        p_old = self.p_old  # Previous step.

        # Initialize iterate.
        np.copyto(x, x0, casting='unsafe')
        f_x = f(x)
        gradf(x, gfilter, grad_x)
        norm_grad_x = np.linalg.norm(grad_x)
        np.copyto(p_old, grad_x, casting='unsafe')  # Set nonexistent previous step to current gradient 
        np.copyto(soln, x, casting='unsafe')  # Best iterate found so far.
        f_best = f_x

        # Store number of backtracks needed.
        backtrack_count = 0

        if self.loud: 
            self.log_iteration(0, f_x, norm_grad_x, np.sum(x))

        for it in range(1, self.max_iter + 1):
            # Compute step, p = learn_rate * grad_x/norm_grad_x
            np.divide(grad_x, norm_grad_x, out=p)
            np.multiply(self.learn_rate, p, out=p)

            # Zig-zag detection.
            if it % self.iter_check_freq == 0:
                if np.dot(p, p_old) < self.learn_rate**2 * self.cos_angle_thresh:
                    if self.loud: self.log_zigzag()
                    self._delicate_backtrack(f, gradf, x, p, self.x_cand, x_new, xfilter, gfilter)
                    backtrack_count += 1
                    
            # Log step direction
            if self.loud: self.log_step(x, grad_x, p)
            
            # Compute new iterate: x_new = x + p.
            np.add(x, p, out=x_new)
            np.clip(x_new, 0, None, out=x_new)

            # Update filter to include any new zero coordinates.
            np.multiply(xfilter, np.greater(x_new, 0, out=self.mask), out=xfilter)
            np.multiply(xfilter, gfilter, out=gfilter)
            
            # Backtrack if new iterate is infeasible (i.e., undefined for objective function).
            if not self.redModel.isfeasible_barrier(x_new):
                if self.loud: self.log_infeasible()

                # Retro actively apply (possibly) new filter to current iterate and update step x and p, so new zeros are not lost.
                np.multiply(x, xfilter, out=x)
                np.multiply(p, xfilter, out=p)

                # Record trouble making indice and fix it if backtracking is successful.
                active_idx = np.argmax(x_new)
                active_coefficient = x_new[active_idx]
                retcode = self._delicate_feasible_backtrack(f, gradf, x, p, self.x_cand, x_new, xfilter, gfilter)
                backtrack_count += 1
                if retcode == 0 and active_coefficient > self.redModel.budget_local / self.redModel.mu_local:
                    gfilter[active_idx] = 0
                    if self.loud: self.log_fixcoordinate(active_idx)

            # Update iterate.
            np.copyto(x, x_new, casting='unsafe')
            f_x = f(x)
            gradf(x, gfilter, grad_x)
            norm_grad_x = np.linalg.norm(grad_x)
            np.copyto(p_old, p, casting='unsafe')

            # Update running best solution if improved.
            if f_x > f_best:
                np.copyto(soln, x, casting='unsafe')
                f_best = f_x

            # Log progress.
            if self.loud: self.log_iteration(it, f_x, norm_grad_x, np.sum(x))

            # Convergence check.
            if norm_grad_x < self.tol: break
            
        self.log_results(it)
        return f_best, it, backtrack_count