####################################################################################
# Manages Red solver runs, encapsulating RedManager object which handles master and
# worker process interaction, assigns workers different initial conditions with
# which to run a Red algorithm on. Worker assignment is handled with a worker queue.
# Workers place solution data in a file.
#
# author: Blake Sisson
# Thu.Oct.30.121200.2025
####################################################################################
import time
import numpy as np
from multiprocessing import Process, Lock, RawArray, Value
from g4t3utils import *
from g4t3red_worker import red_worker

MAX_CUT_LENGTH = 50 # Maximum number of coordinates a cut can have (memory constraint).

def _state_red_version(log):
    '''States the current version of the Red code.
    '''
    log.joint('Red Version: <2025.Oct.30.135600@jasper>\n\n')

class RedManager:
    '''
    Manager class for solving the Red model for given Blue solutions.

    Creates and manages worker subprocesses for delegating Red work (in parallel).
    Utilizes Model and Solver classes for selecting a particular Red model and algorithm.
    '''
    def __init__(self, log, DIM, num_workers=10, num_runs=30, loud=0, loud_algo=0, cutFilePathPrefix=''):
        '''
        Parameters:
            log (danoLogger): Log to record run details and output.
                              Worker processes write to separate log files.
            DIM (int): Dimension of the Blue solution.
            num_workers (int): Number of workers processes to use.
            num_runs (int): Number of cuts to produce for each Blue solution.
            loud (int): Whether to output manager-worker interations.
            loud_algo (int): Whether to output details from the algorithm runtime.
        '''
        _state_red_version(log)
        
        self.redstarttime = time.time()
        self.loud = loud
        self.loud_algo = loud_algo
        self.cutFilePathPrefix = cutFilePathPrefix
        self.log = log
        self.DIM = int(DIM)
        self.num_workers = int(num_workers)
        self.num_runs = int(num_runs)

        self.ispacked = 0 # Whether the blueweight input is a packed vector.
        self.blue_scalar = 10 # Scale blue weights by this amount TODO: choose this more intelligently
        
        # Set Red model settings to default values (shared memory).
        self.model_type = Value('i', 0) #log-barrier
        self.budget_local = Value('i', 5)
        self.budget_global = Value('i', 20)
        self.eps_local = Value('f', 1)
        self.eps_global = Value('f', 1)
        self.mu_local = Value('f', 1)
        self.mu_global = Value('f', 1)
        self.project_const = Value('f', .1)
        self.boundary_tol = Value('f', .01)     
        
        # Set algorithm parameters to default values (shared memory).
        self.algo_type = Value('i', 2)  # AdaDelta
        self.tol = Value('f', 1e-3)
        self.max_iter = Value('i', 100)
        self.learn_rate = Value('f', 0.1)
        self.momentum_param = Value('f', 0)
        self.backtrack_factor = Value('f', 0.75)
        self.min_steplen = Value('f', 1e-6)
        self.decay_factor = Value('f', 0.75)
        self.decay_freq = Value('f', 150)
        self.adadelta_const = Value('f', 0.01)
        self.adadelta_decayrate = Value('f', 0.09)
        self.iter_check_freq = Value('i', 20)
        self.cos_angle_thresh = Value('f', -0.87) #cos(5pi / 6)
        self.beta_interval_thresh = Value('f', .000000001)
        self.derphi_zero_thresh = Value('f', .01)
        self.param_list = ['tol', 'max_iter', 'learn_rate', 'momentum_param', 'backtrack_factor',
                           'min_steplen', 'project_const', 'boundary_tol', 'decay_factor', 'decay_freq',
                           'adadelta_const', 'adadelta_decayrate', 'iter_check_freq', 'cos_angle_thresh',
                           'beta_interval_thresh', 'derphi_zero_thresh']
        
        # Set run settings to default values.
        self.worker_sleep_time = 0.005
        self.master_sleep_time = 0.005
        self.shift_size = 1
        self.do_warmstarts = 1

        # Set sliding window sizes and initialization constants (also done in set_redsettings)
        self.support_window_size = 10 * self.budget_global.value // self.budget_local.value
        self.warmstart_window_size = self.budget_global.value // self.budget_local.value
        self.tiny_const = self.budget_global.value / (10 * self.DIM)
        self.warmstart_const = self.budget_local.value / 2.0
        self.runsetting_list = ['worker_sleep_time', 'master_sleep_time', 'shift_size', 'do_warmstarts',
                                'support_window_size', 'warmstart_window_size', 'tiny_const',
                                'warmstart_const']
        
        # Allocate memory for solution data for individual runs.
        self.cuts = {}
        self.solndata = {}
        for cut in range(1, self.num_runs + 1):
            self.cuts[cut] = {}
            self.cuts[cut]['length'] = 0
            self.cuts[cut]['indices'] = np.zeros(MAX_CUT_LENGTH, dtype=int)
            self.cuts[cut]['unpacked_indices'] = np.zeros(MAX_CUT_LENGTH, dtype=int)
            self.cuts[cut]['coefficients'] = np.zeros(MAX_CUT_LENGTH, dtype=np.float32)
            self.cuts[cut]['effort'] = np.zeros(MAX_CUT_LENGTH, dtype=np.float32)
            self.solndata[cut] = {}
            self.solndata[cut]['supported_indices'] = ''
            self.solndata[cut]['filtered_indices'] = ''
            self.solndata[cut]['warmstarted_indices'] = ''

        # Allocate auxiliary arrays for Master's use
        self.argsort_blueweights = np.zeros(DIM, dtype=int)
        self.packedindices = np.zeros(DIM, dtype=int)
        self.initialpt = np.zeros(DIM, dtype=np.float32)
        self.xfilter = np.zeros(DIM, dtype=np.float32)
        self.gfilter = np.zeros(DIM, dtype=np.float32)
                
        # Initialize worker data.
        self.processes = {}
        self.commonlock = Lock()
        self.signalnumber = Value('i', 0)
        self.blueweights_raw = RawArray('f', DIM)
        self.blueweights_np = np.frombuffer(self.blueweights_raw, dtype=np.float32)
        self.workerqueue_raw = RawArray('l', self.num_workers)
        self.workerqueue_np = np.frombuffer(self.workerqueue_raw, dtype=int)
        np.copyto(self.workerqueue_np, np.arange(1, num_workers + 1), casting='unsafe')
        self.wq_frontptr = Value('i', 0) # Points to front of queue (which is now start of the array)
        self.wq_backptr = Value('i', 0) # Points to index after back of queue (ALSO start of the array)
        self.num_queuedworkers = Value('i', self.num_workers)
        self.status = {}
        self.run_id = {}
        self.speciallock = {}
        self.initialpt_raw = {}
        self.initialpt_np = {}
        self.xfilter_raw = {}
        self.xfilter_np = {}
        self.gfilter_raw = {}
        self.gfilter_np = {}
        for w in range(1, self.num_workers + 1):
            self.status[w] = Value('i', 0)
            self.run_id[w] = Value('i', 0)
            self.speciallock[w] = Lock()
            self.initialpt_raw[w] = RawArray('f', self.DIM)
            self.initialpt_np[w] = np.frombuffer(self.initialpt_raw[w], dtype=np.float32)
            self.xfilter_raw[w] = RawArray('f', self.DIM)
            self.xfilter_np[w] = np.frombuffer(self.xfilter_raw[w], dtype=np.float32)
            self.gfilter_raw[w] = RawArray('f', self.DIM)
            self.gfilter_np[w] = np.frombuffer(self.gfilter_raw[w], dtype=np.float32)

    def set_redsettings(self, settings_dict):
        '''Set Red model settings to the given values (if present) in the dictionary settings_dict.'''
        if 'budget_local' in settings_dict: self.budget_local.value = settings_dict['budget_local']
        if 'budget_global' in settings_dict: self.budget_global.value = settings_dict['budget_global']
        if 'eps_local' in settings_dict: self.eps_local.value = settings_dict['eps_local']
        if 'eps_global' in settings_dict: self.eps_global.value = settings_dict['eps_global']
        if 'mu_local' in settings_dict: self.mu_local.value = settings_dict['mu_local']
        if 'mu_global' in settings_dict: self.mu_global.value = settings_dict['mu_global']
        if 'objective_func' in settings_dict: self.model_type = settings_dict['model_type']

        # Update sliding window and initial point constants
        self.support_window_size = 10 * self.budget_global.value // self.budget_local.value
        self.warmstart_window_size = self.budget_global.value // self.budget_local.value
        self.tiny_const = self.budget_global.value / (10 * self.DIM)
        self.warmstart_const = self.budget_local.value / 2.0

    def set_algoparams(self, params_dict):
        '''Set the algorithm parameters to the given values (if present) in the dictionary params_dict.
        '''
        if 'algo_type' in params_dict: self.algo_type = params_dict['algo_type']
        if 'tol' in params_dict: self.tol = params_dict['tol']
        if 'max_iter' in params_dict: self.max_iter = params_dict['max_iter']
        if 'learn_rate' in params_dict: self.max_iter = params_dict['max_iter']
        if 'momentum_param' in params_dict: self.momentum_param = params_dict['momentum_param']
        if 'backtrack_factor' in params_dict: self.backtrack_factor = params_dict['backtrack_factor']
        if 'min_steplen' in params_dict: self.min_steplen = params_dict['min_steplen']
        if 'decay_factor' in params_dict: self.decay_factor = params_dict['decay_factor']
        if 'decay_freq' in params_dict: self.decay_freq = params_dict['decay_freq']
        if 'adadelta_const' in params_dict: self.adadelta_const = params_dict['adadelta_const']
        if 'adadelta_decayrate' in params_dict: self.adadelta_decayrate = params_dict['adadelta_decayrate']
        if 'iter_check_freq' in params_dict: self.iter_check_freq = params_dict['iter_check_freq']
        if 'cos_angle_thresh' in params_dict: self.cos_angle_thresh = params_dict['cos_angle_thresh']
        if 'beta_interval_thresh' in params_dict: self.beta_interval_thresh = params_dict['beta_interval_thresh']
        if 'derphi_zero_thresh' in params_dict: self.derphi_zero_thresh = params_dict['derphi_zero_thresh']
                                       
    def set_runsettings(self, runsettings_to_update_dict):
        '''
        Set run settings to the given values (if present) in the dictionary runsettings_to_update_dict.
        Cannot chenge number of runs (aka number of cuts).
        '''
        for runsetting in self.runsetting_list:
            if runsetting in runsettings_to_update_dict:
                setattr(self, runsetting, runsettings_to_update_dict[runsetting])

    def set_bluescalar(self, scalar):
        '''Setter for bluescalar.
        '''
        self.blue_scalar = scalar
        
    def set_blueweights(self, blueweights, packedindices='None'):
        '''Set Blue weights to the given vector times the scalar blue_scalar, and update the the array of sorted indices.
        '''
        np.multiply(self.blue_scalar, blueweights, out=self.blueweights_np)
        np.copyto(self.argsort_blueweights, np.argsort(-blueweights), casting='unsafe')
        
        self.ispacked = 0 if packedindices == 'None' else 1
        self.packedindices = packedindices
                
    def get_cuts(self):
        '''Return a data structure containing information for each cut and the number of cuts computed.
        '''

        for run_num in range(1, self.num_runs + 1):
            cutfilename = self.cutFilePathPrefix + str(run_num) + ".txt"
            code, lines = myreadfile(self.log, cutfilename)
            if code: self.log.joint("master unable to open file %s\n"%(cutfilename))
            
            thiscut = self.cuts[run_num]
            thisline = lines[0].split()
            thiscut['length'] = cut_length = int(thisline[1])

            for i in range(cut_length):
                thisline = lines[2 + i].split()
                thiscut['indices'][i] = int(thisline[0])
                thiscut['effort'][i] = np.float32(thisline[1])
                thiscut['coefficients'][i] = np.float32(thisline[2])
                if self.ispacked:
                    thiscut['unpacked_indices'][i] = self.packedindices[thiscut['indices'][i]]
            
        return self.cuts, self.num_runs
    
    def log_settings(self):
        '''Log the current Red settings and parameters.
        '''
        self.log.joint("Loud %d\n"%(self.loud))
        self.log.joint("Loud solver %d\n"%(self.loud_algo))
        self.log.joint("DIM %d\n"%(self.DIM))
        self.log.joint("Blue scalar {}\n".format(self.blue_scalar))

        if self.model_type.value == 0:
            self.log.joint("Red Model: log-barrier\n")

        self.log.joint("Red Model Settings:\n")
        for x in [('budget_local', self.budget_local.value),
                  ('budget_global', self.budget_global.value),
                  ('epsilon local', self.eps_local.value),
                  ('epsilon global', self.eps_global.value),
                  ('mu local', self.mu_local.value),
                  ('mu global', self.mu_global.value),
                  ('projection const', self.project_const.value),
                  ('boundary tol', self.boundary_tol.value)]:
            self.log.joint("  {} {}\n".format(x[0], x[1]))
            
        if self.algo_type.value == 0:
            self.log.joint("Algorithm: First-Order\n")
        elif self.algo_type.value == 1:
            self.log.joint("Algorithm: First-Order with Hueristics\n")
        elif self.algo_type.value == 2:
            self.log.joint("Algorithm: AdaDelta\n")

        self.log.joint("Algorithm Parameters:\n")            
        for x in [('tol', self.tol.value),
                  ('max iter', self.max_iter.value),
                  ('learning rate', self.learn_rate.value),
                  ('momentum parameter', self.momentum_param.value),
                  ('backtracking factor', self.backtrack_factor.value),
                  ('min steplen', self.min_steplen.value),
                  ('decay factor', self.decay_factor.value),
                  ('decay freq', self.decay_freq.value),
                  ('adadelta constant', self.adadelta_const.value),
                  ('adadelta decay rate', self.adadelta_decayrate.value),
                  ('iter check freq', self.iter_check_freq.value),
                  ('cos angle thresh', self.cos_angle_thresh.value),
                  ('beta interval thresh', self.beta_interval_thresh.value),
                  ('derphi zero thresh', self.derphi_zero_thresh.value)]:
            self.log.joint("  {} {}\n".format(x[0], x[1]))

        self.log.joint("Run Settings:\n")
        self.log.joint("  num workers %d\n"%(self.num_workers))
        self.log.joint("  num cuts %d\n"%(self.num_runs))
        self.log.joint("  support window size %d\n"%(self.support_window_size))
        self.log.joint("  initialization constant %f\n"%(self.tiny_const))
        self.log.joint("  shift size %d\n"%(self.shift_size))
        self.log.joint("  warmstarts %d\n"%(self.do_warmstarts))
        if self.do_warmstarts:
            self.log.joint("  warm start window size %d\n"%(self.warmstart_window_size))
            self.log.joint("  warmstart constant %f\n"%(self.warmstart_const))
        self.log.joint("  master sleep time %f\n"%(self.master_sleep_time))
        self.log.joint("  worker sleep time %f\n"%(self.worker_sleep_time))
        self.log.joint("\n")

    def log_solutions(self):
        '''Log the the solution (i.e. cuts) found by the Red algorithm and report information on each run.
        '''
        self.log.joint("--- --- --- CUTs --- --- ---\n")
        self.log.joint("Packed Blue weights: %d\n"%(self.ispacked))
        for run_num in range(1, self.num_runs + 1):
            self.log.joint("CUT #%d\n"%(run_num))

            cutfilename = self.cutFilePathPrefix + str(run_num) + ".txt"
            try:
                cutfile = open(cutfilename, 'r')
            except:
                timestamp = time.time() - self.redstarttime
                self.commonlock.acquire()
                self.log.joint('signal-%d t=%f: master cannot open cut file %s\n'\
                      %(self.signalnumber.value, timestamp, cutfilename))
                self.signalnumber.value += 1
                self.commonlock.release()

            for line in cutfile:
                self.log.joint("  " + line)
            cutfile.close()
            
    def create_workers(self):
        '''Create and start each worker process.'''
        self.log.joint('Creating workers\n')        
        self.log.joint('Starting worker process creation loop\n')
        t0 = time.time()

        # Create each worker.
        for w in range(1, self.num_workers + 1):
            self.processes[w] = Process(target=red_worker,
                                        args=(w, self.DIM,
                                              # Worker data:
                                              self.status[w],
                                              self.run_id[w],
                                              self.signalnumber,
                                              self.speciallock[w],
                                              self.commonlock,
                                              self.worker_sleep_time,
                                              # Worker queue:
                                              self.workerqueue_raw,
                                              self.wq_backptr,
                                              self.num_queuedworkers,
                                              self.num_workers,
                                              # Shared memory for starting conditions: 
                                              self.blueweights_raw,
                                              self.initialpt_raw[w],
                                              self.xfilter_raw[w],
                                              self.gfilter_raw[w],
                                              # Output settings:
                                              self.loud,
                                              self.loud_algo,
                                              self.redstarttime,
                                              self.log.get_filename(),
                                              self.cutFilePathPrefix,
                                              # Red settings:
                                              self.model_type,
                                              self.budget_local,
                                              self.budget_global,
                                              self.eps_local,
                                              self.eps_global,
                                              self.mu_local,
                                              self.mu_global,
                                              self.project_const,
                                              self.boundary_tol,
                                              # Algorithm parameters:
                                              self.algo_type,
                                              self.tol,
                                              self.max_iter,
                                              self.learn_rate,
                                              self.momentum_param,
                                              self.backtrack_factor,
                                              self.min_steplen,
                                              self.decay_factor,
                                              self.decay_freq,
                                              self.adadelta_const,
                                              self.adadelta_decayrate,
                                              self.iter_check_freq,
                                              self.cos_angle_thresh,
                                              self.beta_interval_thresh,
                                              self.derphi_zero_thresh,))
                                              
        t1 = time.time()
        self.log.joint('Finished process creation loop in %g secs\n'%(t1-t0))
        self.log.joint('Starting worker process startup loop\n')
        t2 = time.time()

        # Start each worker.
        for w in range(1, self.num_workers + 1):
            if self.loud:
                timestamp = time.time() - self.redstarttime
                self.commonlock.acquire()
                self.log.joint('signal-%d t=%f: master starting process for worker%d\n'\
                               %(self.signalnumber.value, timestamp, w))
                self.signalnumber.value += 1
                self.commonlock.release()
            self.processes[w].start()

        t3 = time.time()
        self.log.joint('Finished starting worker processes in %g secs\n'%(t3-t2))
        self.log.joint('All workers created and started\n')
        
    def destroy_workers(self):
        '''Kill and rejoin worker processes.'''
        # Order each worker process to quit.
        for w in range(1, self.num_workers + 1):
            self.speciallock[w].acquire()
            self.status[w].value = -1
            self.speciallock[w].release()

        # Wait for each worker process to rejoin.
        for w in range(1, self.num_workers + 1):
            if self.loud:
                timestamp = time.time() - self.redstarttime
                self.commonlock.acquire()
                self.log.joint('signal-%d t=%f: master joining processes for worker%d\n'\
                               %(self.signalnumber.value, timestamp, w))
                self.signalnumber.value += 1
                self.commonlock.release()
            self.processes[w].join()

        self.log.joint('All worker processes destroyed\n')
        
    def _setup_run(self, run_id):
        '''Set initial conditions (initial point and filters) for the given run.'''
        self.initialpt.fill(0)
        self.xfilter.fill(0)
        self.gfilter.fill(0)
        
        # Set endpoints of sliding windows (sorted indices).
        support_window_Lptr = min((run_id - 1) * self.shift_size, self.DIM)
        support_window_Rptr = min(support_window_Lptr + self.support_window_size, self.DIM)
        warmstart_window_Lptr = support_window_Lptr
        warmstart_window_Rptr = min(warmstart_window_Lptr + self.warmstart_window_size, self.DIM)
        
        # Record which blue weights (indices) are filtered out of this run.
        for sorted_idx in range(support_window_Lptr):
            idx = self.argsort_blueweights[sorted_idx]
            self.solndata[run_id]['filtered_indices'] += '%d '%(idx)
            
        # Define initial point, supported by coordinates whose sorted indices are in the support window.
        for sorted_idx in range(support_window_Lptr, support_window_Rptr):
            idx = self.argsort_blueweights[sorted_idx]
            # Exclude any coordinates with zero blue weight from support
            if self.blueweights_np[idx] <= 0: break
            self.initialpt[idx] = self.tiny_const
            self.xfilter[idx] = 1
            self.gfilter[idx] = 1
            self.solndata[run_id]['supported_indices'] += '%d '%(idx)
            
        if self.do_warmstarts:
            # Warm-start coordinates in warmstart window, and record their indices.
            for sorted_idx in range(warmstart_window_Lptr, warmstart_window_Rptr):
                idx = self.argsort_blueweights[sorted_idx]
                # Exclude any coordinates with zero blue weight from being warmstarted
                if self.blueweights_np[idx] <= 0: break
                self.initialpt[idx] = self.warmstart_const
                self.solndata[run_id]['warmstarted_indices'] += '%d '%(idx)
                        
    def run_solver(self, blueweights=None):
        '''
        Computes the specified number of cuts for the given Blue solution.

        Delegates work (i.e. algorithm runs for different initial conditions) to worker processes.
        Uses worker queue to handle job assignemnt.
        '''
        t_start = time.time()
        
        if blueweights is not None:
            self.set_blueweights(blueweights)
            
        self.log.joint('Running red solver\n')
        
        # Track number of runs assigned and remaining.
        assigned_runs = 0
        remaining_runs = self.num_runs

        # Assign runs to workers until all are complete.
        while remaining_runs > 0:
            # Ensure worker queue is not empty
            self.commonlock.acquire()
            n_qw = self.num_queuedworkers.value
            self.commonlock.release()
            if self.loud:
                timestamp = time.time() - self.redstarttime
                self.commonlock.acquire()
                self.log.joint('signal-%d t=%f: master sees %d workers queued\n'%(self.signalnumber.value, timestamp, n_qw))
                self.signalnumber.value += 1
                self.commonlock.release()
            while n_qw == 0:
                self.commonlock.acquire()
                n_qw = self.num_queuedworkers.value
                self.commonlock.release()

            # Dequeque next worker from worker queue
            self.commonlock.acquire()
            w = self.workerqueue_np[self.wq_frontptr.value]
            self.wq_frontptr.value = (self.wq_frontptr.value + 1) % self.num_workers
            self.num_queuedworkers.value -= 1
            self.commonlock.release()
            if self.loud:
                timestamp = time.time() - self.redstarttime
                self.commonlock.acquire()
                self.log.joint('signal-%d t=%f: master dequeues worker%d, now %d workers queued\n'\
                                %(self.signalnumber.value, timestamp, w, self.num_queuedworkers.value))
                self.signalnumber.value += 1
                self.commonlock.release()
                
            remaining_runs -= 1
            assigned_runs += 1
            thisrun_id = assigned_runs

            # Set initial conditions (point and filter) for this run.
            self._setup_run(thisrun_id)

            # Send worker the id and initial conditions for this run, and order worker to start.
            self.speciallock[w].acquire()
            self.run_id[w].value = thisrun_id
            np.copyto(self.initialpt_np[w], self.initialpt, casting='unsafe')
            np.copyto(self.xfilter_np[w], self.xfilter, casting='unsafe')
            np.copyto(self.gfilter_np[w], self.gfilter, casting='unsafe')
            self.status[w].value = 1
            self.speciallock[w].release()

            if self.loud:
                timestamp = time.time() - self.redstarttime
                self.commonlock.acquire()
                self.log.joint('signal-%d t=%f: master assigned run %d to worker%d\n'\
                               %(self.signalnumber.value, timestamp, thisrun_id, w))
                self.signalnumber.value += 1
                self.commonlock.release()

        # All runs assigned.
        if self.loud:
            timestamp = time.time() - self.redstarttime
            self.commonlock.acquire()
            self.log.joint('signal-%d t=%f: master assigned assigned all runs\n'\
                           %(self.signalnumber.value, timestamp))
            self.signalnumber.value += 1
            self.commonlock.release()

        # Wait for all workers to finish (i.e. join worker queue)
        self.commonlock.acquire()
        n_qw = self.num_queuedworkers.value
        self.commonlock.release()
        if self.loud:
            timestamp = time.time() - self.redstarttime
            self.commonlock.acquire()
            self.log.joint('signal-%d t=%f: master sees %d workers queued\n'%(self.signalnumber.value, timestamp, n_qw))
            self.signalnumber.value += 1
            self.commonlock.release()
        
        while n_qw < self.num_workers:
            time.sleep(self.master_sleep_time)
            self.commonlock.acquire()
            n_qw = self.num_queuedworkers.value
            self.commonlock.release()
            if self.loud:
                timestamp = time.time() - self.redstarttime
                self.commonlock.acquire()
                self.log.joint('signal-%d t=%f: master sees %d workers queued\n'%(self.signalnumber.value, timestamp, n_qw))
                self.signalnumber.value += 1
                self.commonlock.release()
            
        t_end = time.time()
        self.log.joint('All runs finished and solutions retrieved: %f secs\n'%(t_end-t_start))
