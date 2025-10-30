# Thu.Oct.30.121200.2025
import time
import numpy as np
from g4t3log import danoLogger
from g4t3red_model import ExpModel_LogBarrier
from g4t3red_solver import PureFirstOrder, HueristicFirstOrder, AdaDelta

def red_worker(worker_id, dim,
               # Worker data:
               status, run_id, signalnumber, speciallock, commonlock, worker_sleep_time,
               # Worker queue (shared memory):
               workerqueue_raw, wq_backptr, num_queuedworkers, num_workers,
               # Starting conditions (shared memory):
               blueweights_raw, initialpt_raw, xfilter_raw, gfilter_raw,
               # Output settings:
               loud, loud_algo, redstarttime, logfilename, cutFilePathPrefix,
               # Red settings (shared memory):
               model_type, budget_local, budget_global, eps_local, eps_global, mu_local, mu_global,
               project_const, boundary_tol,
               # Algorithm parameters (shared memory):
               algo_type, tol, max_iter, learn_rate, momentum_param, backtrack_factor, min_steplen, decay_factor, 
               decay_freq, adadelta_const, adadelta_decayrate, iter_check_freq, cos_angle_thresh, 
               beta_interval_thresh, derphi_zero_thresh):
    '''
    Worker process function, wait for instructions from manager then either:
        -1: quit
        0: sleep
        1: run Red algorithm
        2: update parameters

    Stores memory locally, to be updated if asked. Constructs instance of Model and Solver classes.
    '''
    workerlog = danoLogger(cutFilePathPrefix + "WORKER" + str(worker_id) + "_" + logfilename)
    
    # Local status and run id values.
    my_status = 0
    my_run_id = 0

    # Allocate auxiliary arrays in memory.
    initialpt = np.zeros(dim, dtype=np.float32)
    blueweights = np.zeros(dim, dtype=np.float32)
    soln = np.zeros(dim, dtype=np.float32)
    mask = np.zeros(dim, dtype=bool)  # Memory for storing filter masks.
    tmp = np.zeros(dim, dtype=np.float32)  # Memory for storing temporary calculations.
    xfilter = np.zeros(dim, dtype=np.float32)  # Coordinate filter
    gfilter = np.zeros(dim, dtype=np.float32)  # Gradient filter
    
    # Get Numpy array references to raw shared memory
    commonlock.acquire()
    blueweights_np = np.frombuffer(blueweights_raw, dtype=np.float32)
    workerqueue_np = np.frombuffer(workerqueue_raw, dtype=int)
    commonlock.release()
    speciallock.acquire()
    initialpt_np = np.frombuffer(initialpt_raw, dtype=np.float32)
    xfilter_np = np.frombuffer(xfilter_raw, dtype=np.float32)
    gfilter_np = np.frombuffer(gfilter_raw, dtype=np.float32)
    speciallock.release()

    # Store local values for Red settings and algorithm parameters.
    commonlock.acquire()
    _model_type = model_type.value
    my_budget_local = budget_local.value
    my_budget_global = budget_global.value
    my_eps_local = eps_local.value
    my_eps_global = eps_global.value
    my_mu_local = mu_local.value
    my_mu_global = mu_global.value
    _algo_type = algo_type.value
    my_tol = tol.value
    my_max_iter = max_iter.value
    my_learn_rate = learn_rate.value
    my_momentum_param = momentum_param.value
    my_backtrack_factor = backtrack_factor.value
    my_min_steplen = min_steplen.value
    my_project_const = project_const.value
    my_boundary_tol = boundary_tol.value
    my_decay_factor = decay_factor.value
    my_decay_freq = decay_freq.value
    my_adadelta_const = adadelta_const.value
    my_adadelta_decayrate = adadelta_decayrate.value
    my_iter_check_freq = iter_check_freq.value
    my_cos_angle_thresh = cos_angle_thresh.value
    my_beta_interval_thresh = beta_interval_thresh.value
    my_derphi_zero_thresh = derphi_zero_thresh.value
    commonlock.release()
    
    if loud:
        timestamp = time.time() - redstarttime
        commonlock.acquire()
        workerlog.joint('signal-%d t=%f: worker%d started\n'%(signalnumber.value, timestamp, worker_id))
        signalnumber.value += 1
        commonlock.release()

    # Store all algo parameters in one dictionary.
    params = {}
    params['tol'] = my_tol
    params['max_iter'] = my_max_iter
    params['learn_rate'] = my_learn_rate
    params['momentum_param'] = my_momentum_param
    params['backtrack_factor'] = my_backtrack_factor
    params['min_steplen'] = my_min_steplen
    params['decay_factor'] = my_decay_factor
    params['decay_freq'] = my_decay_freq
    params['adadelta_const'] = my_adadelta_const
    params['adadelta_decayrate'] = my_adadelta_decayrate
    params['iter_check_freq'] = my_iter_check_freq
    params['cos_angle_thresh'] = my_cos_angle_thresh
    params['beta_interval_thresh'] = my_beta_interval_thresh
    params['derphi_zero_thresh'] = my_derphi_zero_thresh

    # Construct a Red Model.
    redModel = None
    if _model_type == 0:
        redModel = ExpModel_LogBarrier(workerlog, loud_algo, blueweights, my_budget_local, my_budget_global, my_eps_local,
                                        my_eps_global, my_mu_local, my_mu_global, my_project_const, my_boundary_tol, mask, tmp)
    else:
        # Invalid model type, use default
        timestamp = time.time() - redstarttime
        commonlock.acquire()
        workerlog.joint('signal-%d t=%f: Invalid model type %d, using default.\n'
                        %(signalnumber.value, timestamp, _model_type))
        signalnumber.value += 1
        commonlock.release()

    # Construct a Red Solver.
    redSolver = None
    if _algo_type == 0:
        # Pure first-order
        redSolver = PureFirstOrder(workerlog, loud_algo, redModel, params, initialpt, xfilter, gfilter, soln, mask, tmp)
    elif _algo_type ==1:
        # Pure first-order with huerisitcs
        redSolver = HueristicFirstOrder(workerlog, loud_algo, redModel, params, initialpt, xfilter, gfilter, soln, mask, tmp)
    elif _algo_type == 2:
        # AdaDelta
        redSolver = AdaDelta(workerlog, loud_algo, redModel, params, initialpt, xfilter, gfilter, soln, mask, tmp)
    else:
        # Invalid algo type, use default
        timestamp = time.time() - redstarttime
        commonlock.acquire()
        workerlog.joint('signal-%d t=%f: Invalid algorithm type %d, using default.\n'
                        %(signalnumber.value, timestamp, _algo_type))
        signalnumber.value += 1
        commonlock.release()

    isasleep = False
    notdone = True
    while notdone:
        speciallock.acquire()
        my_status = status.value
        speciallock.release()
        
        if my_status == -1:
            # Worker ordered to quit.
            if loud:
                timestamp = time.time() - redstarttime
                commonlock.acquire()
                workerlog.joint('signal-%d t=%f: worker%d ordered to quit\n'
                                %(signalnumber.value, timestamp, worker_id))
                signalnumber.value += 1
                commonlock.release()
            notdone = False

        elif my_status == 0:
            # No run assigned, worker goes (back) to sleep.
            if loud and not isasleep:
                timestamp = time.time() - redstarttime
                commonlock.acquire()
                workerlog.joint('signal-%d t=%f: worker%d going to sleep\n'%(signalnumber.value, timestamp, worker_id))
                signalnumber.value += 1
                commonlock.release()
                isasleep = True
            time.sleep(worker_sleep_time)
            
        elif my_status == 1:
            isasleep = False
            if loud:
                timestamp = time.time() - redstarttime
                commonlock.acquire()
                workerlog.joint('signal-%d t=%f: worker%d given work\n'%(signalnumber.value, timestamp, worker_id))
                signalnumber.value += 1
                commonlock.release()

            # Read assigned blueweights vector into Red Model.
            commonlock.acquire()
            redModel.set_blueweights(blueweights_np)
            commonlock.release()

            # Read assigned run id and initial condition into Red Solver.
            speciallock.acquire()
            my_run_id = run_id.value
            redSolver.set_initialpt(initialpt_np)
            redSolver.set_filters(xfilter_np, gfilter_np)
            speciallock.release()
            
            if loud:
                timestamp = time.time() - redstarttime
                commonlock.acquire()
                workerlog.joint('signal-%d t=%f: worker%d working on run %d\n'\
                      %(signalnumber.value, timestamp, worker_id, my_run_id))
                signalnumber.value += 1
                commonlock.release()
                
            # Run the Solver.
            objval, niter, backtrack_count, runtime = redSolver.solve()
            n_objevals, n_gradevals, projection_count = redModel.report_statistics()
            
            my_cut = np.nonzero(soln)[0]
            np.exp(soln, out=tmp) #store coefficients in tmp1

            #Check if cut size smaller than expected (0 perhaps), and rerun with bigger blue scalar.
            if len(my_cut) < redModel.budget_global // redModel.budget_local:
                if loud:
                    timestamp = time.time() - redstarttime
                    commonlock.acquire()
                    workerlog.joint('signal-%d t=%f: worker%d ran cut %d in %f secs, got cut: %s. Trying again with bigger blue_scalar\n'\
                                    %(signalnumber.value, timestamp, worker_id, my_run_id, runtime, ','.join(str(x) for x in my_cut)))
                    signalnumber.value += 1
                    commonlock.release()

                np.copyto(mask, (redSolver.initialpt > 0).astype(bool), casting='unsafe')
                b_min = np.min(redModel.blueweights[mask])
                tiny_const_local = np.min(redSolver.initialpt[mask])
                blue_scalar_2 = 2 * redModel.eps_local * redModel.mu_local / ((redModel.budget_local - redModel.mu_local * tiny_const_local) * b_min * np.exp(tiny_const_local))
                np.multiply(blue_scalar_2, redModel.blueweights, out=redModel.blueweights)

                # Reset filters.
                redSolver.set_filters(mask, mask)
                
                # Run the Solver (again)
                objval, niter, backtrack_count, runtime = redSolver.solve()
                n_objevals, n_gradevals, projection_count = redModel.report_statistics()
            
                my_cut = np.nonzero(soln)[0]
                np.exp(soln, out=tmp) #store coefficients in tmp1
            
            # Get cut file
            cutfilename = cutFilePathPrefix + str(my_run_id) + ".txt"
            try:
                cutfile = open(cutfilename, 'w')
            except:
                timestamp = time.time() - redstarttime
                commonlock.acquire()
                workerlog.joint('signal-%d t=%f: worker%d cannot open cut file %s\n'\
                      %(signalnumber.value, timestamp, worker_id, cutfilename))
                signalnumber.value += 1
                commonlock.release()

            # Write cut data to file
            cutfile.write('cut_length %d\n'%(len(my_cut)))
            cutfile.write('cut_indices_effort_coefficients (blue_weights)\n')
            for idx in range(len(my_cut)):
                cut_idx = my_cut[idx]
                cutfile.write('  %d %f %f  (%f)\n'%(cut_idx, soln[cut_idx], tmp[cut_idx], redModel.blueweights[cut_idx]))
            cutfile.write('niter %d\n'%(niter))
            cutfile.write('runtime %f\n'%(runtime))
            cutfile.write('coordsum %f\n'%(np.sum(soln)))
            cutfile.write('objval %f\n'%(objval))
            cutfile.write('obj evals %d\n'%(n_objevals))
            cutfile.write('grad evals %d\n'%(n_gradevals))
            cutfile.write('worker %d\n'%(worker_id))
            cutfile.close()
    
            # Update status: finished with run.
            speciallock.acquire()
            status.value = 0
            speciallock.release()
            
            # Enqueue to worker queue
            commonlock.acquire()
            workerqueue_np[wq_backptr.value] = worker_id
            wq_backptr.value = (wq_backptr.value + 1) % num_workers
            num_queuedworkers.value += 1
            commonlock.release()

            if loud:
                timestamp = time.time() - redstarttime
                commonlock.acquire()
                workerlog.joint('signal-%d t=%f: worker%d finished run %d in %f seconds\n'
                                %(signalnumber.value, timestamp, worker_id, my_run_id, runtime))
                signalnumber.value += 1
                workerlog.joint('signal-%d t=%f: worker%d enqueues, now %d workers queued\n'\
                                %(signalnumber.value, timestamp, worker_id, num_queuedworkers.value))
                signalnumber.value += 1
                commonlock.release()

    if loud:
        timestamp = time.time() - redstarttime
        commonlock.acquire()
        workerlog.joint('signal-%d t=%f: worker%d done\n'%(signalnumber.value, timestamp, worker_id))
        signalnumber.value += 1
        commonlock.release()