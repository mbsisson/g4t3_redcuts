###########################################################
# Methods for setting red, foa parameter, and blue settings
# either from file or hardcoded defualt values
#
# Author: Blake Sisson
# Mon.Oct.06.104600.2025
###########################################################
from g4t3utils import *

# Reads red settings from the given file 
def read_red_settings(log, filename, dictionary):
    log.joint('Reading red settings file %s\n' %(filename))
    code, lines = myreadfile(log, filename)
    if code: return code

    # red settings
    U = 5              # local (per arc) attack budget
    K = 20              # global (aggregate) attack budget
    epslocal = 1      # constant for local barrier term
    epsglobal = 1     # constant for global barrier term
    
    linenum = 0
    # Read lines of data file and save options
    while linenum < len(lines):
        thisline = lines[linenum].split()

        if len(thisline) <= 0:   # skip empty lines 
            linenum += 1
            continue

        if thisline[0][0] == '#':   # skip commented lines
            linenum += 1
            continue

        elif thisline[0] == 'U':
            U = int(float(thisline[1]))

        elif thisline[0] == 'K':
            K = int(float(thisline[1]))

        elif thisline[0] == 'eps_local' or thisline[0] == 'epslocal':
            epslocal = int(float(thisline[1]))

        elif thisline[0] == 'eps_global' or thisline[0] == 'epsglobal':
            epsglobal = int(float(thisline[1]))
            
        elif thisline[0] == 'END':
            break

        else:
            log.joint("Error: Illegal input %s\n"%thisline[0])

        linenum += 1

    log.joint("Red Settings:\n")

    for x in [('U', U),
              ('K', K),
              ('epslocal', epslocal),
              ('epsglobal', epsglobal)]:
        dictionary[x[0]] = x[1]
        log.joint("  {} {}\n".format(x[0], x[1]))
  
    return code


# Reads red first order ascent parameters from the given file
def read_red_params(log, filename, dictionary):
    log.joint('Reading red parameter file %s\n' %(filename))
    code, lines = myreadfile(log, filename)
    if code: return code

    # worker settings
    num_workers = 10

    do_warmstarts = 1
  
    # parameters: red objective
    cushion = 0        # boundary cushion/buffer for local attack budget
  
    # parameters: gradient ascent
    learn_rate = 0         # learning rate
    momentum_param = 0      # momentum parameter
    backtrack_factor = 0    # backtracking factor
    min_steplen = 0    # min steplength for backtrack search
    project_const = 0     # projection constant if backtracking fails
    boundary_tol = 0   # distance-to-bounary tolerance for projecting gradient

    # hyperparameters: leanring rate/ momentum decay
    decay_factor = 1
    decay_frequency = 0
  
    # parameters: termination criteria
    tol = 0                                # tolerance
    max_iter = 0                           # maximum iterations
    
    # output
    loud = 0           # whether to record/report iteration status updates
  
    linenum = 0
    # Read lines of data file and save options
    while linenum < len(lines):
        thisline = lines[linenum].split()

        if len(thisline) <= 0:   # skip empty lines 
            linenum += 1
            continue

        if thisline[0][0] == '#':   # skip commented lines
            linenum += 1
            continue

        elif thisline[0] == 'loud':
            loud = int(thisline[1])

        elif thisline[0] == 'num_workers':
            num_workers = int(thisline[1])

        elif thisline[0] == 'do_warmstarts':
            do_warmstarts = int(thisline[1])
            
        elif thisline[0] == 'cushion':
            cushion = float(thisline[1])

        elif thisline[0] == 'l_rate' or thisline[0] == 'learn_rate':
            learn_rate = float(thisline[1])

        elif thisline[0] == 'mom_param' or thisline[0] == 'momentumparam':
            momentum_param = float(thisline[1])

        elif thisline[0] == 'back_factor' or thisline[0] == 'backtrack_factor':
            backtrack_factor = float(thisline[1])

        elif thisline[0] == 'min_steplen':
            min_steplen = float(thisline[1])

        elif thisline[0] == 'proj_const' or thisline[0] == 'project_const':
            project_const = float(thisline[1])

        elif thisline[0] == 'boundary_tol':
            boundary_tol = float(thisline[1])

        elif thisline[0] == 'decay_factor':
            decay_factor = float(thisline[1])

        elif thisline[0] == 'decay_frequency':
            decay_frequency = int(thisline[1])
            
        elif thisline[0] == 'tol':
            tol = float(thisline[1])

        elif thisline[0] == 'max_iter':
            max_iter = int(thisline[1])
            
        elif thisline[0] == 'END':
            break
          
        else:
            log.joint("Error: Illegal input %s\n"%thisline[0])
          
        linenum += 1

    if decayfrequency == 0:
        decayfrequency = max_iter

    for x in [('loud', loud),
              ('num_workers', num_workers),
              ('do_warmstarts', do_warmstarts),
              ('cushion', cushion),
              ('learn_rate', learn_rate),
              ('momentum_param', momentum_param),
              ('backtrack_factor', backtrack_factor),
              ('min_steplen', min_steplen),
              ('project_const', project_const),
              ('boundary_tol', boundary_tol),
              ('decay_factor', decay_factor),
              ('decay_frequency', decay_frequency),
              ('tol', tol),
              ('max_iter', max_iter)]:
        dictionary[x[0]] = x[1]

    
    return code
