import sys
from g4t3log import danoLogger
from g4t3red_datareader import read_red_settings, read_red_params   
from g4t3red_manager import RedManager
from localreader import read_config_file, read_blue_data

##############################################
# Tests g4t3red code (manager, solver, model)
#
# Author: Blake Sisson
##############################################
if __name__ == "__main__":
    if len(sys.argv) < 2: # program name followed by config file
        sys.exit("arguments: configfile [logfile]\n")
        
    configfile = sys.argv[1]

    if len(sys.argv) == 3:
        logfile = sys.argv[2]
    else:
        logfile = 'red.log'

    log = danoLogger(logfile)

    # temporary dict for retireving data from files
    local_dict = {}
    
    # retrieve data/setting files
    retcode = read_config_file(log, configfile, local_dict)
    if retcode: sys.exit("problem reading configfile")
    
    bluedatafile = local_dict['bluedatafile']
    redsettingsfile = local_dict['redsettingsfile']
    redparamsfile = local_dict['redparamsfile']
    
    # retrieves DIM and blueweights (unnecessary for Dan)
    retcode = read_blue_data(log, bluedatafile, local_dict)
    if retcode: sys.exit("problem reading blue data file")

    DIM = local_dict['DIM']
    blueweights = local_dict['blueweights']

    ispacked = 0
    packedindices = 'None'
    if 'packedindices' in local_dict:
        packedindices = local_dict['packedindices']
        ispacked = 1
    
    # Set up complete
    ##########################################################

    # Create Red Manager instance.
    redManager = RedManager(log, DIM, num_workers=1, num_runs=2,
                            loud=1, loud_algo=1)

    ''' Choose red settings '''
    local_dict['budget_local'] = 1
    local_dict['budget_global'] = 1
    local_dict['mu_local'] = 1
    local_dict['eps_local'] = 1
    redManager.set_redsettings(local_dict)

    ''' Choose params '''
    #local_dict['learn_rate'] = .1
    # redManager.set_params(local_dict)
    
    ''' Choose run settings '''
    #local_dict['master_sleep_time'] = .01
    #local_dict['do_warmstarts'] = 0
    #local_dict['warmstart_window_size'] = redManager.warmstart_window_size
    #local_dict['warmstart_const'] = redManager.warmstart_const
    #redManager.set_runsettings(local_dict)

    # Choose blue scalar (amount to scale blue weights by).
    #redManager.set_bluescalar(10)
    
    # Log settings and parameters.
    redManager.log_settings()
    
    # Create worker processes.
    redManager.create_workers()

    # Wait for workers to be created.
    #time.sleep(.1)
    #simplebreak()

    # Update blue weights and run solver.
    redManager.set_blueweights(blueweights, packedindices)
    redManager.run_solver()
    
    redManager.log_solutions()

    cuts, num_cuts = redManager.get_cuts()
    #print(cuts)
    
    redManager.destroy_workers()
    log.joint('program finished\n')
    log.closelog()

    
