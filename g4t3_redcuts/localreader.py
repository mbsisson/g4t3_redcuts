import numpy as np
from g4t3utils import myreadfile

def read_config_file(log, filename, dictionary):
    code, lines = myreadfile(log, filename)
    if code: return

    bluedatafile = 'NONE'      # blue's costs and actions vector(s)  
    redsettingsfile = 'NONE'   # red settings
    redparamsfile = 'NONE'     # red_foa settings and method parameters
    
    linenum = 0
    # Read lines of configuration file and save options
    while linenum < len(lines):
        thisline = lines[linenum].split()

        if len(thisline) <= 0:   # skip empty lines
            linenum += 1
            continue

        if thisline[0][0] == '#':   # skip commented lines
            linenum += 1
            continue

        if thisline[0] == 'bluedatafile':
            bluedatafile = thisline[1]

        elif thisline[0] == 'redsettingsfile':
            redsettingsfile = thisline[1]

        elif thisline[0] == 'redparamsfile':
            redparamsfile = thisline[1]
            
        elif thisline[0] == 'END':
            break

        else:
            log.joint("Error: Illegal input %s\n"%thisline[0])
            return 1

        linenum += 1

    log.joint("Settings:\n")
    for x in [('bluedatafile', bluedatafile),
              ('redsettingsfile', redsettingsfile),
              ('redparamsfile', redparamsfile)]:
        dictionary[x[0]] = x[1]
        log.joint("  {} {}\n".format(x[0], x[1]))

    if dictionary['bluedatafile'] == 'NONE' or dictionary['redsettingsfile'] == 'NONE' or dictionary['redparamsfile'] == 'NONE':
       log.joint('Error: type or data not provided\n')
       code = 2

    return code




# Reads cost vector and blue's action solution, i.e. cost_times_actions aka blueweights
def read_blue_data(log, filename, dictionary):
    log.joint("Reading blue data file " + filename + "\n")
    code, lines = myreadfile(log, filename)
    if code: return code
    
    thisline = lines[0].split()
    # If given packed weight file, use separate function
    if thisline[0] == "PACKED":
        return read_packed_blueweights(log, filename, dictionary)

    # Read DIM
    DIM = 0
    if len(thisline) != 2:
        log.joint("illegal file structure; first line MUST be of the form DIM ..\n")
        return 1
    DIM = int(thisline[1])

    #log.joint("  DIM = " + str(DIM) + "\n")
    if DIM <= 0:
        log.joint("illegal size input")
        return 1

    dictionary['DIM'] = DIM
  
    # initialize blue data arrays
    costs = np.zeros(DIM)
    actions = np.zeros(DIM)
    cost_times_actions = np.zeros(DIM)
  
    # Look for costs
    foundCOSTS = 0
    linenum = 0
    while not foundCOSTS and linenum < len(lines)-1:
        linenum += 1
        thisline = lines[linenum].split()
        if len(thisline) > 0:
            if thisline[0] == "COSTS" or thisline[0] == "costs":
                foundCOSTS = 1

    # Read costs
    if foundCOSTS:
        i = 0
        while i < DIM and linenum < len(lines)-1:
            linenum += 1
            thisline = lines[linenum].split()
            if len(thisline) > 0:
                cost = float(thisline[0])
                if cost < 0:
                    log.joint("illegal line " + str(linenum+1)
                              + " : cost should be nonnegative; is " + str(cost)
                              + " \n")
                    return 1
                costs[i] = cost
            i += 1

    # Look for actions
    foundACTIONS = 0
    linenum = 0
    while not foundACTIONS and linenum < len(lines)-1:
        linenum += 1
        thisline = lines[linenum].split()
        if len(thisline) > 0:
            if thisline[0] == "ACTIONS" or thisline[0] == "actions":
                foundACTIONS = 1

    # Read actions
    if foundACTIONS:
        i = 0
        while i < DIM and linenum < len(lines)-1:
            linenum += 1
            thisline = lines[linenum].split()
            if len(thisline) > 0:
                action = float(thisline[0])
                if action < 0:
                    log.joint("illegal line " + str(linenum+1)
                              + " : action should be nonnegative; is " + str(action)
                              + " \n")
                    return 1
                actions[i] = action
            i += 1

    # Check if file uses alternate format
    if not foundCOSTS or not foundACTIONS:

        # Look for cost_times_actions
        foundCOST_TIMES_ACTIONS = 0
        linenum = 0
        while not foundCOST_TIMES_ACTIONS and linenum < len(lines)-1:
            linenum += 1
            thisline = lines[linenum].split()
            if len(thisline) > 0:
                if thisline[0] == "COST_TIMES_ACTIONS" or thisline[0] == "cost_times_actions":
                    foundCOST_TIMES_ACTIONS = 1

    # Check if only using specific column
    use_specific_column = False
    column = 0
    linenum += 1
    thisline = lines[linenum].split()
    if thisline[0] == "column":
        use_specific_column = True
        column = int(thisline[1])
    else:
        linenum -= 1
          
    # Read cost_times_actions with no specific column formatting
    if foundCOST_TIMES_ACTIONS and not use_specific_column:
        idx = 0
        while idx < DIM and linenum < len(lines)-1:
            linenum += 1
            thisline = lines[linenum].split()
            for num in range(len(thisline)):
                cost_times_action = float(thisline[num])
                if cost_times_action < 0:
                    log.joint("illegal line " + str(linenum+1)
                              + " : action should be nonnegative; is "
                              + str(cost_times_actions)
                              + " \n")
                    return 1
                cost_times_actions[idx] = cost_times_action
            idx += 1

    # Read cost_times_actions with specific column formatting
    elif foundCOST_TIMES_ACTIONS and use_specific_column:
        idx = 0
        while idx < DIM and linenum < len(lines)-1:
            linenum += 1
            thisline = lines[linenum].split()
            if len(thisline) > 0:
                cost_times_action = float(thisline[column])
                if cost_times_action < 0:
                    log.joint("illegal line " + str(linenum+1)
                              + " : action should be nonnegative; is "
                              + str(cost_times_actions)
                              + " \n")
                    return 1
                cost_times_actions[idx] = cost_times_action
            idx += 1
          
    if (not foundCOSTS or not foundACTIONS) and not foundCOST_TIMES_ACTIONS:
        log.joint("illegal file structure; must provide costs and actions OR cost_times_actions\n")
        return 1

    if foundCOSTS and foundACTIONS:
        dictionary['blueweights'] = costs * actions
    elif foundCOST_TIMES_ACTIONS:
        dictionary['blueweights'] = cost_times_actions

    return code


def read_packed_blueweights(log, filename, dictionary):
    log.joint("Blue weight is PACKED\n")
    code, lines = myreadfile(log, filename)
    if code: return code

    # Read DIM
    DIM = 0
    thisline = lines[1].split()
    if len(thisline) != 2:
        log.joint("illegal file structure; first line MUST be of the form DIM ..\n")
        return 1
    DIM = int(thisline[1])
    
    blueweights = np.zeros(DIM)
    packedindices = np.zeros(DIM)

    i = 0
    linenum = 1
    while i < DIM and linenum < len(lines)-1:
        linenum += 1
        thisline = lines[linenum].split()
        if len(thisline) > 0:
            packedindices[i] = int(thisline[0])
            blueweight = float(thisline[1])
            if blueweight < 0:
                log.joint("illegal line " + str(linenum+1)
                          + " : blue weight should be nonnegative; is "
                          + str(blueweight)
                          + " \n")
                return 1
            blueweights[i] = blueweight
            i += 1

    dictionary['DIM'] = DIM
    dictionary['blueweights'] = blueweights
    dictionary['packedindices'] = packedindices
    
