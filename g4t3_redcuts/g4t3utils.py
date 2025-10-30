################################################################
# A collection of utility functions for breaking, file handling,
# and printing written by Dan Bienstock
################################################################

import sys
import os.path

def simplebreak():
    stuff = input('break > ')
    if stuff == 'x' or stuff == 'q':
        sys.exit("bye")

def breakexit(foo):
    stuff = input("("+foo+") break> ")
    if stuff == 'x' or stuff == 'q':
        sys.exit("bye")

def returnbreakexit(foo):
    stuff = input("("+foo+") break> ")
    if stuff == 'x' or stuff == 'q':
        return 1
    else:
        return 0

def askfordouble(foo):
    stuff = input("("+foo+") input value> ")
    value = float(stuff)
    return value

def checkstop(log, name):
    if os.path.isfile(name) :
        breakexit("stop?")

def myreadfile(log, filename):
    code = 0
    lines = None
    try:
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
    except:
        log.joint("cannot open file " + filename + "\n")
        code = 1

    return code, lines
        
def myprintfile(log, mfilename, casefilelines):
    try:
        f = open(mfilename, "w")
    except:
        log.stateandquit("cannot open file " + mfilename + "\n")

    for line in casefilelines.values():
        f.write(line)

    f.close()

    return 1

def myshownparray(log, array, arraylen, arrayname):
    log.joint("%s: \n" %(arrayname))

    k = 0
    for h in range(arraylen):
        log.joint('%.3e ' %(array[h]))
        k += 1
        if k == 10:
            log.joint("\n")
            k = 0
    if k>0: log.joint("\n")
