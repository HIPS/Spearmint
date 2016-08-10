def print_nlopt_returncode(returncode, print_func):
    if returncode == 1:
        print_func('NLopt: Normal termination')
    elif returncode == 2:
        print_func('NLopt: stopval reached')
    elif returncode == 3:
        print_func('NLopt: ftol rel or abs reached')
    elif returncode == 4:
        print_func('NLopt: xtol rel or abs reached')
    elif returncode == 5:
        print_func('NLopt: maxeval reached')
    elif returncode == 6:
        print_func('NLopt: max time reached')
    elif returncode == -1:
        print_func('NLopt: failure')
    elif returncode == -2:
        print_func('NLopt: invalid arguments')
    elif returncode == -3:
        print_func('NLopt: out of memory')
    elif returncode == -4:
        print_func('NLopt: roundoff limited (result still useful)')
    elif returncode == -5:
        print_func('NLopt: halted or forced termination')
