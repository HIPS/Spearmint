import sys
from spearmint import cleanup

def main(expt_dir, repeat, repeat_start=0):

    raw_input("Are you sure? Press enter to continue.")

    repeat = int(repeat)
    repeat_start = int(repeat_start)

    for i in xrange(repeat_start, repeat):
        cleanup.cleanup(expt_dir, i)

if __name__ == '__main__':
    main(*sys.argv[1:])