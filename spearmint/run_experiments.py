import os
import sys
import subprocess
import shlex
from spearmint.utils.parsing import repeat_output_dir

def main(expt_dir, repeat, repeat_start=0):
    repeat = int(repeat)
    repeat_start = int(repeat_start)

    for i in xrange(repeat_start, repeat):
        print 'Running experiment %d/%d.' % (i+1, repeat)

        # Create output directory if needed and open output file
        output_dir = repeat_output_dir(expt_dir, i)

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        output_filename = os.path.join(output_dir, "main.log")
        output_file = open(output_filename, 'w')

        subprocess.Popen(shlex.split("python main.py %s --repeat=%d" % (expt_dir, i)),
            stdout=output_file, stderr=output_file)


if __name__ == '__main__':
    main(*sys.argv[1:])