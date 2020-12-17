from subprocess import call
import sys

nargs = len(sys.argv)
if nargs == 4:
  excutable = sys.argv[1]
  target = sys.argv[2]
  inputs = sys.argv[3]
elif nargs == 5:
  excutable = sys.argv[1]
  target = sys.argv[2]
  inputs1 = sys.argv[3]
  inputs2 = sys.argv[4]
  inputs = inputs1 + " " + inputs2

if excutable != "hadd": sys.exit()

call([excutable + " -f " + target + " " +  inputs], shell=True)

#call(["hadd " + target + " " +  inputs], shell=True)
