from subprocess import call
import sys

nargs = len(sys.argv)
if nargs == 4:
  excutable = sys.argv[1]
  target = sys.argv[2]
  inputs = sys.argv[3]
  
  if excutable != "hadd": sys.exit()

  call([excutable + " " + target + " " +  inputs], shell=True)

else: pass

#call(["hadd " + target + " " +  inputs], shell=True)
