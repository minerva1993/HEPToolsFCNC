#/usr/bin/bash
source /cvmfs/cms.cern.ch/slc7_amd64_gcc700/cms/cmssw/CMSSW_10_6_9/external/slc7_amd64_gcc700/bin/thisroot.sh
source /cvmfs/cms.cern.ch/slc7_amd64_gcc700/external/gcc/7.0.0/etc/profile.d/init.sh 
source /cvmfs/cms.cern.ch/slc7_amd64_gcc700/external/boost/1.67.0/etc/profile.d/init.sh 

export PATH=$PATH:$HOME/bin
export PATH=$ROOTSYS/bin:$PATH
export LD_Library_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH
