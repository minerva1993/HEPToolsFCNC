#/usr/bin/bash
export BOOST_ROOT=/cvmfs/cms.cern.ch/slc7_amd64_gcc820/external/boost/1.72.0
export ROOTSYS=/cvmfs/cms.cern.ch/slc7_amd64_gcc820/cms/cmssw/CMSSW_11_2_0_pre1/external/slc7_amd64_gcc820/

source /cvmfs/cms.cern.ch/slc7_amd64_gcc820/external/gcc/8.2.0/etc/profile.d/init.sh
source $ROOTSYS/bin/thisroot.sh
source $BOOST_ROOT/etc/profile.d/init.sh

export PATH=$ROOTSYS/bin:$PATH
export LD_Library_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH
