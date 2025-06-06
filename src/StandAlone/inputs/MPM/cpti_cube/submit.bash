#!/bin/bash
# --- Input Paramters ---
export PREFIX=$1
export WORK_DIR=`pwd` 
export NPROCS=1
export JOB_NUM=`date +%y%m%d`
export SUB_DIR=${JOB_NUM} 
export TMP_DIR=${WORK_DIR}/${SUB_DIR}

# --- Versions ---
export VERSION=opt
export UDIR=${HOME}/uintah/${VERSION}
export UDAVIZ=${HOME}/csm-users/leavy/Codes/uda2xmf/uda2xmf.py
#export MPIHOME=/usr/cta/CSE/Release/openmpi-1.10.7
#export LD_LIBRARY_PATH=${MPIHOME}/lib:/disk2/libxml2/lib:${LD_LIBRARY_PATH}
# --- Information ---
echo 'Directory:'   ${WORK_DIR} 
echo 'Uintah Version:'  ${UDIR}
echo 'File prefix:'     ${PREFIX}
echo 'Number of Processors:'    ${NPROCS}

# --- Preprocess ---
cd ${WORK_DIR}
mkdir ${TMP_DIR}
#cp ${PREFIX}.* ${TMP_DIR}/.
cp * ${TMP_DIR}/.
cd ${TMP_DIR}
# ---  Particle file splitter ---
# -b for binary to save space, pfs2 and ups mods for 8-bit raw
#$MPIHOME/bin/mpirun -np 1 ${UDIR}/StandAlone/tools/pfs ${PREFIX}.ups
${UDIR}/StandAlone/tools/pfs/pfs ${PREFIX}.ups

# --- Uintah ---
echo 'Started: '`date`
${UDIR}/StandAlone/sus ${PREFIX}.ups > Run.log
#$MPIHOME/bin/mpirun -np ${NPROCS} ${UDIR}/StandAlone/sus ${PREFIX}.ups > Run.log
echo 'Stopped: '`date`

# --- Postprocess ---
# Convert UDA to XMF files 
python ${UDAVIZ} ${PREFIX}.uda --verbose
#${UDIR}/StandAlone/tools/puda/puda -US_MMS ${PREFIX}.uda
