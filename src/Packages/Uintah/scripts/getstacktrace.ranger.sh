#!/bin/bash

#this script will get a stack trace for each process associated with the jobid provided.  

if [ $# -ne 2 ]; then
  echo "usage:  getstacktrace jobid executable"
  exit
fi

JOBID=$1
PROCESS=$2

#use idb as gdb tends to segfault
DB=`which idb` 

#get nodes
NODESFILE=$HOME/.sge/job.$JOBID.hostlist.*

if [ ! -f $NODESFILE ] ; then
  echo "error NODESFILE '$NODESFILE' does not exist"
  exit
fi

NODES=`cat $NODESFILE | uniq`

DIR=`pwd`

#create .gdbcommands file
  # these commands will be executed inside of gdb
echo "where" > .gdbcommands
echo "quit" >> .gdbcommands


#remove any left over files from the past.
rm -f .stack.*.*

echo "Starting stack trace command on nodes"

for node in $NODES
do
  #get processors matching command name
  PSOUT=`ssh $node "ps -o pid,command x | grep $PROCESS | grep -v grep | grep -v bash | grep -v ssh | grep -v ibrun"`
  #extract pids
  PIDS=`echo "$PSOUT" | cut -f 1 -d " "`
  for pid in $PIDS
  do
    COMMAND="cd $DIR && echo 'y' | $DB  -gdb -pid $pid -command .gdbcommands 2> /dev/null"
    #execute gdb
    #ssh $node "$COMMAND" > .stack.$node.$pid &
    ssh $node "$COMMAND" 2> /dev/null 1> .stack.$node.$pid &
  done
done

echo "Waiting for commands to finish"

wait
#combine results
for file in .stack.*.*
do
  cat $file 
done

#remove left over files
rm -f .stack.*.*
rm -f .gdbcommands
