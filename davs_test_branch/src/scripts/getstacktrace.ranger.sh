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
NODESFILE=$HOME/.sge/prolog_hostfile.$JOBID.*

if [ ! -f $NODESFILE ] ; then
  echo "error NODESFILE '$NODESFILE' does not exist"
  exit
fi

NODES=`cat $NODESFILE | cut -f 1 -d " "| uniq`

DIR=`pwd`

#create .gdbcommands file
  # these commands will be executed inside of gdb
echo "where" > .gdbcommands
echo "quit" >> .gdbcommands


#remove any left over files from the past.
for file in .stack.*.*; do
  rm $file &> /dev/null
done

echo "Starting stack trace command on nodes"

numnodes=`echo "$NODES" | wc -l`
echo "Number of nodes: $numnodes"
for node in $NODES
do
  echo "Running stack trace commands on node: $node" >&2
  #get processors matching command name
  PSOUT=`ssh $node "ps -o pid,command x | grep $PROCESS | grep -v grep | grep -v bash | grep -v ssh | grep -v ibrun"`
  #extract pids
  PIDS=`echo "$PSOUT" | cut -f 1 -d " "`

  echo $node

  for pid in $PIDS
  do
    COMMAND="cd $DIR && echo 'y' | $DB  -gdb -pid $pid -command .gdbcommands 2> /dev/null"
    #echo "Running command '$COMMAND'" >&2
    #execute gdb
    #ssh $node "$COMMAND" > .stack.$node.$pid &
    echo ".stack.$node.$pid"
    ssh $node "$COMMAND" 2> /dev/null #1> .stack.$node.$pid 
    #cat .stack.$node.$pid
    echo " "
    echo " "
    echo " "
    #rm -f .stack.$node.$pid
  done
done

#remove left over files
rm -f .gdbcommands
