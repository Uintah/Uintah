#!/bin/bash

if [ -d ./runs ] 
then
 found_runs="T"
else
 found_runs="F"
 mkdir ./runs
fi

if [ -d ./runs/z_atoms ]
then
 found_runs_z_atoms="T"
else
 found_runs_z_atoms="F"
 mkdir ./runs/z_atoms
fi

if [ -d ./runs/z_mols ]
then
 found_runs_z_mols="T"
else
 found_runs_z_mols="F"
 mkdir ./runs/z_mols
fi

if [ -d ./runs/atoms ]
then
 found_runs_atoms="T"
else
 found_runs_atoms="F"
 mkdir ./runs/atoms
fi

if [ -d ./runs/mols ]
then
 found_runs_mols="T"
else
 found_runs_mols="F"
 mkdir ./runs/mols
fi

echo "found_runs=" $found_runs
echo "found_runs_z_atoms=" $found_runs_z_atoms
echo "found_runs_z_mols=" $found_runs_z_mols
echo "found_runs_atoms=" $found_runs_atoms
echo "found_runs_mols=" $found_runs_mols

