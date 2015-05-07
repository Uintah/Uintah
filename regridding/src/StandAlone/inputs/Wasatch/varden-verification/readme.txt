This directory contains python scripts that enable immediate verification of Wasatch’s variable density projection algorithm.

You will need to create a symbolic link to sus and to lineextract:
cd to this directory (inputs/Wasatch/varden-verification)
ln -s /uintah/opt/StandAlone/sus sus
ln -s /uintah/opt/StandAlone/lineextract lineextract

To run the advection verification case:
python advection-temporal.py inputfile.ups -levels 5 (levels denotes how many time levels you want to run your verification against).