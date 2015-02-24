#/bin/bash
echo $1
inputsdir=$1/inputs/Wasatch/
replacexml=$1/scripts/replace_XML_value
oldinputfile=$1/inputs/Wasatch/varden-projection-mms.ups
newinputfile=$1/inputs/Wasatch/varden-projection-mms-delme.ups
echo $newinputfile
cp $oldinputfile $newinputfile
$replacexml "/Uintah_specification/Time/max_Timesteps" "5000" $newinputfile
$replacexml "/Uintah_specification/DataArchiver/outputTimestepInterval" "1000" $newinputfile
$replacexml "/Uintah_specification/DataArchiver/checkpoint[@cycle=2]/@interval" "0" $newinputfile
cd $1
echo "now running mms verification test..."
mpirun -np 3 sus $newinputfile > suslog.txt

rm suslog.txt

./lineextract -v u -timestep 1 -istart 0 1 0 -iend 256 1 0 -pr 32 -cellCoords -uda varden-projection-mms.uda -o MMS_xxvol.txt
./lineextract -v f -timestep 1 -istart 0 1 0 -iend 256 1 0 -pr 32 -cellCoords -uda varden-projection-mms.uda -o MMS_xsvol.txt

./lineextract -v u -tlow 1 -istart 0 1 0 -iend 257 1 0 -pr 32 -uda varden-projection-mms.uda -o MMS_u_n.txt
./lineextract -v f -tlow 1 -istart 0 1 0 -iend 255 1 0 -pr 32 -uda varden-projection-mms.uda -o MMS_f_n.txt

mv MMS_* $inputsdir/varden-verification/

rm -rf varden-projection-mms.uda*