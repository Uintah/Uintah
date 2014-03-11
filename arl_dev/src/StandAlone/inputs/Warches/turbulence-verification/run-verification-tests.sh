#!/bin/bash
suspath=$1
sus=$1sus
echo $1
models[0]='dsmag'
models[1]='viscous'
models[2]='csmag'
models[3]='vreman'
models[1]='wale'
#models[5]='inviscid'
captions[0]='Dynamic Smagorinsky Model, '
captions[1]='Viscous, '
captions[2]='Constant Smagorinsky Model, '
captions[3]='Vreman Model, '
captions[1]='W.A.L.E. Model, '
#captions[5]='inviscid, '
res[0]=32
res[1]=64
var='warches-decay-isotropic-turbulence-'${models[1]}'-'${res[1]}.ups
inpPath='inputs/Warches/turbulence-verification/'
# declare -i nmodels
# nmodels=${#models[@]}-1;

echo -------------------------------------------
echo 'Running verification tests'

for grid in ${res[@]}
	do
	  COUNTER=0
		for model in ${models[@]}
			do
			  modRes=$model'-'$grid
			  baseVar='warches-decay-isotropic-turbulence-'$modRes
				var=$baseVar.ups
				modvar=MOD_$var
				sed "s@inputs/ARCHES/periodicTurb/@${suspath}inputs/ARCHES/periodicTurb/@g" $var > $modvar
                echo -------------------------------------------				
				echo 'running' $modvar
				time mpirun -np 8 $sus $modvar > $modvar.log
				rm $modvar
				echo 'Extracting Data for' $var
				extractDataScript='lineextract-warches-'$modRes.sh
				modExtractDataScript=MOD_'lineextract-warches-'$modRes.sh				
				lineextract=$suspath'lineextract'				
				sed 's@./lineextract@'$lineextract'@g' $extractDataScript > $modExtractDataScript
				echo 'Extracting data using ' $modExtractDataScript		
 				chmod +x $modExtractDataScript		
 				./$modExtractDataScript > $modvar.lineextract.log
 				rm $modExtractDataScript 				
 				# copy KE dat files:
#          	    baseVar='warches-decay-isotropic-turbulence-'$model'-'$grid
#			    udaDir=$baseVar.uda
# 				cp $udaDir/TotalKineticEnergy_uintah.dat KEDecay-$modRes.dat
 				matlabfilename="'ke_warches_${modRes}'"
 				matlabbasename="'${grid}_warches_${model}'"
 				matlabcaption="'${captions[$COUNTER]} ${grid}^3'"
 				matlabcommand='plot_energy_spectrum_all('"${matlabfilename}"','"${matlabbasename}"','"${matlabcaption}"');\n'
# 				if [ "$model" != "${models[0]}" ] && [ "$model" != "${models[1]}" ]
# 				then
#					decayfilename="'ke_decay_warches_${modRes}'"
#					matlabcommand=$matlabcommand'plot_energy_decay_all('"${decayfilename}"','"'${model}'"','"${grid}"','"${matlabcaption}"');' 				
# 				fi
 				echo -e "$matlabcommand\nexit" > matlabcommand.m
 				matlab -nodisplay -nosplash < matlabcommand.m
 				let COUNTER=COUNTER+1
		  done	  
	done
	# now plot energy decay
	
echo 'Cleaning up...'
rm matlabcommand.m
rm uvel*.txt
rm vvel*.txt
rm wvel*.txt
rm *.dot
rm *.log
rm -rf *.uda*
echo 'Existing gracefully...'