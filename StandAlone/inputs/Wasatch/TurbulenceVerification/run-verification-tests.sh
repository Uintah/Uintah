#!/bin/bash
suspath=$1
sus=$1sus
echo $1
models[0]='csmag'
models[1]='vreman'
models[2]='wale'
models[3]='dsmag'
captions[0]='Constant Smagorinsky Model, '
captions[1]='Vreman Model, '
captions[2]='W.A.L.E. Model, '
captions[3]='Dynamic Smagorinsky Model, '
res[0]=32
res[1]=64
var='decay-isotropic-turbulence-'${models[1]}'_'${res[1]}.ups
inpPath='inputs/Wasatch/TurbulenceVerification/'
# declare -i nmodels
# nmodels=${#models[@]}-1;
echo -------------------------------------------
echo 'Running verification tests'

for grid in ${res[@]}
	do
	  COUNTER=0
		for model in ${models[@]}
			do
			  modRes=$model'_'$grid
				var='decay-isotropic-turbulence-'$modRes.ups
				modvar=MOD_$var
				sed 's@inputs/Wasatch/TurbulenceVerification/@@g' $var > $modvar
        echo -------------------------------------------				
				echo 'running' $modvar
				mpirun -np 8 $sus $modvar > $modvar.log
				rm $modvar
				echo 'Extracting Data for' $var
				extractDataScript='lineextract-wasatch-'$modRes.sh
				modExtractDataScript=MOD_'lineextract-wasatch-'$modRes.sh				
				lineextract=$suspath'lineextract'				
				sed 's@./lineextract@'$lineextract'@g' $extractDataScript > $modExtractDataScript
# 				echo 'Extracting data using ' $modExtractDataScript		
 				chmod +x $modExtractDataScript		
 				./$modExtractDataScript > $modvar.lineextract.log
 				rm $modExtractDataScript 				
 				matlabfilename="'ke_wasatch_${modRes}'"
 				matlabbasename="'${grid}_wasatch_${model}'"
 				matlabcaption="'${captions[$COUNTER]} ${grid}^3'"
 				matlabcommand='energy_spectrum_plot_all('"${matlabfilename}"','"${matlabbasename}"','"${matlabcaption}"');'
 				echo -e "$matlabcommand\nexit" > matlabcommand.m
 				matlab -nodisplay -nosplash < matlabcommand.m
 				let COUNTER=COUNTER+1
		  done	  
	done
echo 'Cleaning up...'
rm matlabcommand.m
rm uvel*.txt
rm vvel*.txt
rm wvel*.txt
rm *.dot
rm *.log
rm -rf *.uda*
echo 'Existing gracefully...'