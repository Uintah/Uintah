#!/bin/bash
suspath=$1
sus=$1sus
echo $1
models[0]='csmag'
models[1]='vreman'
models[2]='wale'
models[3]='dsmag'
res[0]=32
res[1]=64
var='decay-isotropic-turbulence-'${models[1]}'_'${res[1]}.ups
inpPath='inputs/Wasatch/TurbulenceVerification/'
# declare -i nmodels
# nmodels=${#models[@]}-1;
echo -------------------------------------------
echo 'Running verification tests'
for model in ${models[@]}
	do
		for grid in ${res[@]}
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
		  done	  
	done
echo -------------------------------------------
echo 'Executing MATLAB plotting scripts'
echo -------------------------------------------
matlab -nodisplay -r plot_energy_spectra_all
#cleanup
echo -------------------------------------------
echo 'Cleaning up...'
echo -------------------------------------------
rm uvel*.txt
rm vvel*.txt
rm wvel*.txt
rm *.dot
rm *.log
rm -rf *.uda*
echo -------------------------------------------
echo 'Existing gracefully...'
echo -------------------------------------------