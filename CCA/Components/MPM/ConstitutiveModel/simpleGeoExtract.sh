partID=81606213632
file=halfSpaceExpGeo.uda.003
echo $file
./partextract -partvar p.stress -partid $partID $file > outStress.txt;./partextract -partvar p.plasticStrainVol -partid $partID $file > outPlasticStrain.txt;./partextract -partvar p.elasticStrainVol -partid $partID $file > outElasticStrain.txt;./partextract -partvar p.kappa -partid $partID $file > outKappa.txt
