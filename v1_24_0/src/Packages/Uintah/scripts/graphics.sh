#! /bin/sh

if test ${#TTTS} = 0; then
    echo "Must set the enviroment variable TTTS to the path to the TextToTriSurf exectuble"
    exit
fi

if test $# -ne 2; then
    echo "Usage: $0 mat_id 'uda_to_convert' "
    exit
fi

if test $1 -ge 0; then
    mat_id=$1
    uda_dir=$2
fi 


#  Find the location of the TextToTriSurfField

cd $uda_dir
echo $uda_dir

for i in t0*
do
  cd $i
  if test  -d crackData; then
      cd crackData
      `$TTTS  cx.mat00$mat_id ce.mat00$mat_id mat00$mat_id.fld -noPtsCount -noElementsCount`
      wc -l ce.mat00$mat_id
      cd ../..
  fi
done

exit

