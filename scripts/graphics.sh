#! /bin/sh

if test $# -ne 4; then
    echo "Usage: $0 mat_id 'uda_to_convert' -ttts path_to_TextToTriSurf or"
    echo "Usage: $0 mat_id -ttts path_to_TextToTriSurf 'uda_to_convert' "
    exit
fi

if test $1 -ge 0; then
    mat_id=$1
fi 

if test $2 = "-ttts"; then
    ttts=$3
    uda_dir=$4
fi

if test $3 = "-ttts"; then
    uda_dir=$2
    ttts=$4
fi

if test $3

#  Find the location of the TextToTriSurfField

cd $uda_dir
echo $uda_dir

for i in t0*
do
  cd $i
  if test  -d crackData; then
      cd crackData
      `$ttts  cx.mat00$mat_id ce.mat00$mat_id mat00$mat_id.fld -noPtsCount -noElementsCount`
      wc -l ce.mat00$mat_id
      cd ../..
  fi
done

exit

