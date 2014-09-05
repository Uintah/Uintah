#! /bin/sh

if test $# -ne 3; then
    echo "Usage: $0 'uda_to_convert' -ttts path_to_TextToTriSurf or"
    echo "Usage: $0 -ttts path_to_TextToTriSurf 'uda_to_convert' "
    exit
fi


if test $1 = "-ttts"; then
    ttts=$2
    uda_dir=$3
fi

if test $2 = "-ttts"; then
    uda_dir=$1
    ttts=$3
fi

#  Find the location of the TextToTriSurfField

cd $uda_dir
echo $uda_dir

for i in t0*
do
  cd $i
  if test  -d crackData; then
      cd crackData
      `$ttts  cx.mat000 ce.mat000 mat000.fld -noPtsCount -noElementsCount`
      wc -l ce.mat000
      cd ../..
  fi
done

exit

