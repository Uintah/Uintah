#! /bin/sh

gs=
case=
opt_dbg=

while getopts g:c:d: opt
do
    case "$opt" in
      g)  gs="$OPTARG";;
      c)  case="$OPTARG";;
      d)  opt_dbg="$OPTARG";;
    esac
done
shift `expr $OPTIND - 1`

echo $gs
echo $case
echo $opt_dbg

#DIR=/usr/local/TestData/dbg/IMPM
DIR=$gs/$opt_dbg/$case
echo $DIR

set -x

for i in `ls -d *.uda.000`; do
	b=`basename $i .uda.000`
	d=`basename $i .000`
	cp -a --remove-destination $i $DIR/$b/$d;
done

exit 0	
