#!/bin/csh -f

#__________________________________
# makeMasterUda_index:
# This script generates an index.xml file from 
# a series of udas.  
#_________________________________
if( $#argv < 1 ) then
  echo "makeMasterUda_index.csh < list of uda files >"
  echo "   To use:"
  echo ""
  echo "   mkdir <masterUda>"
  echo "   cd <masterUda> "
  echo "   makeMasterUda_index.csh ../uda.000 ../uda.001 ../uda.00N"
  exit(1)
endif

set rootPath = `dirname $0`

if ( -l $0 ) then
  set rootPath  = `readlink $0 | xargs dirname`  # if you're using a symbolic link to script
endif

set path = ($rootPath $path)

echo $path
set udas = ($argv[*])    # make sure you remove the last / from any entry

#__________________________________
# bulletproofing
set tmp = (`which makeCombinedIndex.sh` )
if ( $status ) then
  echo "ERROR:  makeMasterUda_index.csh:  couldn't find the script makeCombinedIndex.sh it must be in your path"
  exit
endif


echo ""
echo "---------------------------------------"
foreach X ($udas[*])
  echo "Passing $X through bulletproofing section"
  
  # does each uda exist
  if (! -e $X ) then
    echo "ERROR: makeMasterUda: can't find the uda $X"
    exit
  endif
  
  # does each index.xml exist
  if (! -e $X/index.xml ) then
  echo "Working on $X " 
    echo "______________________________________________________________"
    echo "ERROR: makeMasterUda: can't find the file $X/index.xml"
    echo "                   Do you want to continue"
    echo "                             Y or N"
    echo "______________________________________________________________"
    set ans = $<

    if ( $ans == "n" || $ans == "N" ) then
      exit
    endif
  endif
end

#__________________________________
# Look for difference in the variable lists between the udas
# and warn the users if one is detected
echo ""
echo "---------------------------------------"
echo "Looking for differences in the variable lists"

mkdir ~/.scratch  >&/dev/null

foreach X ( $udas[*] )
  set here = `basename $X`
  grep variable $X/index.xml > ~/.scratch/$here
end

set n = $#argv    # counters
@ c  = 1
@ cc = 2

while ( $c != $n)  
  set X = `basename $udas[$c]`
  set Y = `basename $udas[$cc]`
  
  #only look for differences if both index.xml files have a variable list
  set ans1 = `grep -c variables ~/.scratch/$X`
  set ans2 = `grep -c variables ~/.scratch/$Y` 
  
  if ($ans1 == "2" && $ans2 == "2") then
    diff -B ~/.scratch/$X ~/.scratch/$Y >& /dev/null
    if ($status != 0 ) then
      echo "  more Difference in the variable list detected between $X/index.xml and $Y/index.xml"
      sdiff -s -w 170 ~/.scratch/$X ~/.scratch/$Y
    endif
  endif
  @ c  = $c + 1
  @ cc = $cc + 1
end

#__________________________________
# copy the index.xml file from uda[1]
# remove all the timestep data
echo ""
echo "---------------------------------------"
echo "Creating the base index file from the $udas[1]"

cat $udas[1]/index.xml | sed /"timestep "/d > index.tmp

# remove   </timesteps> & </Uintah_DataArchive>
sed /"\/timesteps"/,/"\/Uintah_DataArchive"/d <index.tmp > index.xml

# add the list of timesteps to index.xml
makeCombinedIndex.sh "$udas" >> index.xml

# put the xml tags back
echo  "  </timesteps>" >> index.xml
echo  "</Uintah_DataArchive>" >> index.xml

#__________________________________
# cleanup
/bin/rm -rf ~/.scratch index.tmp
exit


