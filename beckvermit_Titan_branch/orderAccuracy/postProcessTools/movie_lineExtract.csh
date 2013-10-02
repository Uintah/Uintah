#!/bin/csh

#______________________________________________________________________
#  This post processing script runs the script movie_lineExtract
#  which makes line plots of the variables of interest
#  After the all of the frames have been make a movie is make
#______________________________________________________________________

echo "---------------------------------------"
echo "movie_lineExtract:"

set uda = $argv[4]  # The 4th argument is the uda name

#execute post processing script
scripts/ICE/movie_lineExtract $uda
echo "==============="

# turn the individual frames into a movie
mkdir $uda/movie
mv movie.*.png $uda/movie/
cd $uda/movie
../../scripts/ppm_To_mpg <<+
n
n
n
n
n
+
exit
