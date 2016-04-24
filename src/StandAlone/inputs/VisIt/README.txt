NOTE:This file is made as a novice primer to show how to us the python command line to manipulate VisIt. This was intended for users who already have a good grasp of VisIt's GUI and the Uintah Computational Framework. To gain an understanding of what the script can do, please follow the "Procedure" section and look over the attached .py script. All files within this directery were created by Andrew Bezdjian Nov, 17 2013. Free to use.

The files within this directory are described below:

"SliceUda.py": This python script takes an .uda file (read "Procedure" below) as an input and creates an array of pictures using VisIt and the VisIt Command Line Interface (cli).




PROCEDURE:
1. To begin, we need an uda file to analyze. Proceed to /uintah/src/StandAlone/inputs/MPMICE and run the "M2sphere3L.ups" file. Once this has finished running you should have an "M2sphere.2L.uda.000" file in your MPMICE directory. ----Please note that in this example and with this .py file, only the first timetep of the .uda is used to create photos from. So you don't need to run the .ups to completion. Feel free to cut the run after a dozen or so timesteps.

2. Once you have run the .ups file to get "M2sphere.2L.uda.000" the following command can be run from this directory to get a set of pictures:

"visit -cli -nowin -s ./SliceUda.py -uda /path/to/uintah/src/StandAlone/inputs/MPMICE/M2sphere.2L.uda.000 -save /save/directory/to/be/created -resolution 1200x1800 -timestep 0"

This should spit out a few pictures (in a directory made by the script, save/directory) of a plane slice going through a sphere. And this concludes the way to use an uda file to make a set of pictures.

If you would like to make this set of pictures into a movie please consult:

/uintah/opt/StandAlone/scripts/ppm_To_mpg

This script will compile a set of pictures to a movie.



Another common thing users would like to do in VisIt is to set up a session file and then step through that session file timestep by timestep rather than to step through one timestep spatially like we did above. The command to do this is below: (and it is also much simpler)

visit -cli -nowin -movie -v 2.6.1 -format jpeg -geometry 1408x912 -output /home/movie -fps 10 -start 120 -end 149 -sessionfile /home/sims/sessionfile.session

NOTE:The above command would be with visit version 2.6.1, it would output a movie with 10 frames per second with a 1408x912 widthXheight schematic. It would save the photos as /home/movie_0001.jpeg etc. 

WARNING: VisIt often has issues making photos into movies, so if you run the previous command and only get a bunch of pictures out and no movie, use the '/uintah/opt/StandAlone/scripts/ppm_To_mpg' script that is also listed above to make a movie out of those pictures.
