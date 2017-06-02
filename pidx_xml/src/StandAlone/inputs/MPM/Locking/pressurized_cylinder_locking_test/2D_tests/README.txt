Validation of the Axisymmetric Implementation in Uintah as well as the overarching host code. 

Specifically seeking to induce volumetric and shear locking phenomena using extreme material specification. If you are on a linux machine with bash shell, python, numpy, and matplotlib (all but bash shell found in 'free enthought python distribution') and also have mpirun (optional) sus and partextract on your path you can execute the 'RUN_2D_Tests' script. All tests will take approximately 24 hours on 8 processors and take up approximately 150 Gigs. Comparitive plots will be produced in the images directory. A basic summary of the information contained in these plots can be found in the generated text 'file err_tracking.txt'.

Optionally single tests can be run from their directories in the following fashion:

	mpirun -np 4 sus -mpi pressure_cylinder_volumetric_locking_test.ups |& tee runlog.log  
		(with mpirun -- note most decks have 8 patches and some have 4)
	or
	sus pressure_cylinder_volumetric_locking_test.ups |& tee runlog.log 
		(without mpirun)

after which you will need to execute the python script setup_restart.py as follows:

	setup_restart.py ./volumetric_locking_test.uda.000/

and again execute sus:

	mpirun -np 4 sus -mpi -restart -move ./volumetric_locking_test.uda.000/ |& tee restart_runlog.log
		(with mpirun)
	or
	sus -restart -move ./volumetric_locking_test.uda.000/ |& tee restart_runlog.log
		(without mpirun)

After which post processing is done using the '2D_Tests_post_proc.py' python script argumented with the uda path, cell spacing (in meters), bulk, and shear moduli (in Pa). 
	as in - 2D_Tests_post_proc.py uda_path cell_spacing bulk_modulus shear_modulus
or for Aluminum with a cell spacing of 0.0200 meters
	2D_Tests_post_proc.py ./volumetric_locking_test.uda.001/ 0.0200 70.28e9 26.23e9

The problem is that of a cylinder with an applied external pressure. The analytical solution and problem can be found at:

	http://www.solidmechanics.org/text/Chapter8_6/Chapter8_6.htm (8.6.2)
	and
	http://www.solidmechanics.org/text/Chapter4_1/Chapter4_1.htm#Sect4_1_1 (4.1.9)

respectively. Solidmechanics.org is unaffiliated with the University of Utah and is maintained by:

	Allen F. Bower

I would like to thank him for making this material available on the web. 

Dave A.

