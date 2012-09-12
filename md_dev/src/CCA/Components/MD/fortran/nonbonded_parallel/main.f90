
program main ! finally the main program
use local_initializations_module ! this contains the initializations that we use in this test
use IO_module                    ! input (read config) and output 
use list_generator_module        ! generate the list ; for each atom we store in an array how many other atoms are within cut off 
use vdw_forces_modules           ! evaluate energy forces 
implicit none

call init();  ! initialize the paralel environment
call read_config('input.tiny') ! get the configuration with atom coordinates; change 'input.small' with whatever file i.e. input.large (if needed)
call bcast_config();  ! put the config in all computing nodes
call local_init(); ! initialize the local variablibels used in this test program 
call forces(.true.); ! compute forces ! this is the main thing
call out_results();
call finalize();
 
end program main

