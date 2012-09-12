#/bin/bash

speed_options =
FFLAGS = -O0 -g
CFLAGS = -O0 -g
c_options = 
#make a.out speed_options='-CB -g' if want to change speed options
path_F90_Compiler =  
path_Cpp_Compiler =
name_cc=gcc
name_f90=gfortran
Cpp_Compiler = $(path_Cpp_Compiler)$(name_cc)
F90_compiler = $(path_F90_Compiler)$(name_f90)

EXECUTABLE = spme-serial

LINKER = $(F90_compiler) $(speed_options) -o
F90  = $(F90_compiler) $(FFLAGS) -c
CPP  = $(Cpp_Compiler) $(CFLAGS) -c


SRC_F90 = header_list.f90 \
	random_generator.f90 \
	variables_smpe.f90 \
	math.f90  \
	periodic_boundary.f90 \
	allocating.f90 \
	fft-3d.f90 \
	spline2.f90 \
	smpe_utility_pack.f90 \
	smpe_driver.f90 \
	slow_fourier_ewald.f90 \
	IO_module.f90 

		
SRC_CPP = c_functions.c
SRC_MAIN_F90_MD = main.f90

SRC_F90_MD = $(SRC_F90) $(SRC_MAIN_F90_MD)
obj_F90_MD = $(subst .f90,.o,$(SRC_F90_MD))
obj_CPP = $(subst .c,.o,$(SRC_CPP))
obj_all_MD = $(obj_F90_MD) $(obj_CPP)

gen_obj_f90_MD = $(F90) $(SRC_F90_MD)
gen_obj_cpp = $(CPP) $(SRC_CPP)
gen_link_MD = $(LINKER) $(EXECUTABLE) $(obj_all_MD)


all:
	$(gen_obj_f90_MD)
	$(gen_obj_cpp)
	$(gen_link_MD)

clean:
	rm *.o *.mod
	
realclean:
	rm *.o *.mod $(EXECUTABLE)
	

