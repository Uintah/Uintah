
speed_options = 
CFLAGS = -O0 -g
path_F90_Compiler = 
path_Cpp_Compiler = 
MPICC=mpicc
MPIF90=mpif90
Cpp_Compiler = $(path_Cpp_Compiler)$(MPICC)
F90_compiler = $(path_F90_Compiler)$(MPIF90)

EXECUTABLE = forces-parallel

#/opt/mpich.pgi/bin/mpif90

LINKER = $(F90_compiler) $(speed_options) -o
F90    = $(F90_compiler) $(speed_options) -c
CPP    = $(Cpp_Compiler) $(CFLAGS) -c

SRC_F90 = comunications.f90 \
          error_handler_module.f90 \
          misc.f90 \
          list_generator.f90 \
          forces_p.f90 \
		
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
