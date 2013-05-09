
path_F90_Compiler =  
path_Cpp_Compiler =
CC=gcc
CFLAGS =  -O0 -g
FC=gfortran
FFLAGS= -O0 -g -ffree-line-length-none
Cpp_Compiler = $(path_Cpp_Compiler)$(CC)
F90_compiler = $(path_F90_Compiler)$(FC)

executable = a.out

LINKER = $(F90_compiler) $(CFLAGS) -o
F90  = $(F90_compiler) $(FFLAGS) -c
CPP  = $(Cpp_Compiler) $(CFLAGS) -c

SRC_F90 = header_list.f90 \
	lucretius_vars.f90 \
	units_def.f90 \
	variables_short_pairs.f90 \
	variables_smpe.f90 \
	printer_arrays.f90 \
	random_generator.f90 \
	math.f90 eigen.f90 minimize_3.f90 array-math.f90 \
	timing.f90 \
	DoDummyInit.f90 \
	spline2.f90 \
	spline_REAL.f90 \
	spline_z_k0.f90 \
	spline_REAL_NEW.f90 \
	comunications.f90 \
	allocating.f90 \
	periodic_boundary.f90 \
	anneal_temperature.f90 \
	poisson_1D.f90 \
	init_bcasting.f90 \
	chars.f90 \
	convert_mol_to_atom.f90 \
	vdw_def.f90 \
	ewald_def.f90 \
	intramol_def.f90 \
	ensamble_def.f90 \
	rotor_utils.f90 \
	excluded_list_eval.f90 \
	non_bonded_lists_eval.f90\
	parser.f90 \
	mol_utils.f90 \
	rdfs_collect.f90 \
	dummy_module.f90 \
	code_generator.f90 \
	sysdef.f90 \
	set_CG_flags.f90 \
	thetering_forces_module.f90 \
	long_range_corrections.f90 \
	intramol_constrains.f90 \
	intramolecular_forces.f90 \
	compute_14_interactions.f90 \
	REAL_part_intra_correction_sfc_sfc.f90 \
	fft-3d.f90 \
	diverse.f90 \
	fft-3d_modified.f90 \
	initialize_MD.f90 \
	initialize_MD_CYCLE.f90 \
	kinetics.f90 \
	gear_4.f90 \
	ensamble_driver.f90 \
	short_range_pair_forces.f90 \
	pair_short_MCmove_1atom.f90 \
	pair_short_ENERGY.f90 \
	smpe_utility_pack.f90  \
	smpe_dipole.f90 \
	Ew_2D_k0_Q_SLOW.f90 \
	Ew_2D_k0_Q_SLOW_ENERGY.f90 \
	ew_k0_2D_Q.f90 \
	ew_k0_2D_Q_ENERGY.f90 \
	ew_k0_2D_Q_DIP.f90 \
	ew_k0_2D_Q_DIP_ENERGY.f90 \
	Ew_Q_SLOW.f90 \
	Ew_Q_SLOW_ENERGY.f90 \
	exclude_intra.f90 \
	exclude_intra_ENERGY.f90 \
	exclude_intra_ENERGY_MCmove_1atom.f90 \
	cg_Q_DIP.f90 \
    add_on_water_Pt_ff.f90 \
	s_field_Q_energy.f90 \
    smpe_driver.f90 \
	force_driver.f90 \
	energy_driver_ENERGY.f90 \
	MC_move_0.f90 \
	lucretius_integrator.f90 \
	statistics.f90 \
	rsmd.f90 \
	rdfs.f90 \
	read_history_module.f90 \
	write_stuff.f90 \
	statistic_collector_driver.f90 \
	main_loops.f90 


		
SRC_CPP = c_functions.c
SRC_MAIN_F90_MD = main.f90

SRC_F90_MD = $(SRC_F90) $(SRC_MAIN_F90_MD)
obj_F90_MD = $(subst .f90,.o,$(SRC_F90_MD))
obj_CPP = $(subst .c,.o,$(SRC_CPP))
obj_all_MD = $(obj_F90_MD) $(obj_CPP)

gen_obj_f90_MD = $(F90) $(SRC_F90_MD)
gen_obj_cpp = $(CPP) $(SRC_CPP)
gen_link_MD = $(LINKER) $(executable) $(obj_all_MD)

all: exe

$(executable):
	$(gen_obj_f90_MD)
	$(gen_obj_cpp)
	$(gen_link_MD)

exe:
	$(gen_obj_f90_MD)
	$(gen_obj_cpp)
	$(gen_link_MD)


clean:
	rm *.o *.mod
	
realclean:
	rm *.o *.mod a.out
	
tree:
	./tree.sh
	
moveit:
	./briefcase
