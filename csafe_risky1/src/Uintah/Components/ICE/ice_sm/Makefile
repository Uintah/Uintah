#_______________________________________________________________________
#   Set some specific flags
# BIT  =        n32, 64
# MEMORY_LIB    Malloc_audit    Steve Parkers malloc library with the
#                               audit function
#               Malloc_NAN      Steve Parkers malloc libraray that initializes
#                               all of the entries to NAN.
#               efence          Electric fence (PLOTTING WORK WORK)
#               malloc_cv       CVD's malloc library
#
#   Note that you need to set the environmental variable ICE_TEST
#   to compile a test case
#_______________________________________________________________________

IMPLICIT        = no
EXPLICIT        = yes
OS              = sgi
MM              = no   #multimaterial flow
#__________________________________
#  OS dependencies
#___________________________________ 
ifeq ($(OS),linux)
  BIT             = 
  MEMORY_LIB      = 
  CUR_DIR         = $(shell pwd)
  TOP_DIR         = 

  ifeq ($(ICE_DIR),)    # needed by the test directory
    ICE_DIR         = $(CUR_DIR)/$(TOP_DIR)
  endif

  PGPLOT          = $(ICE_DIR)/Libraries/Linux
  MYLIB_PATH      = -L $(ICE_DIR)/Libraries/Linux
  X11_PATH        = -L /usr/X11R6/lib 
  ABI_FLAGS       = 
  CC              = gcc -DPGPLOT_DIR=\"$(PGPLOT)\"
  FULLWARN        = 
  TRAPUV          =
endif
#--SGI-----------
ifeq ($(OS),sgi)
  BIT             = 64
  MEMORY_LIB      = -l malloc_cv
  CUR_DIR         = $(shell pwd)
  TOP_DIR         = 

  ifeq ($(ICE_DIR),)    # needed by the test directory
    ICE_DIR         = $(CUR_DIR)/$(TOP_DIR)
  endif

  MYLIB_PATH      = -L $(ICE_DIR)/Libraries/$(BIT)bit
  PGPLOT          = $(ICE_DIR)/Libraries
  X11_PATH        =
  ABI_FLAGS       = -$(BIT)
  CC              = cc -DPGPLOT_DIR=\"$(PGPLOT)\"
  FULLWARN        = -fullwarn
  TRAPUV          = -trapuv
endif

#_______________________________________________________________________
#      COMMON MACROS REQUIRED BY BOTH EXPLICIT AND IMPLICIT versions of ICE
#_______________________________________________________________________
ICE_INCLUDE     = -I. -I$(ICE_DIR)/Header_files
MYLIB           = $(MYLIB_PATH) $(X11_PATH) -ltecio -lcpgplot -lpgplot \
                    -lX11 -L. $(MEMORY_LIB)

LIB             = $(MYLIB) -lm 

MAKEFLAGS       = --no-keep-going 

CFILES         += main2.c \
                  input.c \
                  Plot_routines/plot_vector.c \
                  Plot_routines/plot_control.c \
                  Plot_routines/plot_face_center.c \
                  Plot_routines/plot_common.c \
                  Plot_routines/plot_contour.c \
                  Plot_routines/plot_2d_line.c \
                  Plot_routines/plot_cursor_pos.c \
                  p_face.c \
                  explicit_delPress.c\
                  equate_ptr_addrss.c \
                  interpolate_vel_CC_to_FC.c \
                  grid.c \
                  flux_or_primitive.c \
                  Equation_of_state/equation_of_state.c \
                  Equation_of_state/speed_of_sound.c \
                  lagrangian.c \
                  commonFunctions.c \
                  timeadvanced.c \
                  Advection_2D/advect_grad_limiter.c \
                  Advection_2D/advect_centroids.c \
                  Advection_2D/advect_preprocess.c \
                  Advection_2D/advect_q.c \
                  Advection_2D/advect_q_flux.c \
                  Advection_2D/advect_q_vertex.c \
                  Boundary_Cond/boundary_cond_FC.c \
                  Boundary_Cond/boundary_cond.c \
                  Write_output/output_FC.c \
                  Write_output/output_CC.c \
                  Write_output/output_misc.c \
                  Source_Sinks/energy.c \
                  Source_Sinks/momentum.c \
                  Source_Sinks/shear_stress.c\
                  initialize_variables.c \
                  nrutil+.c

# This is magic to enable overriding code on a file-by-file basis in the
# Test subdirectories.  We have a list of source files which may or may
# not exist in any particular subdirectory in the variable CFILES.  We
# have a shell script which, given a list of CFILES looks for each in the
# directory from which make was run, and if it exists adds a relative path 
# to that file to the SRCS variable.  If the file does not exist in the
# directory from which make was run, we add a relative path to the file
# in ICE_DIR to the SRCS variable instead.  People need this to write
# effective test cases.
#                              -- mcq, 6/30/00

SRCS            = $(shell for i in $(CFILES) ; do \
                            if [ -r $$i ] ; then \
                              echo $$i ; \
                            else \
                              echo $(ICE_DIR)/$$i ; \
                            fi ; \
                          done)
                  
OBJS            = $(addsuffix .o,$(basename $(SRCS)))

CFLAGS        = -g $(FULLWARN) $(ICE_INCLUDE)

ifeq ($(EXPLICIT),yes)

all: ice
	@echo ''
	@echo '-----------------------'
	@echo 'EXPLICIT ice'
	@echo ''; echo ''

ice:   $(OBJS) $(OBJS_TEST_CASE)
	$(F77) $(TRAPUV) $(ABI_FLAGS) -o ice $(OBJS_TEST_CASE) $(OBJS) $(MYLIB)
       
clean:
	/bin/rm -f $(OBJS) ice core*
endif


#_______________________________________________________________________
#               IMPLICIT 
#_______________________________________________________________________
ifeq ($(IMPLICIT),yes)

PETSC_DIR       = /usr/local/share/petsc-2.0.24
ifeq ($(BIT), 64)
    PETSC_ARCH  = IRIX64
endif
ifeq ($(BIT), 32)
    PETSC_ARCH  = IRIX
endif
PETSC_BOPT      = g
PETSC_INCLUDE   = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/bmake/$(PETSC_ARCH)
PETSC_LIB       = -L$(PETSC_DIR)/lib/lib$(PETSC_BOPT)/$(PETSC_ARCH) \
		    -Wl,-rpath,$(PETSC_DIR)/lib/lib$(PETSC_BOPT)/$(PETSC_ARCH) \
                  -lpetscts -lpetscsnes -lpetscsles \
                  -lpetscdm -lpetscmat -lpetscvec -lpetsc -lmpiuni\
		    -lfortran -lmpi $(PETSC_DIR)/lib/lib$(PETSC_BOPT)/$(PETSC_ARCH)/blas.a

PETSC_FLAGS     = -DUSE_PETSC_DEBUG -DUSE_PETSC_LOG -DUSE_PETSC_BOPT_g -DUSE_PETSC_STACK


#--
  
#--IMPLICIT SOURCES
SRCS2           =$(ICE_DIR)/PressureSolve_PCG/pressure_PCG.c \
                 $(ICE_DIR)/PressureSolve_PCG/computeStencilWeights.c \
                 $(ICE_DIR)/PressureSolve_PCG/computeSource.c \
                 $(ICE_DIR)/PressureSolve_PCG/pressure_residual.c\
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/stencilShell.c\
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/multigridShell.c \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/psol.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/pset.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/stencilMult.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/copy_to_gridfcn.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/setval.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/relax.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/gemv.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/inject.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/gf_axpy.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/setHomogeneousFlowBC.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/copy_from_gridfcn.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/setLevelPtrs.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/setOperator.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/setCoarserOperator.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/interp.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/setLevelShapes.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/normalize.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/gf_copy.f \
                 $(ICE_DIR)/PressureSolve_PCG/pcgmg/setShape.f
       
OBJS_IMPLICIT   = $(addsuffix .o,$(basename $(SRCS2))) 

CFLAGS          = -g $(FULLWARN) -DIMPLICIT $(ICE_INCLUDE) $(PETSC_INCLUDE)
all: iceimplicit
	@echo ''
	@echo '-----------------------'
	@echo 'Now making the implicit ice'
	@echo ''; echo ''

iceimplicit: $(OBJS) $(OBJS_IMPLICIT)
	$(F77) $(TRAPUV) $(ABI_FLAGS) -o iceimplicit $(OBJS) $(OBJS_IMPLICIT) $(MYLIB) $(PETSC_LIB)
       
clean:
	/bin/rm -f $(OBJS) $(OBJS2) iceimplicit core*
endif
#_______________________________________________________________________
#               COMMON MACROS
#_______________________________________________________________________
remake:
	gmake clean
	gmake 
tags:
	/bin/rm -f tags
	ctags $(ICE_DIR)/main2.c
	ctags -a $(SRCS)
       
lib:
	ar qcv libICE.a $(OBJS)
       
.h.o: $<
	$(CC) $(ABI_FLAGS) $(CFLAGS) -o $@ -c $<
       
.c.o: $<
	$(CC) $(ABI_FLAGS) $(CFLAGS) -o $@ -c $<
       
.f.o: $<
	$(F77) $(ABI_FLAGS) $(FFLAGS) -o $@ -c $<
#include Makedepend


#__________________________________
#  Documentation macros
#___________________________________ 
DOC_MACROS = $(ICE_DIR)/Doc_utils
ICE.doc: $(SRCS)
	$(DOC_MACROS)/makedoc $(SRCS)> ICE.doc
       
ICE.html.public: $(SRCS)
	$(DOC_MACROS)/makebody  $(SRCS)> ICE_body.html 
	$(DOC_MACROS)/makemenu  $(SRCS)
	$(DOC_MACROS)/makeindex $(SRCS)> ICE.html ; $(DOC_MACROS)/autoftp
	
       
ICE.html: $(SRCS)
	$(DOC_MACROS)/makebody  $(SRCS)> ICE_body.html
	$(DOC_MACROS)/makemenu  $(SRCS)
	$(DOC_MACROS)/makeindex $(SRCS)> ICE.html

ICE.tex: $(SRC)
	$(DOC_MACROS)/maketex $(SRC) >ICE.tex ; latex ICE.tex

