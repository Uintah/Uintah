# Makefile fragment for this subdirectory
SRCDIR   := Packages/Uintah/CCA/Components/Models

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCS    += \
       $(SRCDIR)/ModelFactory.cc

RADIATION :=

ifeq ($(BUILD_RADIATION),yes)
  RADIATION += $(SRCDIR)/Radiation   
endif

SUBDIRS := $(SRCDIR)/FluidsBased \
           $(RADIATION)

PSELIBS :=              \
        Core/Exceptions \
        Core/Geometry   \
        Core/Thread     \
        Core/Util       \
        Packages/Uintah/CCA/Ports          \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/IO            \
        Packages/Uintah/Core/Util          \
        Packages/Uintah/Core/GeometryPiece \
        Packages/Uintah/Core/Labels        \
        Packages/Uintah/Core/Parallel      \
        Packages/Uintah/Core/ProblemSpec

ifneq ($(BUILD_ICE),no)
  PSELIBS += Packages/Uintah/CCA/Components/ICE
endif

ifneq ($(BUILD_MPM),no)
  PSELIBS += Packages/Uintah/CCA/Components/MPM
endif

ifneq ($(BUILD_MPM),no) 
  ifneq ($(BUILD_ICE),no) 
    PSELIBS += Packages/Uintah/CCA/Components/MPMICE
    SUBDIRS += $(SRCDIR)/HEChem
  endif
endif

######################################################
# Sub-dir recurse has to be after above to make sure
# we include the right sub-dirs.
#
include $(SCIRUN_SCRIPTS)/recurse.mk
#
######################################################

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY) $(M_LIBRARY)

ifneq ($(HAVE_PETSC),)
  LIBS := $(LIBS) $(PETSC_LIBRARY) 
endif

ifneq ($(HAVE_HYPRE),)
  LIBS := $(LIBS) $(HYPRE_LIBRARY) 
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
