# Makefile fragment for this subdirectory
SRCDIR   := Packages/Uintah/CCA/Components/Models

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCS	+= \
       $(SRCDIR)/ModelFactory.cc

SUBDIRS := $(SRCDIR)/test \
	   $(SRCDIR)/Radiation \
           $(SRCDIR)/HEChem
include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
	Core/Exceptions \
	Core/Geometry \
	Core/Thread \
	Core/Util \
	Packages/Uintah/CCA/Components/MPM \
        Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Disclosure \
	Packages/Uintah/Core/Exceptions \
        Packages/Uintah/Core/Grid \
        Packages/Uintah/Core/Util \
        Packages/Uintah/Core/GeometryPiece \
        Packages/Uintah/Core/Labels \
	Packages/Uintah/Core/Parallel \
        Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Components/ICE \
	Packages/Uintah/CCA/Components/MPMICE

LIBS	:= $(XML_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY) $(M_LIBRARY)

ifneq ($(HAVE_PETSC),)
  LIBS := $(LIBS) $(PETSC_LIBRARY) 
endif

ifneq ($(HAVE_HYPRE),)
  LIBS := $(LIBS) $(HYPRE_LIBRARY) 
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
