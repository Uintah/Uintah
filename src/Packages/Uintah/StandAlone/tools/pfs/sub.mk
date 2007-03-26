
SRCDIR := Packages/Uintah/StandAlone/tools/pfs

###############################################
# pfs

SRCS := $(SRCDIR)/pfs.cc
PROGRAM := $(SRCDIR)/pfs

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Packages/Uintah
else
  PSELIBS := \
      Packages/Uintah/Core/Grid \
      Packages/Uintah/Core/Util \
      Packages/Uintah/Core/Parallel \
      Packages/Uintah/Core/Exceptions \
      Packages/Uintah/Core/Math \
      Packages/Uintah/Core/ProblemSpec \
      Packages/Uintah/CCA/Ports \
      Packages/Uintah/CCA/Components/ProblemSpecification \
      Core/Exceptions \
      Core/Geometry 
endif

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

###############################################
# pfs 2 - Steve Maas' version

SRCS := $(SRCDIR)/pfs2.cc
PROGRAM := $(SRCDIR)/pfs2

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Packages/Uintah
else
  PSELIBS := \
     Packages/Uintah/Core/Grid \
     Packages/Uintah/Core/Util \
     Packages/Uintah/Core/Parallel \
     Packages/Uintah/Core/Exceptions \
     Packages/Uintah/Core/Math \
     Packages/Uintah/Core/ProblemSpec \
     Packages/Uintah/CCA/Ports \
     Packages/Uintah/CCA/Components/ProblemSpecification \
     Core/Exceptions \
     Core/Geometry
endif

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

