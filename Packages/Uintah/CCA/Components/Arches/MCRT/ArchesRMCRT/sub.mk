# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Arches/MCRT/ArchesRMCRT

SRCS     += $(SRCDIR)/RMCRTRadiationModel.cc 

# SUBDIRS := $(SRCDIR)/fortran 
# include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/GeometryPiece \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/Util          \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/CCA/Components/OnTheFlyAnalysis \
        Packages/Uintah/CCA/Ports     \
        Packages/Uintah/Core/Parallel \
        Core/Util       \
        Core/Thread     \
        Core/Exceptions \
        Core/Geometry   \
        Core/Containers

LIBS := $(LIBS) $(XML2_LIBRARY) $(F_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

