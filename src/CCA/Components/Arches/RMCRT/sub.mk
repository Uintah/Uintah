# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := CCA/Components/Arches/RMCRT

SRCS     += $(SRCDIR)/Ray.cc

# SUBDIRS := $(SRCDIR)/fortran 
# include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
        Core/ProblemSpec   \
        Core/GeometryPiece \
        Core/Grid          \
        Core/Util          \
        Core/Disclosure    \
        Core/Exceptions    \
        CCA/Components/OnTheFlyAnalysis \
        CCA/Ports     \
        Core/Parallel \
        Core/Util       \
        Core/Thread     \
        Core/Exceptions \
        Core/Geometry   \
        Core/Containers \
	      Core/Math

LIBS := $(LIBS) $(XML2_LIBRARY) $(F_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

