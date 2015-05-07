# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/RMCRT

SRCS     += $(SRCDIR)/Ray.cc

LIBS := $(LIBS) $(XML2_LIBRARY) $(F_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

