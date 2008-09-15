# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Angio

SRCS     += \
	$(SRCDIR)/AngioParticleCreator.cc  \
	$(SRCDIR)/AngioMaterial.cc  \
	$(SRCDIR)/AngioFlags.cc  \
	$(SRCDIR)/Angio.cc	

PSELIBS := Packages/Uintah/Core/Grid \
	Core/Datatypes \
	Core/Util

LIBS := $(XML2_LIBRARY) $(VT_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
