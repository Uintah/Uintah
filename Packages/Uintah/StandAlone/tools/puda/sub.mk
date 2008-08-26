SRCDIR  := Packages/Uintah/StandAlone/tools/puda
PROGRAM := Packages/Uintah/StandAlone/tools/puda/puda

SRCS := \
	$(SRCDIR)/asci.cc        \
	$(SRCDIR)/jim1.cc        \
	$(SRCDIR)/jim2.cc        \
	$(SRCDIR)/jim3.cc        \
        $(SRCDIR)/jim4.cc        \
	$(SRCDIR)/rtdata.cc      \
	$(SRCDIR)/tecplot.cc     \
	$(SRCDIR)/util.cc        \
	$(SRCDIR)/varsummary.cc  \
	$(SRCDIR)/puda.cc

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Packages/Uintah
else
  PSELIBS := \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Packages/Uintah/CCA/Ports          \
        Packages/Uintah/Core/DataArchive   \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/Math          \
        Packages/Uintah/Core/Parallel      \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/Util          \
        Core/Containers  \
        Core/Exceptions  \
        Core/Geometry    \
        Core/OS          \
        Core/Persistent  \
        Core/Thread      \
        Core/Util        \
        Core/XMLUtil     
endif

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY) $(TEEM_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

