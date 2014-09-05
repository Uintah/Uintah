# Makefile fragment for this subdirectory

SRCDIR  := Packages/Uintah/StandAlone/tools/dumpfields
#PROGRAM := Packages/Uintah/StandAlone/dumpfields
PROGRAM := Packages/Uintah/StandAlone/tools/dumpfields/dumpfields

SRCS    := \
	$(SRCDIR)/dumpfields.cc \
	\
	$(SRCDIR)/utils.h $(SRCDIR)/utils.cc \
	$(SRCDIR)/Args.h  $(SRCDIR)/Args.cc \
	$(SRCDIR)/FieldSelection.h $(SRCDIR)/FieldSelection.cc \
	\
	$(SRCDIR)/FieldDiags.h $(SRCDIR)/FieldDiags.cc \
	$(SRCDIR)/ScalarDiags.h $(SRCDIR)/ScalarDiags.cc \
	$(SRCDIR)/VectorDiags.h $(SRCDIR)/VectorDiags.cc \
	$(SRCDIR)/TensorDiags.h $(SRCDIR)/TensorDiags.cc \
	\
	$(SRCDIR)/FieldDumper.h $(SRCDIR)/FieldDumper.cc \
	$(SRCDIR)/TextDumper.h $(SRCDIR)/TextDumper.cc \
	$(SRCDIR)/EnsightDumper.h $(SRCDIR)/EnsightDumper.cc \
	$(SRCDIR)/InfoDumper.h $(SRCDIR)/InfoDumper.cc \
	$(SRCDIR)/HistogramDumper.h $(SRCDIR)/HistogramDumper.cc 

ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/Uintah
else
  PSELIBS := \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/Util          \
        Packages/Uintah/Core/Math          \
        Packages/Uintah/Core/Parallel      \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/DataArchive   \
	Packages/Uintah/CCA/Ports          \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Core/XMLUtil \
        Core/Exceptions  \
        Core/Persistent  \
        Core/Geometry    \
        Core/Thread      \
        Core/Util        \
        Core/OS          \
        Core/Containers
endif

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

