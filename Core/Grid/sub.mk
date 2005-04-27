# Makefile fragment for this subdirectory


SRCDIR   := Packages/Uintah/Core/Grid

SUBDIRS := \
	$(SRCDIR)/BoundaryConditions \
	$(SRCDIR)/Variables \

include $(SCIRUN_SCRIPTS)/recurse.mk          

SRCS     += \
	$(SRCDIR)/Box.cc \
	$(SRCDIR)/Grid.cc \
	$(SRCDIR)/Level.cc \
	$(SRCDIR)/Material.cc \
	$(SRCDIR)/PatchRangeTree.cc \
	$(SRCDIR)/Patch.cc \
	$(SRCDIR)/Ghost.cc \
	$(SRCDIR)/SimulationState.cc \
	$(SRCDIR)/SimulationTime.cc \
	$(SRCDIR)/Task.cc \
	$(SRCDIR)/UnknownVariable.cc \
	$(SRCDIR)/SimpleMaterial.cc

PSELIBS := \
	Core/Geometry \
	Core/Exceptions \
	Core/Thread \
	Core/Util \
	Core/Containers 

LIBS       := $(MPI_LIBRARY) $(Z_LIBRARY)

