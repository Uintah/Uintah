# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Grid

SRCS     += \
	$(SRCDIR)/Box.cc \
	$(SRCDIR)/Grid.cc \
	$(SRCDIR)/Level.cc $(SRCDIR)/Material.cc \
	$(SRCDIR)/PatchRangeTree.cc \
	$(SRCDIR)/Patch.cc \
	$(SRCDIR)/Ghost.cc \
	$(SRCDIR)/SimulationState.cc \
	$(SRCDIR)/SimulationTime.cc \
	$(SRCDIR)/Task.cc \
	$(SRCDIR)/SimpleMaterial.cc

PSELIBS := \
	Packages/Uintah/Core/BoundaryConditions \
	Packages/Uintah/Core/Variables \

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

