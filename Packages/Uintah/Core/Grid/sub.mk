# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Grid

SUBDIRS := \
        $(SRCDIR)/BoundaryConditions \
        $(SRCDIR)/Variables \
        $(SRCDIR)/PatchBVH

include $(SCIRUN_SCRIPTS)/recurse.mk          

SRCS += \
        $(SRCDIR)/AMR.cc                   \
        $(SRCDIR)/Box.cc                   \
        $(SRCDIR)/BSplineInterpolator.cc   \
        $(SRCDIR)/Ghost.cc                 \
        $(SRCDIR)/Grid.cc                  \
        $(SRCDIR)/Level.cc                 \
        $(SRCDIR)/LinearInterpolator.cc    \
        $(SRCDIR)/Material.cc              \
        $(SRCDIR)/Node27Interpolator.cc    \
        $(SRCDIR)/PatchRangeTree.cc        \
        $(SRCDIR)/Patch.cc                 \
	$(SRCDIR)/Proc0Cout.cc             \
        $(SRCDIR)/Region.cc                \
        $(SRCDIR)/SimpleMaterial.cc        \
        $(SRCDIR)/SimulationState.cc       \
        $(SRCDIR)/SimulationTime.cc        \
        $(SRCDIR)/Task.cc                  \
        $(SRCDIR)/TOBSplineInterpolator.cc \
        $(SRCDIR)/UnknownVariable.cc       

PSELIBS := \
        Core/Geometry   \
        Core/Exceptions \
        Core/Thread     \
        Core/Util       \
        Core/Containers \
        Packages/Uintah/Core/Parallel    \
        Packages/Uintah/Core/ProblemSpec \
        Packages/Uintah/Core/Exceptions  \
        Packages/Uintah/Core/Util        \
        Packages/Uintah/Core/Math        \
        Packages/Uintah/Core/Disclosure

LIBS := $(MPI_LIBRARY) $(Z_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
