# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk


SRCDIR   := Packages/Uintah/Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/GridVisualizer.cc \
	$(SRCDIR)/PatchVisualizer.cc \
	$(SRCDIR)/PatchDataVisualizer.cc \
	$(SRCDIR)/RescaleColorMapForParticles.cc \
	$(SRCDIR)/ParticleColorMapKey.cc \
	$(SRCDIR)/ParticleVis.cc \
	$(SRCDIR)/CuttingPlane.cc\
	$(SRCDIR)/FaceCuttingPlane.cc\
	$(SRCDIR)/Hedgehog.cc\
	$(SRCDIR)/AnimatedStreams.cc\
	$(SRCDIR)/VariablePlotter.cc \
	$(SRCDIR)/NodeHedgehog.cc \
	$(SRCDIR)/SubFieldHistogram.cc 
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Dataflow/Network               \
	Dataflow/Ports                 \
	Dataflow/Modules/Visualization \
	Dataflow/Widgets               \
	Core/Datatypes                 \
        Core/Thread                    \
	Core/Persistent                \
	Core/Exceptions                \
        Core/GuiInterface              \
	Core/Containers                \
	Core/Datatypes                 \
        Core/Geom                      \
        Core/GeomInterface             \
	Core/Geometry                  \
	Core/GLVolumeRenderer          \
	Core/Util                      \
	Packages/Uintah/Core/Grid          \
	Packages/Uintah/Core/Math          \
	Packages/Uintah/Core/Disclosure    \
	Packages/Uintah/CCA/Ports          \
	Packages/Uintah/Core/ProblemSpec   \
	Packages/Uintah/Core/Exceptions    \
	Packages/Uintah/Core/Datatypes     \
	Packages/Uintah/Core/DataArchive     \
	Packages/Uintah/Dataflow/Modules/Selectors

LIBS := $(XML_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


ifeq ($(LARGESOS),no)
UINTAH_SCIRUN := $(UINTAH_SCIRUN) $(LIBNAME)
endif
