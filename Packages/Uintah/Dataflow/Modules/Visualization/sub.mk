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
	$(SRCDIR)/RescaleColorMap.cc \
	$(SRCDIR)/GLTextureBuilder.cc \
	$(SRCDIR)/CuttingPlane.cc\
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
	Dataflow/XMLUtil               \
	Core/Datatypes                 \
        Core/Thread                    \
	Core/Persistent                \
	Core/Exceptions                \
        Core/GuiInterface              \
	Core/Containers                \
	Core/Datatypes                 \
        Core/Geom                      \
	Core/Geometry                  \
	Core/GLVolumeRenderer          \
	Core/Util                      \
	Core/Disclosure                \
	Packages/Uintah/Core/Grid          \
	Packages/Uintah/Core/Math          \
	Packages/Uintah/Core/Disclosure    \
	Packages/Uintah/CCA/Ports          \
	Packages/Uintah/Core/ProblemSpec   \
	Packages/Uintah/Core/Exceptions    \
	Packages/Uintah/Core/Datatypes     \
	Packages/Uintah/Dataflow/Modules/Selectors

LIBS := $(XML_LIBRARY) -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


