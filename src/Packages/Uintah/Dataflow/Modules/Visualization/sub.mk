# Makefile fragment for this subdirectory
include $(SRCTOP)/scripts/smallso_prologue.mk


SRCDIR   := Packages/Uintah/Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/GridVisualizer.cc \
	$(SRCDIR)/TimestepSelector.cc \
	$(SRCDIR)/ScalarFieldExtractor.cc \
	$(SRCDIR)/VectorFieldExtractor.cc \
	$(SRCDIR)/TensorFieldExtractor.cc \
	$(SRCDIR)/ParticleFieldExtractor.cc \
	$(SRCDIR)/RescaleColorMapForParticles.cc \
	$(SRCDIR)/ParticleVis.cc \
	$(SRCDIR)/RescaleColorMap.cc \
	$(SRCDIR)/GLTextureBuilder.cc \
	$(SRCDIR)/Isosurface.cc \
	$(SRCDIR)/CuttingPlane.cc\
[INSERT NEW CODE FILE HERE]
#	$(SRCDIR)/NodeHedgehog.cc \
#	$(SRCDIR)/EigenEvaluator.cc \
#	$(SRCDIR)/ParticleEigenEvaluator.cc \
#	$(SRCDIR)/TensorFieldOperator.cc \
#	$(SRCDIR)/TensorParticlesOperator.cc

PSELIBS := \
	Dataflow/Network Dataflow/Ports \
	Dataflow/Modules/Visualization Core/Datatypes \
        Core/Thread Core/Persistent Core/Exceptions \
        Core/GuiInterface Core/Containers Core/Datatypes \
        Core/Geom \
	Core/Geometry Dataflow/Widgets Dataflow/XMLUtil \
	Core/Util \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/CCA/Ports       \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Datatypes   \
	Packages/Uintah/CCA/Components/MPM

LIBS := $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk


