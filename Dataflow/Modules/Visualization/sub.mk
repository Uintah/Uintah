# Makefile fragment for this subdirectory
include $(SRCTOP)/scripts/smallso_prologue.mk


SRCDIR   := Packages/Uintah/Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/GridVisualizer.cc \
	$(SRCDIR)/NodeHedgehog.cc \
	$(SRCDIR)/TimestepSelector.cc \
	$(SRCDIR)/ScalarFieldExtractor.cc \
	$(SRCDIR)/VectorFieldExtractor.cc \
	$(SRCDIR)/TensorFieldExtractor.cc \
	$(SRCDIR)/ParticleFieldExtractor.cc \
	$(SRCDIR)/RescaleColorMapForParticles.cc \
	$(SRCDIR)/ParticleVis.cc \
	$(SRCDIR)/EigenEvaluator.cc \
	$(SRCDIR)/ParticleEigenEvaluator.cc \
	$(SRCDIR)/TensorFieldOperator.cc \
	$(SRCDIR)/TensorParticlesOperator.cc

PSELIBS :=  Dataflow/Network Core/Datatypes \
        Core/Thread Core/Persistent Core/Exceptions \
        Core/TclInterface Core/Containers Core/Datatypes \
        Core/Geom Uintah/Grid Uintah/Interface Uintah/Exceptions \
	Core/Geometry Dataflow/Widgets PSECore/XMLUtil \
	Core/Util  Uintah/Components/MPM Uintah/Datatypes

LIBS := $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk


