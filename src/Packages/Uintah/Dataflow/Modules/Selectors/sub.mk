# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Dataflow/Modules/Selectors

SRCS     += \
	$(SRCDIR)/TimestepSelector.cc \
	$(SRCDIR)/ScalarFieldExtractor.cc \
	$(SRCDIR)/VectorFieldExtractor.cc \
	$(SRCDIR)/TensorFieldExtractor.cc \
	$(SRCDIR)/ParticleFieldExtractor.cc \
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Packages/Uintah/Core/Datatypes \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/CCA/Components/MPM \
	Dataflow/Network \
	Dataflow/Ports \
	Core/Containers \
	Core/Persistent \
	Core/Exceptions \
	Core/GuiInterface \
	Core/Thread \
	Core/Datatypes \
	Core/Geom
LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

