# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Uintah/Dataflow/Ports

SRCS     += \
	$(SRCDIR)/ArchivePort.cc \
	$(SRCDIR)/ScalarParticlesPort.cc \
	$(SRCDIR)/VectorParticlesPort.cc \
	$(SRCDIR)/TensorParticlesPort.cc 

PSELIBS := \
	Dataflow/XMLUtil \
	Dataflow/Network \
	Dataflow/Comm \
	Core/Containers \
        Core/Thread \
	Core/Geom \
	Core/Geometry \
	Core/Exceptions \
        Core/Persistent \
	Core/Datatypes \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Datatypes 

LIBS := $(XML_LIBRARY)


include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


