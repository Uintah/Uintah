# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Packages/Uintah/Core/Datatypes

SRCS     += $(SRCDIR)/Archive.cc  $(SRCDIR)/ArchivePort.cc \
	$(SRCDIR)/NCVectorField.cc $(SRCDIR)/CCVectorField.cc \
	$(SRCDIR)/NCTensorField.cc $(SRCDIR)/CCTensorField.cc \
	$(SRCDIR)/TensorField.cc $(SRCDIR)/TensorFieldPort.cc \
	$(SRCDIR)/ScalarParticles.cc $(SRCDIR)/ScalarParticlesPort.cc \
	$(SRCDIR)/VectorParticles.cc $(SRCDIR)/VectorParticlesPort.cc \
	$(SRCDIR)/TensorParticles.cc $(SRCDIR)/TensorParticlesPort.cc \
	$(SRCDIR)/PSet.cc



PSELIBS := Core/Exceptions Core/Geometry \
	Core/Persistent Core/Datatypes \
	Core/Containers Core/Thread Uintah/Grid Uintah/Interface \
	Dataflow/Network \
        Uintah/Exceptions Dataflow/XMLUtil Uintah/Components/MPM

LIBS := $(XML_LIBRARY)

ifeq ($(BUILD_PARALLEL),yes)
PSELIBS := $(PSELIBS) Core/CCA/Component/CIA Core/CCA/Component/PIDL
LIBS := $(LIBS) $(GLOBUS_LIBS) -lglobus_nexus
endif

include $(SRCTOP)/scripts/smallso_epilogue.mk


