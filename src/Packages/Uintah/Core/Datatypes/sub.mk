# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Uintah/Core/Datatypes

SRCS     += \
	$(SRCDIR)/Archive.cc \
	$(SRCDIR)/ArchivePort.cc \
	$(SRCDIR)/ScalarParticles.cc \
	$(SRCDIR)/ScalarParticlesPort.cc \
	$(SRCDIR)/VectorParticles.cc \
	$(SRCDIR)/VectorParticlesPort.cc \
	$(SRCDIR)/TensorParticles.cc \
	$(SRCDIR)/TensorParticlesPort.cc \
	$(SRCDIR)/PSet.cc \
	$(SRCDIR)/LevelMesh.cc \
	$(SRCDIR)/GLTexture3D.cc

#	$(SRCDIR)/NCTensorField.cc \
#	$(SRCDIR)/CCTensorField.cc \
#	$(SRCDIR)/TensorField.cc \
#	$(SRCDIR)/TensorFieldPort.cc \

#	$(SRCDIR)/CCVectorField.cc \
#	$(SRCDIR)/NCVectorField.cc \

PSELIBS := \
	Dataflow/Network \
	Dataflow/XMLUtil \
	Core/Exceptions  \
	Core/Geometry    \
	Core/Persistent  \
	Core/Datatypes   \
	Core/Containers  \
	Core/Thread      \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports       \
        Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/CCA/Components/MPM

LIBS := $(XML_LIBRARY)

ifeq ($(BUILD_PARALLEL),yes)
PSELIBS := $(PSELIBS) Core/CCA/Component/CIA Core/CCA/Component/PIDL
LIBS := $(LIBS) $(GLOBUS_LIBS) -lglobus_nexus
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


