# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Uintah/Core/Datatypes

SRCS     += \
	$(SRCDIR)/Archive.cc \
	$(SRCDIR)/ScalarParticles.cc \
	$(SRCDIR)/VectorParticles.cc \
	$(SRCDIR)/TensorParticles.cc \
	$(SRCDIR)/PSet.cc \
	$(SRCDIR)/LevelField.cc \
	$(SRCDIR)/LevelMesh.cc \
	$(SRCDIR)/GLTexture3D.cc \
	$(SRCDIR)/GLAnimatedStreams.cc \
	$(SRCDIR)/cd_templates.cc \

PSELIBS := \
	Dataflow/Network \
	Dataflow/XMLUtil \
	Core/Exceptions  \
	Core/Geom        \
	Core/Geometry    \
	Core/GLVolumeRenderer \
	Core/Persistent  \
	Core/Datatypes   \
	Core/Containers  \
	Core/Thread      \
	Core/Disclosure  \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Math        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports        \
        Packages/Uintah/Core/Exceptions  


LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(GL_LIBS) -lm 

ifeq ($(BUILD_PARALLEL),yes)
PSELIBS := $(PSELIBS) Core/CCA/Component/CIA Core/CCA/Component/PIDL
LIBS := $(LIBS) $(GLOBUS_LIBS) -lglobus_nexus 
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


