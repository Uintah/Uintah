# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Uintah/Core/Datatypes

SRCS     += \
	$(SRCDIR)/Archive.cc \
	$(SRCDIR)/ScalarParticles.cc \
	$(SRCDIR)/VectorParticles.cc \
	$(SRCDIR)/TensorParticles.cc \
	$(SRCDIR)/PSet.cc \
	$(SRCDIR)/GLAnimatedStreams.cc \
	$(SRCDIR)/VariableCache.cc \
#	$(SRCDIR)/cd_templates.cc \

PSELIBS := \
	Dataflow/Network \
	Core/Exceptions  \
	Core/Geom        \
	Core/Geometry    \
	Core/GLVolumeRenderer \
	Core/Persistent  \
	Core/Datatypes   \
	Core/Containers  \
	Core/Thread      \
	Core/Util        \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Math        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports        \
        Packages/Uintah/Core/Exceptions  


LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


