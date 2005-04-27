# Makefile fragment for this subdirectory


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
	Core/Persistent  \
	Core/Datatypes   \
	Core/Containers  \
	Core/Thread      \
	Core/Util        


LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) 



