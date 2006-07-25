# Makefile fragment for this subdirectory
include $(SRCTOP)/scripts/smallso_prologue.mk


SRCDIR   := Packages/Kurt/Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/GLSLShader.cc \
	$(SRCDIR)/ParticleFlowRenderer.cc \
	$(SRCDIR)/ParticleFlow.cc\
#[INSERT NEW CODE FILE HERE]
# 	$(SRCDIR)/GridVolVis.cc \
# 	$(SRCDIR)/GridSliceVis.cc \
#	$(SRCDIR)/HarvardVis.cc \
# 	$(SRCDIR)/SCIRex.cc \
#	$(SRCDIR)/ParticleColorMapKey.cc \

PSELIBS := \
	Dataflow/Network \
	Dataflow/Modules/Visualization Core/Datatypes \
        Core/Thread Core/Persistent Core/Exceptions \
        Core/GuiInterface Core/Containers Core/Datatypes \
        Core/Geom Core/GeomInterface \
	Core/Geometry Dataflow/Widgets Core/XMLUtil \
	Core/Util \
	Packages/Kurt/Core/Geom \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/CCA/Ports       \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Datatypes

#Core/GLVolumeRenderer \
#Packages/Uintah/Dataflow \


LIBS := $(XML_LIBRARY)  $(GL_LIBRARY) $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
KURT_MODULES := $(KURT_MODULES) $(LIBNAME)
endif


