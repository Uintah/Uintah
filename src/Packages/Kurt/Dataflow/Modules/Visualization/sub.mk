# Makefile fragment for this subdirectory
include $(SRCTOP)/scripts/smallso_prologue.mk


SRCDIR   := Packages/Kurt/Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/GridVolVis.cc \
	$(SRCDIR)/GridSliceVis.cc \
	$(SRCDIR)/SCIRex.cc \
	$(SRCDIR)/HarvardVis.cc \
#[INSERT NEW CODE FILE HERE]
#	$(SRCDIR)/ParticleColorMapKey.cc \

PSELIBS := \
	Dataflow/Network Dataflow/Ports \
	Dataflow/Modules/Visualization Core/Datatypes \
        Core/Thread Core/Persistent Core/Exceptions \
        Core/GuiInterface Core/Containers Core/Datatypes \
        Core/Geom Core/GeomInterface Core/GLVolumeRenderer \
	Core/Geometry Dataflow/Widgets Dataflow/XMLUtil \
	Core/Util \
	Packages/Kurt/Core/Geom \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Dataflow/Ports \
	Packages/Uintah/CCA/Ports       \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Datatypes

LIBS := $(XML_LIBRARY)  $(GL_LIBRARY) $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk


