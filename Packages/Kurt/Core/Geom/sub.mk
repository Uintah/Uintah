#Makefile fragment for the Packages/Kurt/Core/Geom directory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Kurt/Core/Geom
SRCS   += \
	$(SRCDIR)/VolumeRenderer.cc \
	$(SRCDIR)/SliceRenderer.cc \
	$(SRCDIR)/BrickGrid.cc \
	$(SRCDIR)/GridBrick.cc \
	$(SRCDIR)/GridVolRen.cc \
	$(SRCDIR)/GridSliceRen.cc \
#[INSERT NEW CODE FILE HERE]


PSELIBS := Core/Exceptions Core/Geometry \
	Core/Datatypes \
	Core/Containers Core/Geom Core/Thread \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Datatypes \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Dataflow/Modules/Visualization

LIBS := $(LINK) $(XML_LIBRARY) $(GL_LIBS) -lmpi -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
