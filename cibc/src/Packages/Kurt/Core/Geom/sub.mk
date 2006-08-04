#Makefile fragment for the Packages/Kurt/Core/Geom directory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Kurt/Core/Geom
SRCS   += \
	$(SRCDIR)/VolumeRenderer.cc \
	$(SRCDIR)/SCIRexRenderer.cc \
	$(SRCDIR)/OGLXVisual.cc \
	$(SRCDIR)/OGLXWindow.cc \
	$(SRCDIR)/SCIRexWindow.cc \
	$(SRCDIR)/SCIRexCompositer.cc \
	$(SRCDIR)/SliceRenderer.cc \
	$(SRCDIR)/BrickGrid.cc \
	$(SRCDIR)/GridBrick.cc \
	$(SRCDIR)/GridVolRen.cc \
	$(SRCDIR)/GridSliceRen.cc \
#[INSERT NEW CODE FILE HERE]


PSELIBS := \
	Core/Algorithms/GLVolumeRenderer \
	Core/Containers \
	Core/Datatypes \
	Core/Exceptions \
	Core/Geom \
	Core/Geometry \
	Core/Persistent \
	Core/GLVolumeRenderer \
	Core/Thread \
	Core/Util \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Datatypes \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Dataflow/Modules/Visualization

LIBS := $(LINK) $(XML_LIBRARY) $(GL_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
