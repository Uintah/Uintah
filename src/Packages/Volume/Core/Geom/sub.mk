#Makefile fragment for the Packages/Volume/Core/Geom directory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Volume/Core/Geom
SRCS   += \
	$(SRCDIR)/TextureRenderer.cc \
	$(SRCDIR)/VolumeRenderer.cc \
	$(SRCDIR)/VolShader.cc \
	$(SRCDIR)/SliceRenderer.cc \
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Core/Containers \
	Core/Datatypes \
	Core/Exceptions \
	Core/Geom \
	Core/Geometry \
	Core/Persistent \
	Core/Thread \
	Core/Util \
	Packages/Volume/Core/Util \
	Packages/Volume/Core/Datatypes \

LIBS := $(LINK) $(XML_LIBRARY) $(GL_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
