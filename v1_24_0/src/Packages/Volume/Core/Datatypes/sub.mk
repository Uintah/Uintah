# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Volume/Core/Datatypes

SRCS += \
	$(SRCDIR)/Brick.cc \
	$(SRCDIR)/Colormap2.cc \
	$(SRCDIR)/CM2Shader.cc \
	$(SRCDIR)/CM2Widget.cc \
	$(SRCDIR)/Texture.cc \
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Exceptions Core/Geometry \
	   Core/Persistent Core/Datatypes Core/Util \
	   Core/Containers Core/Geom Core/Thread \
	   Dataflow/Network Dataflow/XMLUtil \
	   Packages/Volume/Core/Util 

LIBS := $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
