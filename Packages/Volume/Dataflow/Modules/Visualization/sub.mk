# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk


SRCDIR   := Packages/Volume/Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/TextureBuilder.cc \
	$(SRCDIR)/VolumeSlicer.cc \
	$(SRCDIR)/VolumeVisualizer.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Dataflow/Network \
	Dataflow/Ports \
	Core/Datatypes \
        Core/Thread Core/Persistent Core/Exceptions \
        Core/GuiInterface Core/Containers Core/Datatypes \
        Core/Geom Core/GeomInterface \
	Core/Geometry Dataflow/Widgets Dataflow/XMLUtil \
	Core/Util \
	Packages/Volume/Core/Algorithms \
	Packages/Volume/Core/Datatypes \
	Packages/Volume/Core/Geom \

LIBS := $(XML_LIBRARY)  $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
VOLUME_MODULES := $(VOLUME_MODULES) $(LIBNAME)
endif


