# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk


SRCDIR   := Packages/Volume/Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/EditTransferFunc2.cc \
	$(SRCDIR)/TextureBuilder.cc \
	$(SRCDIR)/VolumeVisualizer.cc \
	$(SRCDIR)/NrrdTextureBuilder.cc \
	$(SRCDIR)/VolumeSlicer.cc \
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Dataflow/Network \
	Dataflow/Ports \
	Core/Datatypes \
        Core/Thread Core/Persistent Core/Exceptions \
        Core/GuiInterface Core/Containers Core/Datatypes \
        Core/Geom Core/GeomInterface Core/TkExtensions \
	Core/Geometry Dataflow/Widgets Dataflow/XMLUtil \
	Core/Util \
	Core/TkExtensions \
	Packages/Volume/Core/Algorithms \
	Packages/Volume/Core/Datatypes \
	Packages/Volume/Core/Geom \
	Packages/Volume/Core/Util \

LIBS := $(XML_LIBRARY) $(TK_LIBRARY) $(GL_LIBRARY) $(TEEM_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
VOLUME_MODULES := $(VOLUME_MODULES) $(LIBNAME)
endif


