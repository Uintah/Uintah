# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk


SRCDIR   := Packages/rtrt/Dataflow/Modules/Scenes

SRCS     += \
	$(SRCDIR)/GeoProbeScene.cc \
	$(SRCDIR)/Scene0.cc \
	$(SRCDIR)/SimpleScene.cc \
#	$(SRCDIR)/VolumeVisScene.cc \
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Dataflow/Network Dataflow/Ports \
	Dataflow/Modules/Visualization Core/Datatypes \
        Core/Thread Core/Persistent Core/Exceptions \
        Core/GuiInterface Core/Containers Core/Datatypes \
        Core/Geom Core/GeomInterface \
	Core/Geometry Dataflow/Widgets Dataflow/XMLUtil \
	Core/Util \
	Packages/rtrt/Core \
	Packages/rtrt/Dataflow/Ports

LIBS := $(M_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

