# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk


SRCDIR   := Packages/rtrt/Dataflow/Modules/Render

SRCS     += \
	$(SRCDIR)/RTRTViewer.cc \
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Dataflow/Network Dataflow/Ports \
	Dataflow/Modules/Visualization Core/Datatypes \
        Core/Thread Core/Persistent Core/Exceptions \
        Core/GuiInterface Core/Containers Core/Datatypes \
        Core/Geom \
	Core/Geometry Dataflow/Widgets Dataflow/XMLUtil \
	Core/Util \
	Packages/rtrt/Core \
	Packages/rtrt/Dataflow/Ports

LIBS := -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


