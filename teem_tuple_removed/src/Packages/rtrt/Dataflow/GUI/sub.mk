# Makefile fragment for this subdirectory

SRCDIR := Packages/rtrt/Dataflow/GUI

SRCS := \
	$(SRCDIR)/GeoProbeScene.tcl \
	$(SRCDIR)/RTRTViewer.tcl \
	$(SRCDIR)/SimpleScene.tcl \
	$(SRCDIR)/VolumeVisScene.tcl \
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk


