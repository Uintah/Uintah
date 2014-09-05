# Makefile fragment for this subdirectory

SRCDIR := Packages/rtrt/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/GeoProbeScene.tcl \
	$(SRCDIR)/RTRTViewer.tcl \
	$(SRCDIR)/SimpleScene.tcl \
	$(SRCDIR)/VolumeVisScene.tcl \
#[INSERT NEW TCL FILE HERE]

	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/rtrt/Dataflow/GUI


CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

