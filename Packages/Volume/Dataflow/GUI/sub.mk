# Makefile fragment for this subdirectory

SRCDIR := Packages/Volume/Dataflow/GUI

SRCS := \
	$(SRCDIR)/EditTransferFunc2.tcl \
        $(SRCDIR)/NrrdTextureBuilder.tcl \
        $(SRCDIR)/TextureBuilder.tcl \
        $(SRCDIR)/VolumeVisualizer.tcl \
        $(SRCDIR)/VolumeSlicer.tcl \
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk

VOLUME_MODULES := $(VOLUME_MODULES) $(TCLINDEX)

