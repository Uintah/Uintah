# Makefile fragment for this subdirectory

SRCDIR := Packages/Volume/Dataflow/GUI

SRCS := \
        $(SRCDIR)/TextureBuilder.tcl \
        $(SRCDIR)/NrrdTextureBuilder.tcl \
        $(SRCDIR)/VolumeSlicer.tcl \
        $(SRCDIR)/VolumeVisualizer.tcl \
	$(SRCDIR)/GenTransferFunc2.tcl \
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk

VOLUME_MODULES := $(VOLUME_MODULES) $(TCLINDEX)

