# Makefile fragment for this subdirectory

SRCDIR := Packages/Kurt/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/VolVis.tcl $(SRCDIR)/PadField.tcl \
	$(SRCDIR)/TextureVolVis.tcl $(SRCDIR)/GLTextureBuilder.tcl \
	$(SRCDIR)/TexCuttingPlanes.tcl \
	$(SRCDIR)/VisControl.tcl \
	$(SRCDIR)/RescaleColorMapForParticles.tcl \
	$(SRCDIR)/ParticleVis.tcl \
	$(SRCDIR)/AnimatedStreams.tcl \
	$(SRCDIR)/SFRG.tcl

#[INSERT NEW TCL FILE HERE]
	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/Kurt/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

